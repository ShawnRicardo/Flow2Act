"""Scientific contracts for the depth-only frozen-policy reconstruction path."""
from pathlib import Path
import copy
import math
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "xr0")]
from flow2act.depth_probe import (
    Flow2ActDepthProbeConfig, Flow2ActDepthProbe,
    depth_only_losses_flow2act, depth_metric_sums_flow2act,
    finalize_depth_metrics_flow2act, stratified_episode_split_flow2act,
    episode_frame_indices_flow2act, encode_probe_teacher_depthonly_flow2act,
    validate_depthonly_resume_flow2act,
)


class FrozenDepthHead(nn.Module):
    """A differentiable frozen decoder that uses all four feature taps."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(48)
        self.output = nn.Linear(48, 1)

    def forward(self, features):
        value = torch.stack([self.norm(feature) for feature in features]).mean(0)
        return (torch.nn.functional.softplus(self.output(value)) + 0.01).reshape(-1, 2, 6, 6)


class DepthOnlyProbeTestsFlow2Act(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_pure_depth_updates_all_four_heads_and_shared_spatial_trunk(self):
        probe = Flow2ActDepthProbe(Flow2ActDepthProbeConfig(24, 16, 48, 6))
        taps = [torch.randn(2, 2, 2, 2, 24, requires_grad=True) for _ in range(4)]
        predicted = probe(taps)
        self.assertEqual([tuple(value.shape) for value in predicted], [(2, 2, 36, 48)] * 4)
        self.assertEqual(sum(isinstance(module, nn.Conv2d) for module in probe.modules()), 2)
        head = FrozenDepthHead().eval().requires_grad_(False)
        head_before = {name: value.detach().clone() for name, value in head.state_dict().items()}
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        adapter_before = probe.channel.weight.detach().clone()
        depth = head(predicted)
        target = torch.ones_like(depth, requires_grad=True)
        confidence = torch.full_like(depth, 3, requires_grad=True)
        losses = depth_only_losses_flow2act(depth, target, confidence)
        self.assertEqual(set(losses), {"loss", "si_log", "multiscale_gradient", "log_depth"})
        losses["loss"].backward()
        for name in ("input_norm.weight", "channel.weight", "refine1.weight", "refine2.weight",
                     "heads.0.weight", "heads.1.weight", "heads.2.weight", "heads.3.weight"):
            gradient = dict(probe.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0, name)
        self.assertTrue(all(tap.grad is None for tap in taps))
        self.assertIsNone(target.grad)
        self.assertIsNone(confidence.grad)
        self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(adapter_before, probe.channel.weight))
        for name, value in head.state_dict().items():
            self.assertTrue(torch.equal(head_before[name], value), name)

    def test_global_scale_si_invariance_and_nonzero_absolute_log_penalty(self):
        target = torch.linspace(0.5, 2.0, 72).reshape(1, 2, 6, 6)
        confidence = torch.full_like(target, 3)
        same = depth_only_losses_flow2act(target * 2, target, confidence)
        self.assertAlmostEqual(float(same["si_log"]), 0, places=6)
        self.assertAlmostEqual(float(same["multiscale_gradient"]), 0, places=6)
        self.assertAlmostEqual(float(same["log_depth"]), math.log(2), places=6)
        self.assertAlmostEqual(float(same["loss"]), 0.1 * math.log(2), places=6)
        changed = target.clone()
        changed[:, 0] *= 2
        result = depth_only_losses_flow2act(changed, target, confidence)
        self.assertGreater(float(result["si_log"]), 0.1)
        only_scale = depth_only_losses_flow2act(
            target * 2, target, confidence, si_log_weight=0, gradient_weight=0, log_depth_weight=2)
        self.assertAlmostEqual(float(only_scale["loss"]), 2 * math.log(2), places=5)

    def test_tap_and_camera_order_are_not_mixed_by_shared_adapter(self):
        probe = Flow2ActDepthProbe(Flow2ActDepthProbeConfig(24, 16, 48, 6))
        taps = [torch.randn(2, 2, 2, 2, 24) for _ in range(4)]
        original = probe(taps)
        changed = [value.clone() for value in taps]
        changed[2][1, 0, :, :, 0] += 3
        output = probe(changed)
        self.assertFalse(torch.equal(original[2][1, 0], output[2][1, 0]))
        for index in range(4):
            if index != 2:
                torch.testing.assert_close(original[index], output[index], rtol=0, atol=0)
        torch.testing.assert_close(original[2][0], output[2][0], rtol=0, atol=0)
        torch.testing.assert_close(original[2][1, 1], output[2][1, 1], rtol=0, atol=0)

    def test_bounded_teacher_confidence_weight_is_used_inside_each_camera(self):
        target = torch.ones((1, 2, 6, 6))
        prediction = target.clone()
        confidence = torch.ones_like(target)
        prediction[0, 0, 0, 0] = 2
        confidence[0, 0, 0, 0] = 3  # Weight 2/3, erroneous pixel.
        confidence[0, 0, 0, 1] = 2  # Weight 1/2, perfect pixel.
        losses = depth_only_losses_flow2act(prediction, target, confidence)
        self.assertAlmostEqual(float(losses["log_depth"]), math.log(2) * 4 / 7, places=6)

    def test_invalid_teacher_values_are_masked_before_arithmetic(self):
        depth = torch.full((1, 2, 6, 6), 2.0, requires_grad=True)
        target = torch.ones_like(depth)
        confidence = torch.full_like(depth, 3)
        for index, value in enumerate((float("nan"), float("inf"), 0.0, -1.0)):
            target[0, 0, 0, index] = value
        confidence[0, 0, 1, :4] = torch.tensor([float("nan"), float("inf"), 1.0, 0.0])
        result = depth_only_losses_flow2act(depth, target, confidence)
        self.assertTrue(all(torch.isfinite(value) for value in result.values()))
        self.assertAlmostEqual(float(result["log_depth"].detach()), math.log(2), places=6)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertEqual(float(depth.grad[0, 0, 0, :4].abs().sum()), 0)
        self.assertEqual(float(depth.grad[0, 0, 1, :4].abs().sum()), 0)

    def test_all_invalid_is_graph_connected_zero(self):
        depth = torch.ones((2, 2, 6, 6), requires_grad=True)
        result = depth_only_losses_flow2act(
            depth, torch.full_like(depth, float("nan")), torch.zeros_like(depth))
        for value in result.values():
            self.assertTrue(value.requires_grad)
            self.assertEqual(float(value.detach()), 0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertEqual(float(depth.grad.abs().sum()), 0)

    def test_one_invalid_view_does_not_dilute_valid_view(self):
        target = torch.ones((1, 2, 6, 6))
        confidence = torch.ones_like(target)
        confidence[:, 0] = 3
        depth = target.clone()
        depth[:, 0] = 2
        depth[:, 1] = 100
        depth.requires_grad_()
        result = depth_only_losses_flow2act(depth, target, confidence)
        self.assertAlmostEqual(float(result["log_depth"].detach()), math.log(2), places=6)
        result["loss"].backward()
        self.assertEqual(float(depth.grad[:, 1].abs().sum()), 0)

    def test_sparse_teacher_without_neighbour_pairs_is_finite(self):
        target = torch.ones((1, 2, 6, 6))
        confidence = torch.ones_like(target)
        confidence[:, 0, 0, 0] = 3
        depth = (target * 2).requires_grad_()
        result = depth_only_losses_flow2act(depth, target, confidence)
        self.assertEqual(float(result["multiscale_gradient"].detach()), 0)
        self.assertAlmostEqual(float(result["log_depth"].detach()), math.log(2), places=6)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_camera_then_sample_equal_reduction(self):
        target = torch.ones((2, 2, 6, 6))
        depth = target.clone()
        confidence = torch.ones_like(target)
        depth[0, 0] = 2
        confidence[0, 0] = 3  # 36 pixels with log(2) error.
        confidence[0, 1, 0, 0] = 3  # One perfect pixel; still equal camera weight.
        confidence[1] = 3  # Second sample is perfect.
        losses = depth_only_losses_flow2act(depth, target, confidence)
        self.assertAlmostEqual(float(losses["log_depth"]), math.log(2) / 4, places=6)

    def test_batch_partitioning_preserves_empty_sample_weight(self):
        target = torch.ones((2, 2, 6, 6))
        confidence = torch.ones_like(target)
        confidence[0] = 3
        depth = (target * 2).requires_grad_()
        loss = depth_only_losses_flow2act(depth, target, confidence)["loss"]
        parts = [depth_only_losses_flow2act(depth[i:i+1], target[i:i+1], confidence[i:i+1])["loss"]
                 for i in range(2)]
        torch.testing.assert_close(loss, torch.stack(parts).mean())
        full_gradient = torch.autograd.grad(loss, depth, retain_graph=True)[0]
        split_gradient = torch.autograd.grad(torch.stack(parts).mean(), depth)[0]
        torch.testing.assert_close(full_gradient, split_gradient)

    def test_invalid_student_predictions_fail_even_outside_teacher_mask(self):
        for invalid in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(invalid=invalid):
                depth = torch.ones((1, 2, 6, 6))
                depth[0, 1, 0, 0] = invalid
                with self.assertRaises(FloatingPointError):
                    depth_only_losses_flow2act(depth, torch.ones_like(depth), torch.zeros_like(depth))

    def test_tiny_positive_student_depth_is_valid_with_finite_loss_and_gradient(self):
        for value in (1e-8, 1e-10, 1e-20):
            with self.subTest(value=value):
                target = torch.ones((1, 2, 6, 6))
                confidence = torch.full_like(target, 3)
                depth = torch.full_like(target, value)
                depth[0, 0, 0, 0] *= 2
                depth.requires_grad_()
                result = depth_only_losses_flow2act(depth, target, confidence)
                self.assertTrue(all(torch.isfinite(loss) for loss in result.values()))
                result["loss"].backward()
                self.assertTrue(torch.isfinite(depth.grad).all())
                self.assertGreater(float(depth.grad.abs().max()), 0)
                metrics = finalize_depth_metrics_flow2act(
                    depth_metric_sums_flow2act(depth, target, confidence))
                self.assertTrue(math.isfinite(metrics["depth_absrel_raw"]))
                self.assertTrue(math.isfinite(metrics["depth_rmse_aligned"]))

    def test_teacher_near_zero_threshold_remains_invalid(self):
        depth = torch.full((1, 2, 6, 6), 1e-10, requires_grad=True)
        target = torch.full_like(depth, 1e-8)
        confidence = torch.full_like(depth, 3)
        result = depth_only_losses_flow2act(depth, target, confidence)
        self.assertTrue(all(float(loss.detach()) == 0 for loss in result.values()))
        result["loss"].backward()
        self.assertEqual(float(depth.grad.abs().sum()), 0)
        metrics = finalize_depth_metrics_flow2act(
            depth_metric_sums_flow2act(depth, target, confidence))
        self.assertEqual(metrics["valid_depth_pixels"], 0)
        self.assertIsNone(metrics["depth_absrel_raw"])

    def test_normal_range_matches_original_relative_loss_values_and_gradients(self):
        from mibot.models.VLA.losses_flow2act import relative_depth_loss_flow2act
        target = torch.linspace(0.5, 2.5, 216).reshape(3, 2, 6, 6)
        confidence = torch.linspace(1.2, 5.0, 216).reshape_as(target)
        confidence[0, 0, :3, :3] = 1
        confidence[2] = 0  # Fixed sample denominator also matches for empty evidence.
        target[0, 0, 0, 0] = float("nan")
        depth_old = torch.linspace(0.02, 3.0, 216).reshape_as(target).requires_grad_()
        depth_new = depth_old.detach().clone().requires_grad_()
        original = relative_depth_loss_flow2act(
            depth_old, target, confidence, torch.ones((3, 2), dtype=torch.bool),
            gradient_weight=0.5, skip_empty_teacher_evidence=True)
        result = depth_only_losses_flow2act(depth_new, target, confidence, log_depth_weight=0)
        torch.testing.assert_close(result["si_log"], original.si_log, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(result["multiscale_gradient"], original.multiscale_gradient, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(result["loss"], original.loss, rtol=1e-6, atol=1e-7)
        original.loss.backward()
        result["loss"].backward()
        torch.testing.assert_close(depth_new.grad, depth_old.grad, rtol=1e-5, atol=1e-7)

    def test_teacher_encoder_retains_joint_float_resize_without_feature_targets(self):
        class Teacher:
            device = torch.device("cpu")
            image_mean = torch.zeros((1, 1, 3, 1, 1))
            image_std = torch.ones((1, 1, 3, 1, 1))

            def backbone(self, images, **kwargs):
                self.images, self.kwargs = images, kwargs
                self.grad_enabled = torch.is_grad_enabled()
                return [(torch.zeros(1, 2, 324, 48, requires_grad=True), None)] * 4, None

            def decode_student(self, features, grid, **kwargs):
                self.features, self.grid, self.decode_kwargs = features, grid, kwargs
                return {"depth": torch.ones(1, 2, 252, 252, requires_grad=True),
                        "confidence": torch.ones(1, 2, 252, 252, requires_grad=True) * 3}

        teacher = Teacher()
        raw = torch.randint(0, 256, (1, 2, 3, 256, 256), dtype=torch.uint8)
        result = encode_probe_teacher_depthonly_flow2act(teacher, raw)
        self.assertEqual(set(result), {"depth", "confidence"})
        self.assertEqual(tuple(teacher.images.shape), (1, 2, 3, 252, 252))
        self.assertEqual(teacher.images.dtype, torch.float32)
        self.assertFalse(teacher.grad_enabled)
        self.assertEqual(teacher.kwargs["ref_view_strategy"], "first")
        self.assertEqual(teacher.grid, (18, 18))
        self.assertEqual(teacher.decode_kwargs["output_hw"], (252, 252))
        self.assertFalse(any(value.requires_grad for value in result.values()))
        scaled = teacher.images * 255
        self.assertGreater(float((scaled - scaled.round()).abs().mean()), 0.01)

    def test_raw_and_shared_scale_aligned_metrics(self):
        target = torch.linspace(0.5, 2.0, 72).reshape(1, 2, 6, 6)
        confidence = torch.full_like(target, 3)
        metrics = finalize_depth_metrics_flow2act(
            depth_metric_sums_flow2act(target * 2, target, confidence))
        self.assertAlmostEqual(metrics["depth_absrel_raw"], 1, places=6)
        self.assertGreater(metrics["depth_rmse_raw"], 0)
        self.assertEqual(metrics["depth_delta1_raw"], 0)
        self.assertAlmostEqual(metrics["depth_absrel_aligned"], 0, places=6)
        self.assertAlmostEqual(metrics["depth_rmse_aligned"], 0, places=6)
        self.assertEqual(metrics["depth_delta1_aligned"], 1)
        self.assertEqual(metrics["valid_depth_pixels"], 72)
        for view in ("main", "wrist"):
            self.assertAlmostEqual(metrics["per_view"][view]["depth_absrel_raw"], 1, places=6)
        different = target.clone()
        different[:, 0] *= 2
        metrics = finalize_depth_metrics_flow2act(
            depth_metric_sums_flow2act(different, target, confidence))
        self.assertGreater(metrics["depth_absrel_aligned"], 0.1)
        empty = finalize_depth_metrics_flow2act(
            depth_metric_sums_flow2act(target, target, torch.zeros_like(target)))
        self.assertIsNone(empty["depth_absrel_raw"])
        self.assertEqual(empty["valid_depth_fraction"], 0)

    def test_episode_split_has_no_frame_leakage(self):
        episodes = [{"suite": suite, "task_index": task, "episode_index": i}
                    for suite in ("goal", "spatial") for task in (0, 1) for i in range(10)]
        train, val = stratified_episode_split_flow2act(episodes)
        self.assertEqual((len(train), len(val)), (36, 4))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(40)))
        self.assertEqual((train, val), stratified_episode_split_flow2act(episodes))
        ends = torch.arange(1, 41) * 5
        train_frames = set(episode_frame_indices_flow2act(train, ends))
        val_frames = set(episode_frame_indices_flow2act(val, ends))
        self.assertFalse(train_frames & val_frames)
        self.assertEqual(train_frames | val_frames, set(range(200)))

    def test_resume_rejects_old_kind_and_any_scientific_config_change(self):
        saved = {"kind": "independent_depth_adapter_depthonly_flow2act",
                 "policy_checkpoint": "/tmp/policyA.pt", "policy_global_step": 100,
                 "policy_contract": {"stage": "static", "student_taps": [17, 23, 29, 35]},
                 "probe_config": {"width": 512}, "split": {"seed": 42},
                 "training_config": {"log_depth_weight": 0.1, "lr": 1e-4}}
        args = {name: copy.deepcopy(saved[name]) for name in saved if name != "kind"}
        validate_depthonly_resume_flow2act(saved, **args)
        for key, new in (("policy_checkpoint", "/tmp/policyB.pt"), ("policy_global_step", 200),
                         ("policy_contract", {"stage": "motion3d"}), ("probe_config", {"width": 128}),
                         ("split", {"seed": 43}), ("training_config", {"log_depth_weight": 0})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_depthonly_resume_flow2act(saved, **{**args, key: new})
        with self.assertRaises(ValueError):
            validate_depthonly_resume_flow2act(
                {**saved, "kind": "independent_depth_probe_flow2act"}, **args)

    def test_probe_update_preserves_real_tiny_policy_actions(self):
        from xr0.tests.test_vlm_suffix_flow2act import tiny_policy, tiny_batch
        for stage, memory in (("baseline", 1), ("static", 1), ("motion3d", 3)):
            with self.subTest(stage=stage, memory=memory):
                policy = tiny_policy(stage=stage, memory_length=memory).eval().requires_grad_(False)
                batch = tiny_batch(1, memory)
                original = {name: value.detach().clone() for name, value in policy.state_dict().items()}
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(123)
                    before_action = policy.generate(batch).clone()
                    encoded = policy._encode_observation(batch, use_cache=False, feature_taps=policy.student_taps)
                    grids = tuple(policy._image_grids(batch, tap) for tap in encoded.hidden_states)
                probe = Flow2ActDepthProbe(Flow2ActDepthProbeConfig(64, 16, 48, 6))
                optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
                head = FrozenDepthHead().eval().requires_grad_(False)
                before = probe.channel.weight.detach().clone()
                depth = head(probe(grids))
                losses = depth_only_losses_flow2act(depth, torch.ones_like(depth), torch.full_like(depth, 3))
                losses["loss"].backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, probe.channel.weight))
                self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in head.parameters()))
                for name, value in policy.state_dict().items():
                    self.assertTrue(torch.equal(original[name], value), f"Frozen policy changed: {name}")
                with torch.random.fork_rng(devices=[]), torch.no_grad():
                    torch.manual_seed(123)
                    after_action = policy.generate(batch)
                torch.testing.assert_close(before_action, after_action, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
