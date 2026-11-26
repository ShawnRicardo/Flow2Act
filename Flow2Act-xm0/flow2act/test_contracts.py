"""CPU checks for the scientific boundaries of the Flow2Act policy modules."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .config import Flow2ActConfig
from .modules import Flow2ActGeometryMotion
from .losses import (
    downsample_teacher_features_flow2act, feature_loss_flow2act,
    motion_metrics_flow2act, trajectory_loss_flow2act,
)
from .memory import DecisionMemoryFlow2Act


torch.set_num_threads(2)


class Flow2ActContracts(unittest.TestCase):
    def config(self, **overrides):
        cfg = Flow2ActConfig(vlm_hidden_dim=24, state_dim=8, teacher_dim=48,
                                       adapter_hidden_dim=16, width=16, readout_hidden_dim=8,
                                       future_horizon=3, motion_grid_size=4, readout_horizon_chunk=2)
        return replace(cfg, **overrides)

    def grids(self, requires_grad=False):
        return tuple(torch.randn(2, 2, 2, 2, 24, requires_grad=requires_grad) for _ in range(4))

    def teachers(self, requires_grad=False):
        return tuple(torch.randn(2, 2, 5, 5, 48, requires_grad=requires_grad) for _ in range(4))

    def test_ablation_modes_create_only_authorized_parameters(self):
        base = Flow2ActGeometryMotion(self.config(stage="baseline"))
        self.assertEqual(sum(p.numel() for p in base.parameters()), 0)
        self.assertEqual(base.static_predictions(()), ())
        static = Flow2ActGeometryMotion(self.config())
        names = dict(static.named_parameters())
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("shared_adapter.") for name in names))
        self.assertFalse(hasattr(static, "future_queries"))
        for stage, expected3 in (("motion2d", False), ("motion3d", True)):
            module = Flow2ActGeometryMotion(self.config(stage=stage, memory_length=3))
            self.assertEqual(module.future_queries.shape, (3, 24))
            self.assertEqual(hasattr(module, "motion_3d_head"), expected3)
            self.assertFalse(hasattr(module, "future_decoder"))
            self.assertFalse(hasattr(module, "action_attention"))

    def test_arbitrary_memory_length_and_stage_validation(self):
        for count in (1, 2, 3, 5, 9):
            cfg = self.config(stage="motion2d", memory_length=count)
            self.assertEqual(cfg.history_offsets, tuple(10 * j for j in range(1, count)))
        for override in ({"memory_length": 3}, {"memory_length": 0}, {"future_horizon": 51},
                         {"geo_loss_weight": float("nan")}, {"stage": "unknown"}):
            with self.assertRaises(ValueError):
                self.config(**override)

    def test_native_grid_whole_adapter_is_shared_and_all_taps_receive_gradients(self):
        module = Flow2ActGeometryMotion(self.config())
        grids, teachers = self.grids(True), self.teachers(True)
        with patch("flow2act.modules.F.interpolate", side_effect=AssertionError("No student resize")):
            predictions = module.static_predictions(grids)
        self.assertEqual(len(predictions), 4)
        self.assertTrue(all(value.shape == (2, 2, 2, 2, 48) for value in predictions))
        identical = module.static_predictions((grids[0],) * 4)
        for value in identical[1:]:
            torch.testing.assert_close(value, identical[0], atol=0, rtol=0)
        losses = module.auxiliary_losses(grids, teachers, {}, {})
        self.assertEqual(set(losses), {"loss", "loss_geo"})
        losses["loss"].backward()
        self.assertTrue(all(grid.grad is not None and grid.grad.abs().sum() > 0 for grid in grids))
        self.assertTrue(all(target.grad is None for target in teachers))
        self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))

    def test_teacher_pool_is_fixed_antialiased_and_loss_is_pure_cosine(self):
        target = torch.randn(1, 2, 5, 5, 6, requires_grad=True)
        expected = F.interpolate(target.detach().permute(0, 1, 4, 2, 3).reshape(2, 6, 5, 5),
                                 (2, 2), mode="bilinear", align_corners=False, antialias=True)
        expected = expected.reshape(1, 2, 6, 2, 2).permute(0, 1, 3, 4, 2)
        actual = downsample_teacher_features_flow2act(target, (2, 2))
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)
        flat = target.reshape(1, 2, 25, 6)
        torch.testing.assert_close(downsample_teacher_features_flow2act(flat, (2, 2)), actual)
        p = (actual * 7).requires_grad_()
        loss = feature_loss_flow2act((p,) * 4, (target,) * 4)
        self.assertLess(abs(float(loss)), 1e-6)
        loss.backward()
        self.assertIsNone(target.grad)
        opposite = feature_loss_flow2act((-actual,) * 4, (target,) * 4)
        self.assertAlmostEqual(float(opposite), 2, places=6)

    def test_camera_correspondence_and_equal_camera_reduction(self):
        p = torch.tensor([[[[[1., 0.]]]], [[[[0., 1.]]]]]).transpose(0, 1)
        teacher = p.detach().clone()
        good = feature_loss_flow2act((p,) * 4, (teacher,) * 4)
        swapped = feature_loss_flow2act((p,) * 4, (teacher.flip(1),) * 4)
        self.assertAlmostEqual(float(good), 0)
        self.assertAlmostEqual(float(swapped), 1)
        bad = teacher.clone(); bad[:, 1] *= -1
        both = feature_loss_flow2act((p,) * 4, (bad,) * 4)
        main = feature_loss_flow2act((p,) * 4, (bad,) * 4, torch.tensor([[True, False]]))
        self.assertAlmostEqual(float(both), 1)
        self.assertAlmostEqual(float(main), 0)

    def test_suffix_preserves_states_queries_and_masks_missing_history(self):
        module = Flow2ActGeometryMotion(self.config(stage="motion2d", memory_length=3))
        history = [torch.randn(2, 2, 2, 2, 24, requires_grad=True) for _ in range(2)]
        history[1].data[0].fill_(float("nan"))
        old_states = torch.randn(2, 2, 8, requires_grad=True)
        old_states.data[0, 1].fill_(float("nan"))
        valid = torch.tensor([[True, False], [True, True]])
        ages = torch.tensor([[10, 0], [10, 20]])
        current = torch.randn(2, 8, requires_grad=True)
        suffix, mask = module.build_suffix(history, old_states, valid, ages, current)
        # Each history holds two 2x2 images plus its own state, then current state and HF queries.
        self.assertEqual(suffix.shape, (2, 22, 24))
        self.assertEqual(mask.shape, (2, 22))
        self.assertTrue(mask[0, :9].all()); self.assertFalse(mask[0, 9:18].any())
        self.assertTrue(mask[:, -4:].all())
        self.assertTrue(torch.isfinite(suffix).all())
        self.assertEqual(float(suffix[0, 9:18].detach().abs().sum()), 0)
        torch.testing.assert_close(suffix[:, -3:], module.future_queries[None].expand(2, -1, -1))
        suffix.square().sum().backward()
        self.assertGreater(float(history[0].grad.abs().sum()), 0)
        self.assertEqual(float(history[1].grad[0].abs().sum()), 0)
        self.assertEqual(float(old_states.grad[0, 1].abs().sum()), 0)
        self.assertGreater(float(current.grad.abs().sum()), 0)
        self.assertIsNotNone(module.future_queries.grad)
        with self.assertRaises(ValueError):
            module.build_suffix(history, old_states, valid, torch.zeros_like(ages), current)

    def test_motion_readout_horizon_shapes_and_gradient(self):
        for stage in ("motion2d", "motion3d"):
            module = Flow2ActGeometryMotion(self.config(stage=stage))
            query = torch.randn(2, 3, 24, requires_grad=True)
            grid = torch.randn(2, 2, 2, 2, 24, requires_grad=True)
            out = module.read_motion(query, grid)
            self.assertEqual(out["motion_2d"].shape, (2, 3, 2, 4, 4))
            if stage == "motion3d":
                self.assertEqual(out["motion_3d"].shape, (2, 3, 3, 4, 4))
            sum(value.square().mean() for value in out.values()).backward()
            self.assertTrue((query.grad.abs().sum((0, 2)) > 0).all())
            self.assertGreater(float(grid.grad[:, 0].abs().sum()), 0)
            self.assertEqual(float(grid.grad[:, 1].abs().sum()), 0)

    def test_tail_mask_nan_sanitization_and_all_empty_zero_gradient(self):
        p = torch.randn(2, 3, 2, 4, 4, requires_grad=True)
        t = torch.randn_like(p, requires_grad=True)
        t.data[:, 2] = float("nan")
        valid = torch.ones(2, 3, 4, 4, dtype=torch.bool); valid[:, 2] = False
        loss = trajectory_loss_flow2act(p, t, valid, beta=1/256, moving_target=t)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(float(p.grad[:, 2].abs().sum()), 0)
        self.assertIsNone(t.grad)
        p.grad.zero_()
        empty = trajectory_loss_flow2act(p, t, torch.zeros_like(valid), beta=1/256, moving_target=t)
        self.assertEqual(float(empty), 0)
        empty.backward()
        self.assertEqual(float(p.grad.abs().sum()), 0)

    def test_auxiliary_weights_and_detached_reference_metrics(self):
        module = Flow2ActGeometryMotion(self.config(stage="motion3d"))
        predictions = {"motion_2d": torch.randn(2, 3, 2, 4, 4, requires_grad=True),
                       "motion_3d": torch.randn(2, 3, 3, 4, 4, requires_grad=True)}
        targets = {"motion_2d": torch.randn(2, 3, 2, 4, 4), "motion_3d": torch.randn(2, 3, 3, 4, 4),
                   "valid_2d": torch.ones(2, 3, 4, 4, dtype=torch.bool),
                   "valid_3d": torch.ones(2, 3, 4, 4, dtype=torch.bool)}
        out = module.auxiliary_losses(self.grids(True), self.teachers(), predictions, targets)
        expected = .02 * out["loss_geo"] + out["loss_motion_2d"] + .1 * out["loss_motion_3d"]
        torch.testing.assert_close(out["loss"], expected)
        metrics = {key: value for key, value in out.items() if key.startswith("metric_")}
        self.assertTrue(metrics)
        self.assertTrue(all(not value.requires_grad and torch.isfinite(value) for value in metrics.values()))
        self.assertIn("metric_motion_2d_moving_epe_px_h03", metrics)
        self.assertIn("metric_valid_3d_fraction_h03", metrics)
        out["loss"].backward()
        self.assertGreater(float(predictions["motion_2d"].grad.abs().sum()), 0)
        self.assertGreater(float(predictions["motion_3d"].grad.abs().sum()), 0)

    def test_metric_units_missing_population_counts_and_validity(self):
        p = torch.zeros(1, 2, 2, 1, 1, requires_grad=True)
        t = torch.zeros_like(p); t.data[:, 0, 0] = 2/256
        valid = torch.ones(1, 2, 1, 1, dtype=torch.bool)
        metrics = motion_metrics_flow2act({"motion_2d": p}, {"motion_2d": t, "valid_2d": valid})
        self.assertAlmostEqual(float(metrics["metric_motion_2d_epe_px_h01"]), 2)
        self.assertAlmostEqual(float(metrics["metric_motion_2d_epe_px"]), 1)
        self.assertEqual(float(metrics["metric_motion_2d_moving_epe_px_h02_count"]), 0)
        self.assertEqual(float(metrics["metric_motion_2d_direction_coverage"]), 0)
        self.assertEqual(float(metrics["metric_motion_2d_zero_prediction_fraction"]), 1)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_memory_supports_three_and_episode_isolation(self):
        memory = DecisionMemoryFlow2Act(max_length=3)
        memory.commit("a", "ep1", 0, {"state": [0]})
        memory.commit("a", "ep1", 10, {"state": [1]})
        self.assertFalse(memory.commit("a", "ep1", 10, {"state": [999]}))
        memory.commit("b", "ep1", 0, {"state": [2]})
        self.assertEqual([r["decision_step"] for r in memory.history("a", "ep1", 20, 3)], [10, 0])
        self.assertEqual(memory.history("a", "ep2", 20, 3), [])
        self.assertEqual(len(memory.history("b", "ep1", 20, 3)), 1)
        history = memory.history("a", "ep1", 20, 3); history[0]["observation"]["state"][0] = 999
        self.assertEqual(memory.history("a", "ep1", 20, 3)[0]["observation"]["state"], [1])
        memory.reset("a", "ep1")
        self.assertEqual(memory.history("a", "ep1", 20, 3), [])


if __name__ == "__main__":
    unittest.main()
