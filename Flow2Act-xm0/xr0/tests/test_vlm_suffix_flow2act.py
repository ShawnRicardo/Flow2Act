"""Real tiny Qwen forward tests for causal learned suffixes (CPU, no weights)."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3VLConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "xr0")]
from mibot.models.VLM.qwen3vl_flow2act import Qwen3VLForConditionalGeneration
from mibot.models.VLA.flow2act import (
    Flow2ActXR0, DiT, MLPProjector, TimestepEmbedder, Qwen3VLTextRotaryEmbedding,
)


def tiny_config():
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
                         pad_token_id=0),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=64, patch_size=2, temporal_patch_size=2,
                           spatial_merge_size=2, deepstack_visual_indexes=[0, 1], num_position_embeddings=16),
        image_token_id=5, video_token_id=6, vision_start_token_id=7, vision_end_token_id=8,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return config


def tiny_policy(stage="motion3d", memory_length=1, *, action_horizon=3,
                future_horizon=3, execute_horizon=2):
    def build(model):
        model.vlm = Qwen3VLForConditionalGeneration(tiny_config()).to(torch.bfloat16)
        model.vlm.get_input_embeddings().requires_grad_(False)
        d = model.dit_hidden_size
        model.dit = DiT(d, layer_num=2, head_dim=16, kv_heads=2)
        model.state_projector = MLPProjector(32, d, 2)
        model.action_projector = MLPProjector(32, d, 2)
        model.action_output_layer = MLPProjector(d, 32, 2)
        model.t_embedder = TimestepEmbedder(d)
        model.t_projector = MLPProjector(d, 6*d, bias=True)
        model.rotary_emb = Qwen3VLTextRotaryEmbedding(model.vlm.config.text_config)
        model.sink = nn.Embedding(1, d)
        size = model.action_shape[0] + 2
        causal = torch.tril(torch.ones(size, size)).int()
        model.register_buffer("saved_causal_mask", causal[None], persistent=False)
        model.to(torch.bfloat16)
    with patch.object(Flow2ActXR0, "_build_model", build):
        return Flow2ActXR0(stage=stage, memory_length=memory_length,
            action_shape=(action_horizon, 32), future_horizon=future_horizon,
            execute_horizon=execute_horizon, dit_hidden_size=32,
            dit_num_layers=2, student_taps=(0, 1, 2, 3), training_repeat=1)


def tiny_batch(batch_size=2, memory_length=1, *, action_horizon=3,
               future_horizon=3, execute_horizon=2, stage=None):
    # Two real tiny vision inputs per sample, each merging four patches to one token.
    ids = torch.tensor([[9, 7, 5, 8, 7, 5, 8, 10, 11], [9, 7, 5, 8, 7, 5, 8, 10, 0]])[:batch_size]
    prefix = dict(input_ids=ids, attention_mask=(ids != 0).long(),
                  image_grid_thw=torch.tensor([[1, 2, 2]] * (batch_size*2)),
                  pixel_values=torch.randn(batch_size*2*4, 24, dtype=torch.bfloat16))
    result = {**prefix, "state": torch.randn(batch_size, 1, 32, dtype=torch.bfloat16),
              "action": torch.randn(batch_size, action_horizon, 32, dtype=torch.bfloat16),
              "action_mask": torch.zeros(batch_size, action_horizon, 32, dtype=torch.int32)}
    result["action_mask"][..., :7] = 1
    result["history_inputs"] = [copy.deepcopy(prefix) for _ in range(memory_length-1)]
    result["history_state"] = torch.randn(batch_size, memory_length-1, 1, 32, dtype=torch.bfloat16)
    result["history_valid"] = torch.ones(batch_size, memory_length-1, dtype=torch.bool)
    result["history_ages"] = torch.arange(1, memory_length)[None].expand(batch_size, -1)*execute_horizon
    if stage in ("motion2d", "motion3d"):
        result.update(
            motion_2d=torch.randn(batch_size, future_horizon, 2, 32, 32).view(torch.int32),
            valid_2d=torch.ones(batch_size, future_horizon, 32, 32, dtype=torch.bool))
    if stage == "motion3d":
        result.update(
            motion_3d=torch.randn(batch_size, future_horizon, 3, 32, 32).view(torch.int32),
            valid_3d=torch.ones(batch_size, future_horizon, 32, 32, dtype=torch.bool))
    return result


class SuffixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prefix_unchanged_queries_observe_prefix_and_backpropagate(self):
        torch.manual_seed(2)
        model = Qwen3VLForConditionalGeneration(tiny_config()).model.eval()
        ids = torch.tensor([[9, 10, 11, 12], [9, 10, 0, 0]])
        mask = (ids != 0).long()
        suffix = torch.randn(2, 3, 64, requires_grad=True)
        original = model(input_ids=ids, attention_mask=mask, feature_taps=(0, 1, 2, 3))
        encoded = model(input_ids=ids, attention_mask=mask, feature_taps=(0, 1, 2, 3),
                        suffix_embeds=suffix, future_query_count=2)
        torch.testing.assert_close(original.last_hidden_state[mask.bool()], encoded.last_hidden_state[:, :4][mask.bool()], atol=2e-6, rtol=2e-6)
        self.assertEqual((encoded.prefix_length, encoded.query_start, encoded.query_count), (4, 5, 2))
        self.assertEqual(encoded.augmented_attention_mask.tolist(), [[1,1,1,1,1,1,1], [1,1,0,0,1,1,1]])
        altered = model(input_ids=ids+1, attention_mask=mask, suffix_embeds=suffix, future_query_count=2)
        self.assertFalse(torch.allclose(encoded.last_hidden_state[:, -2:], altered.last_hidden_state[:, -2:]))
        encoded.last_hidden_state[:, -2:, 0].sum().backward()
        self.assertGreater(suffix.grad.abs().sum().item(), 0)
        self.assertGreater(model.language_model.layers[0].self_attn.k_proj.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(encoded.position_ids[:, 1, 4:], torch.tensor([[2,3,4]]).expand(3,-1))

    def test_real_vision_deepstack_and_filtered_action_cache(self):
        torch.manual_seed(3)
        model = tiny_policy(memory_length=3)
        batch = tiny_batch(memory_length=3)
        with torch.no_grad():
            plain = model._encode_observation(batch)
        suffix, valid = model._history_suffix(batch, batch["state"])
        encoded = model._encode_observation(batch, suffix_embeds=suffix, suffix_attention_mask=valid,
                                           future_query_count=3)
        for original, current in zip(plain.hidden_states, encoded.hidden_states):
            torch.testing.assert_close(model._image_grids(batch, original), model._image_grids(batch, current))
        cache, mask = model._action_cache(encoded)
        self.assertEqual(mask.shape, (2, batch["input_ids"].shape[1]+3))
        self.assertEqual(cache[0][0].shape[-2], mask.shape[1])
        q = encoded.last_hidden_state[:, encoded.query_start:]
        q.float().square().mean().backward()
        self.assertIsNotNone(model.geo_motion.future_queries.grad)
        self.assertIsNotNone(model.geo_motion.memory_projection.weight.grad)
        self.assertTrue(all(not item.requires_grad for item in batch["history_inputs"][0].values()))

    def test_explicit_positions_and_history_batch_axis(self):
        model = tiny_policy(memory_length=3)
        batch = tiny_batch(memory_length=3)
        width = batch["input_ids"].shape[1]
        explicit = torch.arange(width)[None].expand(2, -1)
        batch["position_ids"] = explicit
        for history in batch["history_inputs"]:
            history["position_ids"] = explicit[None].expand(3, -1, -1)
        suffix, valid = model._history_suffix(batch, batch["state"])
        result = model._encode_observation(batch, suffix_embeds=suffix,
            suffix_attention_mask=valid, future_query_count=3)
        self.assertEqual(result.position_ids.shape[:2], (3,2))

    def test_all_stage_forward_backward_and_inference(self):
        for stage, memory in (("baseline",1),("static",1),("motion2d",1),
                              ("motion3d",1),("motion3d",2),("motion3d",3),("motion3d",4)):
            with self.subTest(stage=stage, memory=memory):
                model = tiny_policy(stage, memory).train()
                batch = tiny_batch(1, memory)
                teacher = tuple(torch.randn(1,2,3,3,3072) for _ in range(4)) if stage != "baseline" else None
                if "motion" in stage:
                    batch.update(motion_2d=torch.randn(1,3,2,32,32).view(torch.int32), valid_2d=torch.ones(1,3,32,32,dtype=torch.bool))
                if stage == "motion3d":
                    batch.update(motion_3d=torch.randn(1,3,3,32,32).view(torch.int32), valid_3d=torch.ones(1,3,32,32,dtype=torch.bool))
                losses = model(batch, return_loss=True, teacher_features=teacher)
                self.assertTrue(torch.isfinite(losses["loss"]))
                losses["loss"].backward()
                self.assertIsNotNone(model.action_output_layer.layers[0].weight.grad)
                if "motion" in stage:
                    self.assertGreater(model.geo_motion.future_queries.grad.abs().sum().item(), 0)
                model.eval()
                self.assertEqual(model.generate(batch).shape, (1,3,32))

    def test_thirty_actions_ten_future_queries_and_ten_step_memory(self):
        for stage, memory in (("baseline", 1), ("motion3d", 2)):
            with self.subTest(stage=stage, memory=memory):
                model = tiny_policy(stage, memory, action_horizon=30,
                    future_horizon=10, execute_horizon=10).train()
                batch = tiny_batch(1, memory, stage=stage, action_horizon=30,
                    future_horizon=10, execute_horizon=10)
                self.assertEqual(model.action_shape, (30, 32))
                self.assertEqual(model.future_horizon, 10)
                self.assertEqual(model.execute_horizon, 10)
                teacher = None
                if stage == "motion3d":
                    teacher = tuple(torch.randn(1, 2, 3, 3, 3072) for _ in range(4))
                    self.assertEqual(model.geo_motion.future_queries.shape[0], 10)
                    self.assertEqual(batch["motion_2d"].shape, (1, 10, 2, 32, 32))
                    self.assertEqual(batch["motion_3d"].shape, (1, 10, 3, 32, 32))
                    self.assertEqual(batch["history_ages"].tolist(), [[10]])
                losses = model(batch, return_loss=True, teacher_features=teacher)
                self.assertTrue(torch.isfinite(losses["loss"]))
                losses["loss"].backward()
                action_grad = model.action_output_layer.layers[0].weight.grad
                self.assertIsNotNone(action_grad)
                self.assertTrue(torch.isfinite(action_grad).all())
                self.assertGreater(action_grad.abs().sum().item(), 0)
                if stage == "motion3d":
                    query_grad = model.geo_motion.future_queries.grad
                    self.assertIsNotNone(query_grad)
                    self.assertTrue(torch.isfinite(query_grad).all())
                    self.assertGreater(query_grad.abs().sum().item(), 0)
                model.eval()
                with torch.no_grad():
                    predicted = model.generate(batch)
                self.assertEqual(predicted.shape, (1, 30, 32))
                self.assertTrue(torch.isfinite(predicted).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA FlashAttention comparison")
    def test_flash_attention_ragged_prefix_and_invalid_memory(self):
        # Exercise the production attention backend with both right padding and
        # holes from unavailable history. Compare only unmasked token outputs.
        torch.manual_seed(17)
        eager = tiny_policy("motion3d", 3).cuda().eval()
        flash = copy.deepcopy(eager)
        for config in (flash.vlm.config, flash.vlm.config.text_config, flash.vlm.config.vision_config):
            config._attn_implementation = "flash_attention_2"
        batch = tiny_batch(memory_length=3)
        batch["history_valid"] = torch.tensor([[True, False], [False, False]])
        def move(value):
            if torch.is_tensor(value):
                return value.cuda()
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value
        batch = move(batch)
        with torch.no_grad():
            suffix, valid = eager._history_suffix(batch, batch["state"])
            expected = eager._encode_observation(batch, suffix_embeds=suffix,
                suffix_attention_mask=valid, future_query_count=3)
            actual = flash._encode_observation(batch, suffix_embeds=suffix,
                suffix_attention_mask=valid, future_query_count=3)
        mask = expected.augmented_attention_mask.bool()
        torch.testing.assert_close(actual.last_hidden_state[mask], expected.last_hidden_state[mask],
                                   atol=0.04, rtol=0.04)
        torch.testing.assert_close(actual.position_ids, expected.position_ids, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(actual.last_hidden_state[mask]).all())
        prefix_mask = batch["attention_mask"][1].bool()
        prefix_max = expected.position_ids[:, 1, :batch["input_ids"].shape[1]][:, prefix_mask].max()
        # Sample 1 has no valid history: only its current state precedes queries.
        torch.testing.assert_close(expected.position_ids[:, 1, -3:],
            (prefix_max + torch.arange(2, 5, device="cuda"))[None].expand(3, -1))


if __name__ == "__main__":
    unittest.main()
