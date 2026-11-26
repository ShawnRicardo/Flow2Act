"""Strict train/deploy temporal and metadata contracts without weights or GPUs."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "xr0")]
from deploy.server_flow2act import PolicyFlow2Act
from deploy.client_flow2act import resolve_policy_metadata_flow2act
from flow2act.memory import DecisionMemoryFlow2Act
from mibot.utils.checkpoint_contract_flow2act import read_checkpoint_contract_flow2act


class DeployFlow2Act(unittest.TestCase):
    def test_strict_loader_restores_tied_weights_and_real_buffers(self):
        from xr0.tests.test_vlm_suffix_flow2act import tiny_policy
        from deploy.server_flow2act import load_policy_flow2act
        from mibot.utils.checkpoint_contract_flow2act import (
            CHECKPOINT_KEY_FLOW2ACT, build_checkpoint_contract_flow2act)
        model = tiny_policy("motion3d", 3)
        model.vlm.config.tie_word_embeddings = True
        model.vlm.config.text_config.tie_word_embeddings = True
        model.vlm.tie_weights()
        contract = build_checkpoint_contract_flow2act(
            model_config={"type": "Flow2ActXR0", "stage": "motion3d", "memory_length": 3},
            method_config=model.geo_motion.config.to_dict(), vlm_config=model.vlm.config.to_dict(),
            experiment_config={"test": True}, action_stats={"mean": [], "std": [],
                "gripper_encoding": "signed_1_minus_2g_v1", "action_encoding": "test"})
        # DeepSpeed can serialize tied names as distinct tensor objects.
        payload = {"module": {"model."+key: value.clone() for key, value in model.state_dict().items()},
                   "global_step": 2, CHECKPOINT_KEY_FLOW2ACT: contract}
        module = "deploy.server_flow2act"
        with patch(module+".torch.load", return_value=payload), patch(module+".Flow2ActXR0", side_effect=lambda **_: copy.deepcopy(model)):
            loaded, _ = load_policy_flow2act("not-a-real-checkpoint.pt")
        self.assertIs(loaded.vlm.get_input_embeddings().weight, loaded.vlm.lm_head.weight)
        self.assertEqual(loaded.checkpoint_global_step, 2)
        self.assertFalse(any(value.requires_grad for value in loaded.parameters()))
        self.assertTrue(all(not value.is_meta and torch.isfinite(value).all() for value in loaded.buffers()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)

    def policy(self, length=3, stage="motion2d"):
        policy = PolicyFlow2Act.__new__(PolicyFlow2Act)
        policy.device = torch.device("cpu")
        policy.mean = torch.zeros(10, 32)
        policy.std = torch.ones(10, 32)
        policy.checkpoint_global_step = 12
        policy.checkpoint_kind = "trained_policy"
        policy.encoded_steps = []
        policy.generated = []

        def encode(observation):
            policy.encoded_steps.append(observation["step_index"])
            return {"input_ids": torch.tensor([[observation["step_index"]]]),
                    "attention_mask": torch.ones(1, 1, dtype=torch.long)}

        def generate(batch):
            policy.generated.append(batch)
            return torch.zeros_like(batch["action"])

        policy._encode = encode
        policy.model = SimpleNamespace(action_shape=(10, 32), execute_horizon=10, memory_length=length,
            future_horizon=10, stage=stage, generate=generate,
            geo_motion=SimpleNamespace(config=SimpleNamespace(motion_3d_enabled=stage == "motion3d")))
        return policy

    def observation(self, step, env="env", episode="ep"):
        return {"step_index": step, "state": np.full(32, step + 1, dtype=np.float32),
                "environment_id": env, "episode_id": episode}

    def request(self, length=3, step=20):
        request = {**self.observation(step), "execute_horizon": 10, "memory_length": length,
                   "decision_id": step // 10, "seed": 7, "history": []}
        for j in range(1, length):
            past = step - 10 * j
            if past >= 0:
                request["history"].append({"decision_step": past, "observation": self.observation(past)})
        return request

    def test_exact_offsets_and_only_episode_start_padding_for_n1_n2_n3_n4(self):
        for length in (1, 2, 3, 4):
            for step in (0, 5, 10, 20, 30, 50):
                with self.subTest(length=length, step=step):
                    policy = self.policy(length)
                    request = self.request(length, step)
                    response = policy(request)
                    self.assertEqual(response["actions"].shape, (1, 10, 32))
                    self.assertEqual(response["episode_id"], request["episode_id"])
                    self.assertEqual(response["decision_id"], request["decision_id"])
                    batch = policy.generated[0]
                    self.assertEqual(batch["history_state"].shape, (1, length - 1, 1, 32))
                    self.assertEqual(len(batch["history_inputs"]), length - 1)
                    for j in range(length - 1):
                        offset = 10 * (j + 1)
                        if step >= offset:
                            self.assertTrue(batch["history_valid"][0, j])
                            self.assertEqual(batch["history_ages"][0, j].item(), offset)
                            self.assertTrue(torch.equal(batch["history_state"][0, j, 0].float(),
                                                        torch.full((32,), float(step - offset + 1))))
                            self.assertEqual(batch["history_inputs"][j]["input_ids"].item(), step - offset)
                        else:
                            self.assertFalse(batch["history_valid"][0, j])
                            self.assertEqual(batch["history_ages"][0, j].item(), 0)
                            self.assertEqual(batch["history_state"][0, j].abs().sum().item(), 0)
                            self.assertEqual(batch["history_inputs"][j]["input_ids"].item(), step)
                    self.assertTrue(torch.equal(batch["action_mask"][..., :7], torch.ones(1, 10, 7, dtype=torch.int32)))
                    self.assertFalse(batch["action_mask"][..., 7:].any())

    def test_irregular_missing_reordered_duplicate_and_future_history_rejected(self):
        valid = self.request(3, 20)
        cases = []
        wrong_offset = copy.deepcopy(valid)
        wrong_offset["history"][0] = {"decision_step": 9, "observation": self.observation(9)}
        cases.append((wrong_offset, "trained offset"))
        missing = copy.deepcopy(valid); missing["history"].pop()
        cases.append((missing, "every available"))
        reordered = copy.deepcopy(valid); reordered["history"].reverse()
        cases.append((reordered, "trained offset"))
        duplicate = copy.deepcopy(valid); duplicate["history"][1] = copy.deepcopy(duplicate["history"][0])
        cases.append((duplicate, "unique past"))
        future = copy.deepcopy(valid)
        future["history"][0] = {"decision_step": 20, "observation": self.observation(20)}
        cases.append((future, "unique past"))
        for request, message in cases:
            policy = self.policy(3)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                policy(request)
            self.assertEqual(policy.generated, [])

    def test_history_never_crosses_episode_environment_or_observation_step(self):
        for key, wrong, error in (("episode_id", "other", "crossed"),
                                  ("environment_id", "other", "crossed"),
                                  ("step_index", 99, "own decision step")):
            policy = self.policy(3)
            request = self.request(3, 20)
            request["history"][0]["observation"][key] = wrong
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, error):
                policy(request)
            self.assertEqual(policy.generated, [])

    def test_client_execution_and_memory_mismatch_rejected(self):
        for key, wrong in (("execute_horizon", 5), ("memory_length", 2), ("step_index", -1)):
            policy = self.policy(3)
            request = self.request(3, 20); request[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                policy(request)
            self.assertEqual(policy.generated, [])
        policy = self.policy(1, "baseline")
        request = self.request(1, 10)
        request["history"] = [{"decision_step": 0, "observation": self.observation(0)}]
        with self.assertRaisesRegex(ValueError, "Too many"):
            policy(request)

    def test_fifo_and_server_agree_at_every_replan_and_reset(self):
        for length in (1, 2, 3, 4):
            policy = self.policy(length)
            memory = DecisionMemoryFlow2Act(max_length=length)
            for step in range(0, 51, 10):
                request = self.request(length, step)
                request["history"] = memory.history("env", "ep", step, length)
                policy(request)
                self.assertTrue(memory.commit("env", "ep", step, self.observation(step)))
                self.assertFalse(memory.commit("env", "ep", step, self.observation(step)))
            memory.reset("env", "ep")
            request = self.request(length, 0)
            request["episode_id"] = "new"
            request["history"] = memory.history("env", "new", 0, length)
            policy(request)

    def test_metadata_handshake_identifies_stage_and_rejects_legacy_schema(self):
        for stage, length in (("baseline", 1), ("static", 1), ("motion2d", 2), ("motion3d", 3), ("motion3d", 4)):
            policy = self.policy(length, stage)
            metadata = policy({"request_type": "metadata"})
            self.assertEqual(policy.encoded_steps, [])
            self.assertEqual(policy.generated, [])
            resolved = resolve_policy_metadata_flow2act(metadata, replan_steps=10, memory_length=length)
            self.assertEqual(resolved["architecture_version"], "Flow2Act-XR0-checkpoint-v1")
            self.assertEqual(resolved["stage"], stage)
            self.assertEqual(resolved["future_horizon"], 10)
            self.assertEqual(resolved["motion_3d_enabled"], stage == "motion3d")
            for key in ("architecture_version", "stage"):
                old = dict(metadata); old.pop(key)
                with self.subTest(stage=stage, missing=key), self.assertRaises(ValueError):
                    resolve_policy_metadata_flow2act(old)
            for version in ("_flow2act-XR0-checkpoint-v1", "official_Xiaomi-Robotics-0-LIBERO"):
                with self.assertRaises(ValueError):
                    resolve_policy_metadata_flow2act({**metadata, "architecture_version": version})
            with self.assertRaises(ValueError):
                resolve_policy_metadata_flow2act(metadata, replan_steps=5)
            with self.assertRaises(ValueError):
                resolve_policy_metadata_flow2act(metadata, memory_length=length+1)
        # Server loader's schema reader also refuses old unversioned/Geo payloads.
        for payload in ({}, {"flow2act": {"schema": "_flow2act-XR0-checkpoint-v1"}}):
            with self.assertRaisesRegex(ValueError, "import external artifacts"):
                read_checkpoint_contract_flow2act(payload)

    def test_metadata_rejects_inconsistent_horizon_stage_and_memory(self):
        metadata = self.policy(3, "motion2d")({"request_type": "metadata"})
        for value in (0, -1, 51, True, 10.5, "10", None):
            with self.subTest(future_horizon=value), self.assertRaisesRegex(ValueError, "future_horizon"):
                resolve_policy_metadata_flow2act({**metadata, "future_horizon": value})
        for key in ("future_horizon", "motion_3d_enabled"):
            missing = dict(metadata); missing.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                resolve_policy_metadata_flow2act(missing)
        for value in ("false", 0, None):
            with self.subTest(motion_flag=value), self.assertRaisesRegex(ValueError, "boolean"):
                resolve_policy_metadata_flow2act({**metadata, "motion_3d_enabled": value})
        for stage in ("baseline", "static", "motion2d", "motion3d"):
            valid_length = 1 if stage in ("baseline", "static") else 3
            valid = self.policy(valid_length, stage)({"request_type": "metadata"})
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "disagree"):
                resolve_policy_metadata_flow2act({**valid, "motion_3d_enabled": stage != "motion3d"})
            if stage in ("baseline", "static"):
                with self.assertRaisesRegex(ValueError, "memory_length=1"):
                    resolve_policy_metadata_flow2act({**valid, "memory_length": 3})
        for edge in (1, 50):
            self.assertEqual(resolve_policy_metadata_flow2act({**metadata, "future_horizon": edge})["future_horizon"], edge)

    def test_metadata_accepts_explicit_mechanical_kind_but_not_legacy_kind(self):
        policy = self.policy(1, "baseline")
        policy.checkpoint_kind = "mechanical_training_smoke"
        metadata = policy({"request_type": "metadata"})
        self.assertEqual(resolve_policy_metadata_flow2act(metadata)["checkpoint_kind"], "mechanical_training_smoke")
        for old_kind in ("native_zero_bridge_smoke", "unknown", None):
            with self.subTest(kind=old_kind), self.assertRaisesRegex(ValueError, "purpose"):
                resolve_policy_metadata_flow2act({**metadata, "checkpoint_kind": old_kind})


if __name__ == "__main__":
    unittest.main()
