"""Artifact imports rename metadata while preserving scientific tensors and masks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "xr0")]
from flow2act.cache import FILES_FLOW2ACT, SCHEMA_FLOW2ACT, MotionCacheFlow2Act
from flow2act.config import Flow2ActConfig
from mibot.utils.checkpoint_contract_flow2act import CHECKPOINT_KEY_FLOW2ACT, read_checkpoint_contract_flow2act
from scripts.import_flow2act_artifacts import (
    canonical_policy_contract, import_checkpoint, import_da3_weights, import_motion_cache,
)


def source_contract():
    method = Flow2ActConfig(stage="motion3d").to_dict()
    model = {"type": "PreviousResearchPolicy", "state_shape": [1, 32], "action_shape": [10, 32],
             **{name: method[name] for name in (
                 "stage", "future_horizon", "execute_horizon", "memory_length", "student_taps",
                 "geo_loss_weight", "motion_2d_loss_weight", "motion_3d_loss_weight")}}
    model = json.loads(json.dumps(model))
    return dict(schema="previous-policy-v1", kind="trained_policy", model_config=model,
                method_config=json.loads(json.dumps(method)), vlm_config={"text_config": {"hidden_size": 2560}},
                experiment_config={"model": {"type": "PreviousRunner", "params": {"model": copy.deepcopy(model)}},
                                   "data": {"type": "PreviousDataModule"}},
                action_stats={"mean": [[0.0] * 32 for _ in range(10)],
                              "std": [[1.0] * 7 + [0.0] * 25 for _ in range(10)],
                              "action_encoding": "libero_delta_xyz_axis_angle_gripper_signed_v1",
                              "gripper_encoding": "signed_1_minus_2g_v1"})


def cache_metadata(three_dimensional=False):
    return dict(schema="previous-native-uv-v1", horizon=2, image_hw=[256, 256], grid_hw=[32, 32],
                has_motion_3d=three_dimensional, coordinate="displacement_xy_div_wh",
                source_camera="observation.images.image", canonical_context_frames=128, anchor_ownership=78,
                decoder_sequence_length=16, decoder_stride=8, decoder_iterations=4,
                decoder_temporal_attention=True, decoder_overlap_warm_start=True, geometry_batch=1,
                geometry_real_frames_only=True, decoder_source_self_slot=True,
                visibility="native_sigmoid_visibility_and_confidence", endpoint_bounds_before_float16=True,
                optional_3d="native_UV_plus_original_DA3_depth_fullK_C2W_in_current_camera_div_anchor_median_depth",
                raw_xyz_flow_used=False)


class ArtifactImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_contract_is_unique_complete_and_source_is_unchanged(self):
        source = source_contract()
        untouched = copy.deepcopy(source)
        imported = canonical_policy_contract({"previous_contract": source})
        self.assertEqual(source, untouched)
        self.assertEqual(imported["schema"], "Flow2Act-XR0-checkpoint-v1")
        self.assertEqual(imported["model_config"]["type"], "Flow2ActXR0")
        self.assertEqual(imported["experiment_config"]["model"]["type"], "BaseRunnerFlow2Act")
        self.assertEqual(imported["experiment_config"]["model"]["params"]["model"]["type"], "Flow2ActXR0")
        self.assertEqual(imported["experiment_config"]["data"]["type"], "BaseDataModuleFlow2Act")
        self.assertEqual(imported["method_config"], source["method_config"])
        nested = canonical_policy_contract({"client_state": {"previous_contract": source}})
        self.assertEqual(imported, nested)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            canonical_policy_contract({"one": source, "two": copy.deepcopy(source)})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            canonical_policy_contract({"module": {"weight": torch.ones(1)}})

    def test_contract_rejects_changed_scientific_semantics(self):
        for field, replacement in (("query_location", "external_decoder"), ("static_loss", "smooth_l1"),
                                   ("motion_target", "raw_xyz_flow"), ("image_size", 224)):
            with self.subTest(field=field):
                source = source_contract()
                source["method_config"][field] = replacement
                with self.assertRaisesRegex(ValueError, "Unsupported source"):
                    canonical_policy_contract({"source": source})
        for mutate in (
            lambda item: item.update(kind="mechanical_training_smoke"),
            lambda item: item["model_config"].update(action_shape=[30, 32]),
            lambda item: item["model_config"].update(memory_length=3),
            lambda item: item["action_stats"].update(gripper_encoding="unsigned"),
            lambda item: item["method_config"].pop("static_alignment"),
            lambda item: item["vlm_config"]["text_config"].update(hidden_size=1024),
            lambda item: item["experiment_config"]["model"]["params"]["model"].update(future_horizon=30),
        ):
            source = source_contract()
            mutate(source)
            with self.assertRaises(ValueError):
                canonical_policy_contract({"source": source})

    def test_checkpoint_import_keeps_exact_tensors_and_discards_optimizer(self):
        tensors = {"model.vlm.test": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
                   "model.geo_motion.test": torch.tensor([0.25, -0.75])}
        source = self.root / "source.ckpt"
        torch.save({"module": tensors, "global_step": 30000, "previous_contract": source_contract(),
                    "optimizer": {"unwanted": torch.ones(100)}, "scheduler": {"step": 30000}}, source)
        source_bytes = source.read_bytes()
        output = import_checkpoint(source, self.root / "converted.ckpt")
        imported = torch.load(output, weights_only=False, mmap=True)
        self.assertEqual(set(imported), {"module", "global_step", CHECKPOINT_KEY_FLOW2ACT})
        self.assertEqual(imported["global_step"], 30000)
        read_checkpoint_contract_flow2act(imported, allowed_kinds={"trained_policy"})
        self.assertEqual(set(imported["module"]), set(tensors))
        for name, tensor in tensors.items():
            self.assertEqual(imported["module"][name].dtype, tensor.dtype)
            self.assertTrue(torch.equal(imported["module"][name], tensor))
        self.assertEqual(source.read_bytes(), source_bytes)
        with self.assertRaises(FileExistsError):
            import_checkpoint(source, output)
        with self.assertRaises(FileExistsError):
            import_checkpoint(source, source)

    def test_checkpoint_directory_and_nested_contract_are_supported(self):
        directory = self.root / "distributed"
        (directory / "checkpoint").mkdir(parents=True)
        torch.save({"state_dict": {"model.weight": torch.ones(2)}, "global_step": 5,
                    "client_state": {"source": source_contract()}},
                   directory / "checkpoint/mp_rank_00_model_states.pt")
        output = import_checkpoint(directory, self.root / "portable.ckpt")
        self.assertEqual(torch.load(output, weights_only=False)["global_step"], 5)
        torch.save({}, directory / "mp_rank_00_model_states.pt")
        with self.assertRaisesRegex(ValueError, "one model-state"):
            import_checkpoint(directory, self.root / "ambiguous.ckpt")

    def test_existing_dangling_output_link_is_not_followed(self):
        source = self.root / "source.ckpt"
        torch.save({}, source)
        destination = self.root / "existing.ckpt"
        destination.symlink_to(self.root / "missing-target.ckpt")
        with self.assertRaises(FileExistsError):
            import_checkpoint(source, destination)
        self.assertFalse((self.root / "missing-target.ckpt").exists())

    def make_cache(self, three_dimensional=False):
        source = self.root / "source-cache"
        episode = source / "libero_10/episode_000000"
        episode.mkdir(parents=True)
        (source / "metadata_previous.json").write_text(json.dumps(cache_metadata(three_dimensional)))
        arrays = {
            "displacement_normalized": np.full((2, 2, 2, 32, 32), 0.125, dtype=np.float16),
            "native_visconf": np.ones((2, 2, 2, 32, 32), dtype=np.float16),
            "in_bounds": np.ones((2, 2, 32, 32), dtype=np.bool_),
            "horizon_valid": np.asarray([[True, True], [True, False]]),
        }
        if three_dimensional:
            arrays.update(delta_xyz_normalized=np.full((2, 2, 3, 32, 32), 0.5, dtype=np.float16),
                          valid3d=np.ones((2, 2, 32, 32), dtype=np.bool_), anchor_scale=np.ones(2, dtype=np.float32))
        for name, array in arrays.items():
            np.save(episode / f"{name}_previous.npy", array)
        return source, episode

    def test_motion_cache_view_preserves_values_and_tail_masks(self):
        source, episode = self.make_cache(three_dimensional=True)
        previous_header = (source / "metadata_previous.json").read_text()
        destination = import_motion_cache(source, self.root / "new-cache")
        self.assertEqual((source / "metadata_previous.json").read_text(), previous_header)
        self.assertEqual(json.loads((destination / "metadata_flow2act.json").read_text())["schema"], SCHEMA_FLOW2ACT)
        cache = MotionCacheFlow2Act(destination, horizon=2, motion_3d_enabled=True)
        first = cache.load("10", 0, 0)
        tail = cache.load("10", 0, 1)
        self.assertTrue(np.all(first["motion_2d"] == 0.125))
        self.assertTrue(np.all(first["motion_3d"] == 0.5))
        self.assertTrue(first["valid_2d"].all())
        self.assertFalse(tail["valid_2d"][1].any())
        self.assertFalse(tail["valid_3d"][1].any())
        for filename in FILES_FLOW2ACT.values():
            self.assertTrue((destination / episode.relative_to(source) / filename).is_symlink())
        with self.assertRaises(FileExistsError):
            import_motion_cache(source, destination)
        with self.assertRaisesRegex(ValueError, "outside"):
            import_motion_cache(source, source / "nested-view")

    def test_motion_cache_hardlinks_and_ambiguous_inputs(self):
        source, episode = self.make_cache()
        destination = import_motion_cache(source, self.root / "linked-cache", link_mode="hardlink")
        linked = destination / "libero_10/episode_000000" / FILES_FLOW2ACT["motion_2d"]
        self.assertEqual(linked.stat().st_ino, (episode / "displacement_normalized_previous.npy").stat().st_ino)
        np.save(episode / "displacement_normalized_other.npy", np.zeros((2, 2, 2, 32, 32), dtype=np.float16))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            import_motion_cache(source, self.root / "ambiguous-cache")
        self.assertFalse((self.root / "ambiguous-cache").exists())

    def test_motion_cache_rejects_wrong_geometry_and_shapes(self):
        source, episode = self.make_cache()
        metadata = cache_metadata()
        metadata["coordinate"] = "endpoint_pixel"
        (source / "metadata_previous.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "coordinate"):
            import_motion_cache(source, self.root / "bad-coordinate")
        (source / "metadata_previous.json").write_text(json.dumps(cache_metadata()))
        np.save(episode / "horizon_valid_previous.npy", np.ones((2, 3), dtype=np.bool_))
        with self.assertRaisesRegex(ValueError, "Malformed horizon_valid"):
            import_motion_cache(source, self.root / "bad-shape")

    def test_da3_import_validates_headers_and_links_without_tensor_changes(self):
        source = self.root / "source-da3"
        source.mkdir()
        arrays = {"da3_anyview_giant": {"backbone.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
                  "da3_dualdpt": {"head.weight": torch.ones(3, dtype=torch.float32)}}
        specs = []
        for stem, tensors in arrays.items():
            save_file(tensors, source / f"{stem}_previous.safetensors")
            specs.append(SimpleNamespace(filename=f"{stem}_flow2act.safetensors", expected_key_count=1,
                                         expected_numel=sum(value.numel() for value in tensors.values()),
                                         expected_shapes={key: tuple(value.shape) for key, value in tensors.items()}))
        output = import_da3_weights(source, self.root / "new-da3", specs=specs)
        for spec in specs:
            self.assertTrue((output / spec.filename).is_symlink())
        specs[0].expected_numel = 100
        with self.assertRaisesRegex(ValueError, "dimensions"):
            import_da3_weights(source, self.root / "bad-da3", specs=specs)
        self.assertFalse((self.root / "bad-da3").exists())

    def test_cli_help_documents_three_explicit_import_commands(self):
        for command in ([], ["checkpoint"], ["motion-cache"], ["da3-weights"]):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, str(ROOT / "scripts/import_flow2act_artifacts.py"),
                                         *command, "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("source" if command else "checkpoint", result.stdout)


if __name__ == "__main__":
    unittest.main()
