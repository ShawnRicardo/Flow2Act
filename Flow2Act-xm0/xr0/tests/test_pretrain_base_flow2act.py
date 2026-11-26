"""Fresh initialization must use the Pretrain package; no weights are loaded."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "xr0")]
from mibot.utils.pretrain_base_flow2act import pretrain_base_directory_flow2act


class PretrainBaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "relocated-weights"
        self.path.mkdir()
        self.metadata = {
            "action_config": {
                name: {"mean": [[0.0]], "std": [[1.0]]}
                for name in ("agibot_pt_v3", "droid_pt", "midata_mix_ann_pt_v1", "molmoact_pt")
            }
        }

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, value):
        (self.path / "preprocessor_config.json").write_text(json.dumps(value))

    def test_complete_pretrain_metadata_accepts_a_relocated_directory(self):
        self.write(self.metadata)
        self.assertEqual(pretrain_base_directory_flow2act(self.path), self.path)

    def test_single_task_metadata_is_rejected_even_under_a_neutral_name(self):
        self.write({"action_config": {"downstream_task": {"mean": [[0.0]], "std": [[1.0]]}}})
        with self.assertRaisesRegex(ValueError, "Xiaomi-Robotics-0-Pretrain"):
            pretrain_base_directory_flow2act(self.path)

    def test_missing_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preprocessor_config.json"):
            pretrain_base_directory_flow2act(self.path)

    def test_malformed_metadata_is_rejected(self):
        (self.path / "preprocessor_config.json").write_text("{incomplete")
        with self.assertRaises(ValueError):
            pretrain_base_directory_flow2act(self.path)

    def test_non_object_or_incomplete_metadata_is_rejected(self):
        for value in (None, [], {}, {"action_config": []}):
            with self.subTest(value=value):
                self.write(value)
                with self.assertRaises(ValueError):
                    pretrain_base_directory_flow2act(self.path)

    def test_missing_statistics_are_rejected(self):
        self.metadata["action_config"]["droid_pt"] = {"mean": [[0.0]]}
        self.write(self.metadata)
        with self.assertRaises(ValueError):
            pretrain_base_directory_flow2act(self.path)

    def test_a_file_is_not_a_pretrain_directory(self):
        path = self.path / "model.pt"
        path.touch()
        with self.assertRaises(ValueError):
            pretrain_base_directory_flow2act(path)

    def test_runner_rejects_before_model_allocation_and_tensor_io(self):
        from mibot.models.runner import base_runner_flow2act as runner
        self.write({"action_config": {"downstream_task": {}}})
        dummy = SimpleNamespace(_pretrained=str(self.path))
        with patch.object(runner.MIMODEL, "build") as build, patch.object(runner, "load_file") as load:
            with self.assertRaises(ValueError):
                runner.BaseRunnerFlow2Act.configure_model(dummy)
            with self.assertRaises(ValueError):
                runner.BaseRunnerFlow2Act.load_pretrained(dummy, str(self.path))
            build.assert_not_called()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
