"""Exercise copied shell entrypoints without starting CUDA or model loading."""
import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hydra.core.override_parser.overrides_parser import OverridesParser


REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "train_scripts/train_flow2act_libero.sh"
PY_ENTRY = "tools/train_flow2act.py"


class EntryFlow2ActTests(unittest.TestCase):
    def setUp(self):
        work = Path(os.environ.get("FLOW2ACT_TEST_OUTPUT", tempfile.gettempdir()))
        work.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=work, prefix="entry_steps_flow2act_")
        self.root = Path(self.temporary.name)
        (self.root / "env/bin").mkdir(parents=True)
        (self.root / "env/bin/python").symlink_to(sys.executable)
        executable = self.root / "env/bin/torchrun"
        executable.write_text('''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
expected_stats = os.environ.get("EXPECT_STATS_BEFORE_LAUNCH")
if expected_stats:
    stats = json.loads(Path(expected_stats).read_text())
    assert len(stats["mean"]) == int(os.environ["ACTION_LENGTH"])
Path(os.environ["ARGV_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))
print("fake torchrun stdout", flush=True)
print("fake torchrun stderr", file=sys.stderr, flush=True)
sys.exit(int(os.environ.get("FAKE_EXIT_CODE", "0")))
''')
        executable.chmod(0o777)
        self.environment = dict(os.environ)
        for name in ("OUTPUT_DIR", "EXP_NAME", "STAGE", "NNODES", "NODE_RANK", "MASTER_ADDR",
                     "NPROC_PER_NODE", "RESOURCE_GPU", "RESUME_CHECKPOINT", "CUDA_VISIBLE_DEVICES",
                     "MLP_WORKER_NUM", "MLP_ROLE_INDEX", "MLP_WORKER_0_HOST", "MLP_WORKER_0_PORT",
                     "ACTION_LENGTH", "FUTURE_HORIZON", "EXECUTE_HORIZON", "NORM_STATS"):
            self.environment.pop(name, None)
        stats_root = self.root / "checkpoints/Flow2Act"
        stats_root.mkdir(parents=True)
        for horizon in (10, 30):
            (stats_root / f"libero4mix_H{horizon}_norm_stats_flow2act.json").write_text(
                json.dumps({"mean": [[0.0]*32]*horizon, "std": [[1.0]*32]*horizon}))
        self.environment["CHECKPOINT_ROOT"] = str(self.root / "checkpoints")
        self.environment.update(ENV_PATH=str(self.root / "env"), OUTPUT_ROOT=str(self.root / "output"),
                                ARGV_CAPTURE=str(self.root / "argv.json"))

    def tearDown(self):
        self.temporary.cleanup()

    def launch(self, overrides=None, arguments=()):
        environment = dict(self.environment, **(overrides or {}))
        result = subprocess.run(["bash", str(ENTRY), *arguments], env=environment,
                                text=True, capture_output=True, timeout=90)
        if result.returncode:
            return result, None
        raw = json.loads(Path(environment["ARGV_CAPTURE"]).read_text())
        values = {item.key_or_group: item.value() for item in
                  OverridesParser.create().parse_overrides(raw[raw.index(PY_ENTRY) + 1:])}
        return result, values

    def write_action_fixture(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        data = self.root / "libero_lerobot"
        for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial"):
            directory = data / f"{suite}_no_noops_1.0.0_lerobot"
            (directory / "meta").mkdir(parents=True)
            (directory / "data/chunk-000").mkdir(parents=True)
            info = dict(splits={"train": "0:1"}, total_episodes=1, chunks_size=1000,
                        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
            (directory / "meta/info.json").write_text(json.dumps(info))
            (directory / "meta/episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "length": 33}) + "\n")
            actions = [[float(frame + dimension) for dimension in range(6)] + [float(frame % 2)]
                       for frame in range(33)]
            pq.write_table(pa.table({"action": actions}),
                           directory / "data/chunk-000/episode_000000.parquet")
        return data

    def test_missing_statistics_are_generated_before_launch_for_matching_horizon(self):
        data = self.write_action_fixture()
        for horizon in (10, 30):
            with self.subTest(horizon=horizon):
                stats = self.root / f"checkpoints/Flow2Act/libero4mix_H{horizon}_norm_stats_flow2act.json"
                stats.unlink()
                result, values = self.launch(dict(
                    DATA_ROOT=str(data), ACTION_LENGTH=str(horizon),
                    FUTURE_HORIZON=str(horizon),
                    EXPECT_STATS_BEFORE_LAUNCH=str(stats)))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                generated = json.loads(stats.read_text())
                self.assertEqual(generated["action_length"], horizon)
                self.assertEqual(generated["num_windows"], 4 * (33 - horizon + 1))
                self.assertEqual(len(generated["mean"]), horizon)
                self.assertTrue(all(len(row) == 32 for row in generated["mean"]))
                self.assertAlmostEqual(generated["mean"][0][0], (33 - horizon) / 2)
                self.assertEqual(values["data.params.train_datasets.stats_path"], str(stats))
                self.assertEqual(result.stdout.count("complete action windows"), 4)

    def test_existing_statistics_are_reused_without_reading_dataset(self):
        stats = self.root / "checkpoints/Flow2Act/libero4mix_H30_norm_stats_flow2act.json"
        before = stats.read_bytes(), stats.stat().st_mtime_ns
        result, _ = self.launch(dict(DATA_ROOT=str(self.root / "absent_dataset")))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((stats.read_bytes(), stats.stat().st_mtime_ns), before)
        self.assertNotIn("complete action windows", result.stdout)

    def test_statistics_failure_is_logged_without_starting_training(self):
        stats = self.root / "checkpoints/Flow2Act/libero4mix_H30_norm_stats_flow2act.json"
        stats.unlink()
        output = self.root / "invalid_data"
        result, _ = self.launch(dict(DATA_ROOT=str(self.root / "absent_dataset"),
                                     OUTPUT_DIR=str(output)))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / "argv.json").exists())
        self.assertFalse(stats.exists())
        for name in ("log.log", "console_node0_flow2act.log"):
            self.assertIn("FileNotFoundError", (output / name).read_text())

    def test_only_node_zero_generates_shared_statistics(self):
        stats = self.root / "checkpoints/Flow2Act/libero4mix_H30_norm_stats_flow2act.json"
        stats.unlink()
        topology = dict(OUTPUT_DIR=str(self.root / "two_nodes"), NNODES="2",
                        MASTER_ADDR="127.0.0.1", NPROC_PER_NODE="4")
        result, _ = self.launch(dict(topology, NODE_RANK="1",
                                     DATA_ROOT=str(self.root / "absent_dataset")))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(stats.exists())
        self.assertNotIn("complete action windows", result.stdout)
        data = self.write_action_fixture()
        result, _ = self.launch(dict(topology, NODE_RANK="0", DATA_ROOT=str(data),
                                     ACTION_LENGTH="30", EXPECT_STATS_BEFORE_LAUNCH=str(stats)))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(stats.read_text())["num_windows"], 16)
        self.assertEqual(result.stdout.count("complete action windows"), 4)

    def test_default_topology_and_viz_are_explicit(self):
        result, values = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = Path(values["trainer.output_dir"])
        self.assertIn("motion3d-L1-ha30-hf30-m10-8gpu-gbs256-ga1", output.name)
        self.assertIn("lr1e-4-min5e-7-wlr5e-7-wu2000-steps30000-s42-fr4", output.name)
        self.assertEqual(values["model.params.model.stage"], "motion3d")
        self.assertIn("-geo0.02-m2d1.0-m3d0.1", output.name)
        self.assertEqual(values["data.params.train_datasets.batch_size"], 32)
        self.assertEqual(values["model.params.expected_world_size"], 8)
        self.assertEqual(values["model.params.model.action_shape"], [30, 32])
        self.assertEqual(values["data.params.train_datasets.action_length"], 30)
        self.assertEqual(values["model.params.model.future_horizon"], 30)
        self.assertEqual(values["model.params.model.execute_horizon"], 10)
        self.assertEqual(values["data.params.train_datasets.stats_path"],
                         str(self.root / "checkpoints/Flow2Act/libero4mix_H30_norm_stats_flow2act.json"))
        self.assertEqual(values["model.params.visualization.every_n_steps"], 0)
        self.assertIsNone(values["model.params.visualization.output_dir"])
        self.assertFalse((output / "visualizations").exists())
        self.assertEqual((output / "log.log").stat().st_mode & 0o777, 0o777)

    def test_explicit_h10_recipe_keeps_its_matching_stats(self):
        result, values = self.launch(dict(ACTION_LENGTH="10", FUTURE_HORIZON="10"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(values["model.params.model.action_shape"], [10, 32])
        self.assertIn("-ha10-hf10-m10-", values["trainer.output_dir"])
        self.assertEqual(values["data.params.train_datasets.stats_path"],
                         str(self.root / "checkpoints/Flow2Act/libero4mix_H10_norm_stats_flow2act.json"))

    def test_h10_stats_rejected_for_h30_before_output_or_launch(self):
        result, _ = self.launch(dict(NORM_STATS=str(
            self.root / "checkpoints/Flow2Act/libero4mix_H10_norm_stats_flow2act.json")))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Action statistics mismatch", result.stderr)
        self.assertIn("ACTION_LENGTH=30", result.stderr)
        self.assertFalse((self.root / "output").exists())
        self.assertFalse((self.root / "argv.json").exists())

    def test_recipe_changes_and_tag_change_default_directory(self):
        names = set()
        recipes = [dict(STAGE="static"), dict(STAGE="motion2d", NPROC_PER_NODE="4", GRAD_ACCUM="4"),
                   dict(STAGE="motion3d"), dict(STAGE="motion3d", MEMORY_LENGTH="3"),
                   dict(STAGE="static", LR="2e-4"), dict(STAGE="static", EXP_NAME="repeatA")]
        for recipe in recipes:
            result, values = self.launch(recipe)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = Path(values["trainer.output_dir"])
            self.assertNotIn(output.name, names)
            self.assertLessEqual(len(output.name.encode()), 255)
            names.add(output.name)
            if recipe["STAGE"] == "motion2d":
                self.assertIn("-4gpu-gbs256-ga4-", output.name)
                self.assertIn("-geo0.02-m2d1.0", output.name)
                self.assertNotIn("-m3d", output.name)
                self.assertEqual(values["data.params.train_datasets.batch_size"], 16)

    def test_existing_nonempty_directory_rejected_without_modification(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "config.yaml").write_text("stage: static\n")
        result, _ = self.launch(dict(OUTPUT_DIR=str(output), STAGE="motion2d"))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual((output / "config.yaml").read_text(), "stage: static\n")
        self.assertEqual(list(output.iterdir()), [output / "config.yaml"])
        self.assertFalse((self.root / "argv.json").exists())

    def test_explicit_empty_path_and_resume_quoting(self):
        output = self.root / "output=quoted space"
        output.mkdir()
        result, values = self.launch(dict(OUTPUT_DIR=str(output)))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(values["trainer.output_dir"], str(output))
        for resume in (str(self.root / "step-step=0000002_flow2act.ckpt"), "last", "best"):
            result, values = self.launch(dict(OUTPUT_DIR=str(output), RESUME_CHECKPOINT=resume))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(values["trainer.ckpt_path"], resume)

    def test_stdout_stderr_and_original_failure_are_preserved(self):
        output = self.root / "failure"
        result, _ = self.launch(dict(OUTPUT_DIR=str(output), FAKE_EXIT_CODE="7"))
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        for name in ("log.log", "console_node0_flow2act.log"):
            text = (output / name).read_text()
            self.assertIn("fake torchrun stdout", text)
            self.assertIn("fake torchrun stderr", text)

    def test_resume_rejects_different_saved_stage_before_log_write(self):
        output = self.root / "resume_wrong_stage"
        output.mkdir()
        config = output / "config_flow2act.yaml"
        text = "model:\n  params:\n    model:\n      stage: static\n"
        config.write_text(text)
        result, _ = self.launch(dict(OUTPUT_DIR=str(output), RESUME_CHECKPOINT="last", STAGE="motion2d"))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Resume configuration mismatch", result.stderr)
        self.assertEqual(config.read_text(), text)
        self.assertFalse((output / "log.log").exists())

    def test_active_output_kernel_lock_rejects_even_explicit_resume(self):
        output = self.root / "already_active"
        output.mkdir()
        with (output / "log.log").open("w") as handle:
            handle.write("Existing training console\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result, _ = self.launch(dict(OUTPUT_DIR=str(output), RESUME_CHECKPOINT="last"))
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("already owned by a running job", result.stderr)
            self.assertEqual((output / "log.log").read_text(), "Existing training console\n")
            self.assertFalse((output / "console_node0_flow2act.log").exists())

    def test_node_zero_accepts_other_nodes_bootstrap_only(self):
        output = self.root / "multinode"
        topology = dict(OUTPUT_DIR=str(output), NNODES="2", MASTER_ADDR="127.0.0.1", NPROC_PER_NODE="4")
        result, _ = self.launch(dict(topology, NODE_RANK="1"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((output / "log.log").exists())
        bootstrap = output / "console_node1_flow2act.log"
        bootstrap_text = bootstrap.read_text()
        (output / "torchrun_node1_flow2act").mkdir()
        result, values = self.launch(dict(topology, NODE_RANK="0"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(values["model.params.expected_world_size"], 8)
        self.assertEqual(bootstrap.read_text(), bootstrap_text)

    def test_conflicting_hydra_recipe_override_rejected_before_output(self):
        result, _ = self.launch(arguments=("model.params.model.stage=motion2d",))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("environment variable", result.stderr)
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
