"""Check depth-only shell routing without loading CUDA, policy or DA3."""
import fcntl
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch


REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "train_scripts/train_flow2act_depth_probe.sh"
PY_ENTRY = str(REPO / "xr0/tools/train_flow2act_depth_probe.py")


class DepthOnlyEntryTestsFlow2Act(unittest.TestCase):
    def setUp(self):
        # Tiny disposable launcher fixtures avoid EFS directory-cache delays;
        # no model, dataset or persistent training artifact is written here.
        self.temporary = tempfile.TemporaryDirectory(prefix="entry_depthonly_flow2act_")
        self.root = Path(self.temporary.name)
        (self.root / "env/bin").mkdir(parents=True)
        prepare = self.root / "env/bin/python"
        prepare.write_text('''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
Path(os.environ["PREPARE_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))
if int(os.environ.get("FAKE_PREPARE_EXIT", "0")):
    print("prepare-test-stderr", file=sys.stderr, flush=True)
    sys.exit(int(os.environ["FAKE_PREPARE_EXIT"]))
output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
output.mkdir(parents=True, exist_ok=True)
print(output.resolve())
''')
        prepare.chmod(0o777)
        executable = self.root / "env/bin/torchrun"
        executable.write_text('''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
Path(os.environ["ARGV_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))
print("fake-depthonly-stdout", flush=True)
print("fake-depthonly-stderr", file=sys.stderr, flush=True)
sys.exit(int(os.environ.get("FAKE_EXIT_CODE", "0")))
''')
        executable.chmod(0o777)
        self.environment = dict(os.environ)
        for name in ("OUTPUT_DIR", "NNODES", "NODE_RANK", "MASTER_ADDR", "NPROC_PER_NODE", "RESOURCE_GPU",
                     "CUDA_VISIBLE_DEVICES", "MLP_WORKER_NUM", "MLP_ROLE_INDEX", "MLP_WORKER_0_HOST",
                     "MLP_WORKER_0_PORT", "GLOBAL_BATCH_SIZE", "GRAD_ACCUM"):
            self.environment.pop(name, None)
        self.environment.update(ENV_PATH=str(self.root / "env"), OUTPUT_DIR=str(self.root / "output"),
                                POLICY_CHECKPOINT=str(self.root / "source-policy.ckpt"),
                                ARGV_CAPTURE=str(self.root / "argv.json"),
                                PREPARE_CAPTURE=str(self.root / "prepare.json"))

    def tearDown(self):
        self.temporary.cleanup()

    def launch(self, overrides=None, arguments=()):
        return subprocess.run(["bash", str(ENTRY), *arguments], env=dict(self.environment, **(overrides or {})),
                              text=True, capture_output=True, timeout=60)

    def arguments(self):
        values = json.loads((self.root / "argv.json").read_text())
        index = values.index(PY_ENTRY)
        return values[:index], values[index+1:]

    def test_default_single_node_eight_gpu_and_depth_only_weights(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        topology, args = self.arguments()
        self.assertIn("--nnodes=1", topology)
        self.assertIn("--nproc_per_node=8", topology)
        self.assertEqual(args[args.index("--global-batch-size")+1], "32")
        self.assertEqual(args[args.index("--grad-accum")+1], "1")
        self.assertEqual(args[args.index("--si-log-weight")+1], "1.0")
        self.assertEqual(args[args.index("--gradient-weight")+1], "0.5")
        self.assertEqual(args[args.index("--log-depth-weight")+1], "0.1")
        self.assertNotIn("--feature-weight", args)
        log = (self.root / "output/log.log").read_text()
        self.assertIn("batch/GPU=4", log)
        self.assertIn("fake-depthonly-stdout", log)
        self.assertIn("fake-depthonly-stderr", log)

    def test_multinode_ga_and_node_rank_logging(self):
        result = self.launch(dict(NNODES="2", NODE_RANK="1", MASTER_ADDR="test-master",
                                  NPROC_PER_NODE="4", GLOBAL_BATCH_SIZE="32", GRAD_ACCUM="2"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        topology, args = self.arguments()
        self.assertIn("--nnodes=2", topology)
        self.assertIn("--node_rank=1", topology)
        self.assertIn("--nproc_per_node=4", topology)
        self.assertEqual(args[args.index("--grad-accum")+1], "2")
        output = self.root / "output"
        self.assertFalse((output / "log.log").exists())
        log = (output / "console_node1_DepthOnly_flow2act.log").read_text()
        self.assertIn("batch/GPU=2", log)
        self.assertIn("fake-depthonly-stderr", log)

    def test_platform_topology_is_used_when_no_explicit_overrides(self):
        result = self.launch(dict(MLP_WORKER_NUM="2", MLP_ROLE_INDEX="1", RESOURCE_GPU="2",
                                  MLP_WORKER_0_HOST="platform-master", MLP_WORKER_0_PORT="12345"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        topology, _ = self.arguments()
        for value in ("--nnodes=2", "--node_rank=1", "--nproc_per_node=2",
                      "--master_addr=platform-master", "--master_port=12345"):
            self.assertIn(value, topology)

    def test_invalid_gbs_rejected_before_prepare_or_output(self):
        result = self.launch(dict(NPROC_PER_NODE="4", GLOBAL_BATCH_SIZE="31", GRAD_ACCUM="2"))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("GBS must be divisible", result.stderr)
        self.assertFalse((self.root / "prepare.json").exists())
        self.assertFalse((self.root / "argv.json").exists())
        self.assertFalse((self.root / "output").exists())

    def test_all_worker_console_streams_are_forwarded_and_original_exit_preserved(self):
        result = self.launch(dict(FAKE_EXIT_CODE="17"))
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        topology, _ = self.arguments()
        self.assertEqual(topology[topology.index("--tee")+1], "3")
        self.assertIn("--log-dir", topology)
        output = self.root / "output"
        log = (output / "log.log").read_text()
        self.assertEqual(log, (output / "console_node0_DepthOnly_flow2act.log").read_text())
        self.assertIn("fake-depthonly-stderr", log)

    def test_prepare_failure_is_reported_without_starting_training(self):
        result = self.launch(dict(FAKE_PREPARE_EXIT="23"))
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertIn("prepare-test-stderr", result.stderr)
        self.assertFalse((self.root / "argv.json").exists())
        self.assertFalse((self.root / "output").exists())

    def test_policy_and_output_spaces_are_preserved(self):
        output = self.root / "output with spaces=1"
        policy = self.root / "policy with spaces=0030000.ckpt"
        result = self.launch(dict(POLICY_CHECKPOINT=str(policy), OUTPUT_DIR=str(output)))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        _, args = self.arguments()
        self.assertEqual(args[args.index("--policy-checkpoint")+1], str(policy))
        self.assertEqual(args[-2:], ["--output-dir", str(output.resolve())])

    def test_live_node_console_lock_prevents_second_writer(self):
        output = self.root / "output"
        output.mkdir()
        console = output / "console_node0_DepthOnly_flow2act.log"
        with console.open("w") as handle:
            handle.write("running first job\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.launch()
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("already owned", result.stderr)
            self.assertFalse((self.root / "argv.json").exists())
            self.assertEqual(console.read_text(), "running first job\n")


class DepthOnlyPrepareTestsFlow2Act(unittest.TestCase):
    """Read actual checkpoint metadata, without building either model."""

    def setUp(self):
        from xr0.tools import train_flow2act_depth_probe as entry
        self.entry = entry
        self.temporary = tempfile.TemporaryDirectory(prefix="prepare_depthonly_flow2act_")
        self.root = Path(self.temporary.name)
        self.policy = self.root / "source-static-run/filename-step999999.ckpt"
        self.policy.parent.mkdir()
        self.contract = {
            "schema": "Flow2Act-XR0-checkpoint-v1", "kind": "trained_policy", "model_config": {},
            "method_config": {"stage": "static", "memory_length": 1}, "vlm_config": {},
            "experiment_config": {}, "action_stats": {},
        }
        self.save_policy()

    def tearDown(self):
        self.temporary.cleanup()

    def save_policy(self, *, contract=None, step=30000):
        torch.save({"flow2act": self.contract if contract is None else contract,
                    "global_step": step}, self.policy)

    def prepare(self, arguments=(), *, node=0):
        argv = [PY_ENTRY, "--prepare-output", "--policy-checkpoint", str(self.policy),
                "--output-root", str(self.root / "output"), *arguments]
        console = io.StringIO()
        environment = {"NNODES": "1", "NPROC_PER_NODE": "8", "NODE_RANK": str(node), "WORLD_SIZE": "8"}
        with patch("sys.argv", argv), patch.dict(os.environ, environment), redirect_stdout(console), \
             patch("torch.cuda._lazy_init", side_effect=AssertionError("prepare initialized CUDA")):
            self.entry.main()
        lines = console.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        output = Path(lines[0])
        self.assertTrue(output.is_absolute())
        self.assertTrue(output.is_dir())
        return output

    def test_auto_directory_uses_real_checkpoint_step_and_changes_with_recipe(self):
        stat = self.policy.stat()
        default = self.prepare()
        self.assertIn("source-static-run-step0030000-static-L1-8gpu-gbs32-ga1", default.name)
        self.assertNotIn("999999", default.name)
        self.assertIn("-steps5000-si1-gr0.5-log0.1-", default.name)
        names = {default.name}
        for args in (("--lr", "2e-4"), ("--grad-accum", "2"), ("--global-batch-size", "64"),
                     ("--log-depth-weight", "0.2"), ("--seed", "7")):
            output = self.prepare(args)
            self.assertNotIn(output.name, names)
            names.add(output.name)
        self.assertEqual(self.policy.stat().st_size, stat.st_size)
        self.assertEqual(self.policy.stat().st_mtime_ns, stat.st_mtime_ns)
        self.assertEqual(list(default.iterdir()), [])

    def test_explicit_output_with_spaces_and_existing_empty_directory(self):
        output = self.root / "explicit output=1"
        output.mkdir()
        self.assertEqual(self.prepare(("--output-dir", str(output))), output)

    def test_fresh_nonempty_output_is_rejected_without_appending_or_overwriting(self):
        output = self.root / "existing"
        output.mkdir()
        for name in ("log.log", "config_depth_probe_DepthOnly_flow2act.json"):
            (output / name).write_text("old experiment\n")
        with self.assertRaises(FileExistsError):
            self.prepare(("--output-dir", str(output)))
        self.assertEqual(len(list(output.iterdir())), 2)
        self.assertTrue(all(path.read_text() == "old experiment\n" for path in output.iterdir()))

    def test_node_zero_can_follow_other_node_console_bootstrap(self):
        output = self.root / "multinode"
        output.mkdir()
        console = output / "console_node1_DepthOnly_flow2act.log"
        console.write_text("node1 waiting\n")
        (output / "torchrun_node1_DepthOnly_flow2act").mkdir()
        self.assertEqual(self.prepare(("--output-dir", str(output))), output)
        self.assertEqual(console.read_text(), "node1 waiting\n")
        (output / "console_node0_DepthOnly_flow2act.log").write_text("old root console\n")
        with self.assertRaises(FileExistsError):
            self.prepare(("--output-dir", str(output)))

    def test_legacy_mechanical_and_untrained_policy_are_rejected(self):
        for contract, step in (({}, 30000), ({**self.contract, "kind": "mechanical_training_smoke"}, 30000),
                               (self.contract, 0)):
            with self.subTest(kind=contract.get("kind"), step=step):
                self.save_policy(contract=contract, step=step)
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse((self.root / "output").exists())


    def test_invalid_gbs_loss_and_validation_settings_fail_before_output(self):
        for args in (("--global-batch-size", "0"), ("--global-batch-size", "31"),
                     ("--grad-accum", "0"), ("--si-log-weight", "-1"),
                     ("--val-fraction", "1"), ("--clip-grad-norm", "0")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.prepare(args)
            self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
