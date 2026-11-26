"""Check automatic statistics preparation without reading LIBERO or using GPUs."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools/compute_libero_stats_flow2act.py"
STATS = {
    "mean": [[0.25] * 7 + [0.0] * 25 for _ in range(3)],
    "std": [[0.5] * 7 + [0.0] * 25 for _ in range(3)],
    "normalization": "zscore",
    "action_length": 3,
    "valid_action_dims": list(range(7)),
    "num_windows": 12,
}

CHILD = r'''
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

tool, output, calls, ready, gate, mode, payload = sys.argv[1:]
spec = importlib.util.spec_from_file_location("statistics_tool", tool)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def compute_stats(root, horizon):
    assert horizon == 3
    with open(calls, "a") as stream:
        stream.write("compute\n")
        stream.flush()
    time.sleep(0.2)
    if mode == "fail":
        raise RuntimeError("simulated statistics failure")
    return json.loads(payload)

module.compute_stats = compute_stats
Path(ready).touch()
deadline = time.monotonic() + 20
while not Path(gate).exists():
    if time.monotonic() > deadline:
        raise TimeoutError("test start gate was not released")
    time.sleep(0.01)
sys.argv = [tool, "--data-root", str(Path(output).parent / "absent_dataset"),
            "--output", output, "--action-length", "3", "--if-missing"]
module.main()
print("RESULT_JSON=" + json.dumps(json.loads(Path(output).read_text())))
'''


class AutomaticStatsFlow2ActTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="auto_stats_flow2act_")
        self.root = Path(self.temporary.name)
        self.output = self.root / "checkpoints/stats.json"
        self.calls = self.root / "calls.txt"
        self.gate = self.root / "start"
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
        self.temporary.cleanup()

    def start(self, number, mode="success"):
        ready = self.root / f"ready{number}"
        process = subprocess.Popen(
            [sys.executable, "-c", CHILD, str(TOOL), str(self.output),
             str(self.calls), str(ready), str(self.gate), mode, json.dumps(STATS)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.processes.append(process)
        return process, ready

    def release(self, children):
        deadline = time.monotonic() + 30
        while not all(ready.exists() for _, ready in children):
            for process, ready in children:
                if process.poll() is not None and not ready.exists():
                    out, err = process.communicate()
                    self.fail(f"statistics child exited before startup: {out}{err}")
            if time.monotonic() > deadline:
                self.fail("statistics children did not become ready")
            time.sleep(0.01)
        self.gate.touch()

    def test_concurrent_missing_statistics_are_computed_once(self):
        children = [self.start(0), self.start(1)]
        self.release(children)
        for process, _ in children:
            out, err = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, out + err)
            result = next(line.removeprefix("RESULT_JSON=")
                          for line in out.splitlines() if line.startswith("RESULT_JSON="))
            self.assertEqual(json.loads(result), STATS)
        self.assertEqual(self.calls.read_text().splitlines(), ["compute"])
        self.assertEqual(json.loads(self.output.read_text()), STATS)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_existing_statistics_are_reused_without_accessing_dataset(self):
        self.output.parent.mkdir()
        original = json.dumps(STATS).encode()
        self.output.write_bytes(original)
        result = subprocess.run(
            [sys.executable, str(TOOL), "--data-root", str(self.root / "absent_dataset"),
             "--output", str(self.output), "--action-length", "3", "--if-missing"],
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_failed_calculation_propagates_and_leaves_no_statistics(self):
        child = self.start(0, "fail")
        self.release([child])
        process, _ = child
        out, err = process.communicate(timeout=30)
        self.assertNotEqual(process.returncode, 0, out + err)
        self.assertIn("simulated statistics failure", err)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
