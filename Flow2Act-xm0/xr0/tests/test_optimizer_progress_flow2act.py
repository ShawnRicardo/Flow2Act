"""Real CPU Lightning tests for optimizer-update progress and resume."""

import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "xr0"))

from mibot.utils.optimizer_progress_flow2act import OptimizerStepProgressFlow2Act

MODULE = "mibot.utils.optimizer_progress_flow2act"


class LinearModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.seen_batches = 0

    def training_step(self, batch, batch_idx):
        self.seen_batches += 1
        # A progress-bar metric must not reintroduce Epoch/v_num labels.
        loss = self.linear(batch[0]).square().mean()
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class RecordingProgress(OptimizerStepProgressFlow2Act):
    def __init__(self):
        super().__init__(mininterval=0)
        self.steps_after_batch = []
        self.initial_step = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.initial_step = self.progress_bar.n

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        before = self.progress_bar.n
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.steps_after_batch.append((int(trainer.global_step), before, self.progress_bar.n))


class OptimizerProgressTest(unittest.TestCase):
    def setUp(self):
        # All checkpoint/log artifacts stay on the project's EFS results link.
        scratch = REPO / "results/smoke/Flow2Act"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="progress_cpu_", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.output = io.StringIO()
        self.addCleanup(self.output.close)

        def make_bar(*args, **kwargs):
            return tqdm(*args, file=self.output, **kwargs)

        self.patch = patch(MODULE + ".tqdm", side_effect=make_bar)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def trainer(self, progress, accumulation, updates, **kwargs):
        return L.Trainer(
            accelerator="cpu", devices=1, max_steps=updates, max_epochs=-1,
            accumulate_grad_batches=accumulation,
            callbacks=[progress] if progress is not None else [],
            logger=False, enable_checkpointing=False, enable_model_summary=False,
            default_root_dir=self.temp.name, **kwargs,
        )

    def data(self, size=64):
        return DataLoader(TensorDataset(torch.ones(size, 2)), batch_size=1, num_workers=0)

    def test_ga_one_and_four_count_only_real_updates(self):
        for accumulation in (1, 4):
            with self.subTest(accumulation=accumulation):
                progress = RecordingProgress()
                trainer = self.trainer(progress, accumulation, updates=3)
                model = LinearModel()
                trainer.fit(model, self.data())
                self.assertEqual(trainer.global_step, 3)
                self.assertIs(trainer.progress_bar_callback, progress)
                self.assertEqual(model.seen_batches, 3 * accumulation)
                self.assertEqual(progress.initial_step, 0)
                self.assertEqual(progress.progress_bar.total, 3)
                self.assertEqual(progress.progress_bar.n, 3)
                self.assertEqual(sum(after - before for _, before, after in progress.steps_after_batch), 3)
                self.assertEqual(sum(after > before for _, before, after in progress.steps_after_batch), 3)
                self.assertTrue(all(step == after for step, _, after in progress.steps_after_batch))
        rendered = self.output.getvalue()
        self.assertIn("Updates", rendered)
        self.assertIn("3/3", rendered)
        self.assertNotIn("Epoch", rendered)
        self.assertNotIn("v_num", rendered)

    def test_resume_starts_at_saved_update(self):
        first = self.trainer(RecordingProgress(), accumulation=4, updates=2)
        first.fit(LinearModel(), self.data())
        checkpoint = str(Path(self.temp.name) / "step2.ckpt")
        first.save_checkpoint(checkpoint)
        progress = RecordingProgress()
        resumed = self.trainer(progress, accumulation=4, updates=4)
        resumed.fit(LinearModel(), self.data(), ckpt_path=checkpoint)
        self.assertEqual(progress.initial_step, 2)
        self.assertEqual(progress.progress_bar.n, 4)
        self.assertEqual(progress.progress_bar.total, 4)
        self.assertEqual(sum(after - before for _, before, after in progress.steps_after_batch), 2)

    def test_epoch_boundary_does_not_reset_update_counter(self):
        progress = RecordingProgress()
        trainer = self.trainer(progress, accumulation=1, updates=5)
        trainer.fit(LinearModel(), self.data(size=2))
        self.assertGreaterEqual(trainer.current_epoch, 2)
        self.assertEqual(progress.progress_bar.n, 5)
        self.assertEqual([after for _, _, after in progress.steps_after_batch], [1, 2, 3, 4, 5])

    def test_disabled_trainer_has_no_default_or_custom_bar(self):
        trainer = self.trainer(None, accumulation=4, updates=1, enable_progress_bar=False)
        trainer.fit(LinearModel(), self.data())
        self.assertIsNone(trainer.progress_bar_callback)
        self.assertEqual(self.output.getvalue(), "")

    def test_nonzero_rank_and_disable_are_silent(self):
        for rank_zero in (False, True):
            progress = OptimizerStepProgressFlow2Act()
            trainer = SimpleNamespace(is_global_zero=rank_zero, global_step=0, max_steps=5)
            progress.setup(trainer, None, "fit")
            if rank_zero:
                progress.disable()
            progress.on_train_start(trainer, None)
            trainer.global_step = 1
            progress.on_train_batch_end(trainer, None, None, None, 0)
            progress.on_train_end(trainer, None)
            self.assertIsNone(progress.progress_bar)
        self.assertEqual(self.output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
