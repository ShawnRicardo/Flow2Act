"""Display optimizer updates, independently of batches and gradient accumulation."""

from typing import Any

from lightning.pytorch.callbacks.progress import ProgressBar
from tqdm import tqdm


class OptimizerStepProgressFlow2Act(ProgressBar):
    """Use Lightning's restored ``global_step`` as the only progress counter.

    Add this callback only when ``Trainer(enable_progress_bar=True)``. Inheriting
    ProgressBar also prevents Lightning from installing its epoch/batch bar.
    """

    def __init__(self, mininterval: float = 1.0) -> None:
        super().__init__()
        self._enabled = True
        self.mininterval = mininterval
        self.progress_bar: tqdm | None = None
        self._last_step = 0

    def disable(self) -> None:
        self._enabled = False
        if self.progress_bar is not None:
            self.progress_bar.disable = True

    def enable(self) -> None:
        self._enabled = True
        if self.progress_bar is not None and self.trainer.is_global_zero:
            self.progress_bar.disable = False

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        if not self._enabled or not trainer.is_global_zero:
            return
        self._close()
        self._last_step = int(trainer.global_step)
        maximum = trainer.max_steps
        self.progress_bar = tqdm(
            total=int(maximum) if maximum is not None and maximum > 0 else None,
            initial=self._last_step,
            desc="Updates",
            unit="update",
            dynamic_ncols=True,
            mininterval=self.mininterval,
        )

    def _sync(self, trainer: Any) -> None:
        if self.progress_bar is None or not self._enabled or not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        # Accumulation-only batches leave global_step unchanged. DeepSpeed and
        # native Lightning both expose completed optimizer updates here.
        if step > self._last_step:
            self.progress_bar.update(step - self._last_step)
            self._last_step = step

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self._sync(trainer)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        self._sync(trainer)
        self._close()

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        self._close()

    def teardown(self, trainer: Any, pl_module: Any, stage: str) -> None:
        self._close()

    def _close(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
