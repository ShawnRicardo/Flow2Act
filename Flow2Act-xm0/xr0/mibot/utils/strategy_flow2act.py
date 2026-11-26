"""Native ZeRO-2 checkpoints on three non-shared, task-local filesystems.

Checkpoint handling changes only filesystem placement. DeepSpeed still writes
and loads its native model/optimizer format, and rigid resume requires the same
data-parallel world size. The chief host retains complete checkpoints; workers
reuse a bounded save spool. Before resume, run tools/prepare_checkpoint_flow2act.py
once per physical host, with that host's actual global ranks. An explicit
process-exit-only option skips the native model-to-CPU teardown copy; its
default retains native cleanup behavior.
"""
from pathlib import Path
import subprocess
from typing import Any

import torch.distributed as dist
from lightning.pytorch.strategies import DeepSpeedStrategy


def checkpoint_relative_flow2act(path: str, namespace_root: str) -> Path:
    """Reject escapes; preserve the chief's checkpoint-relative path on workers."""
    target, namespace = Path(path), Path(namespace_root)
    if not target.is_absolute() or not namespace.is_absolute():
        raise ValueError("Checkpoint path and namespace root must be absolute")
    if ".." in target.parts or ".." in namespace.parts:
        raise ValueError("Parent traversal is not permitted in checkpoint paths")
    relative = target.relative_to(namespace)
    if relative == Path("."):
        raise ValueError("Checkpoint must be below, not equal to, namespace root")
    return relative


class DeepSpeedStrategyFlow2Act(DeepSpeedStrategy):
    """Preserve native ZeRO-2 math and gather only native optimizer files.

    ``checkpoint_chief_host`` is true for *every* rank on the physical chief
    host. Global rank zero must be on that host. ``checkpoint_rsync_url`` is a
    dedicated writable rsync module exporting exactly the chief local root.
    Use immutable checkpoint names and native save_last="link" (or False).
    The URL and local roots are deployment settings, not scientific metadata.
    """

    def __init__(self, *, checkpoint_namespace_root: str,
                 checkpoint_local_root: str, checkpoint_chief_host: bool,
                 checkpoint_rsync_url: str, checkpoint_transfer_timeout: int = 1800,
                 teardown_model_to_cpu: bool = True,
                 use_multi_rank_bucket_allreduce: bool | None = None,
                 **kwargs: Any) -> None:
        if not isinstance(teardown_model_to_cpu, bool):
            raise TypeError("teardown_model_to_cpu must be a bool")
        self.teardown_model_to_cpu = teardown_model_to_cpu
        super().__init__(**kwargs)
        if use_multi_rank_bucket_allreduce is not None:
            self.config["zero_optimization"]["use_multi_rank_bucket_allreduce"] = use_multi_rank_bucket_allreduce
        self._validate_teardown_mode()
        self.checkpoint_namespace_root = str(Path(checkpoint_namespace_root))
        self.checkpoint_local_root = Path(checkpoint_local_root)
        self.checkpoint_chief_host = bool(checkpoint_chief_host)
        self.checkpoint_rsync_url = checkpoint_rsync_url.rstrip("/")
        self.checkpoint_transfer_timeout = int(checkpoint_transfer_timeout)
        for root in (Path(self.checkpoint_namespace_root), self.checkpoint_local_root):
            if not root.is_absolute() or ".." in root.parts:
                raise ValueError("Checkpoint roots must be absolute without parent traversal")
        if not self.checkpoint_rsync_url.startswith("rsync://"):
            raise ValueError("A task-specific rsync:// module is required")
        if self.checkpoint_transfer_timeout <= 0:
            raise ValueError("Checkpoint transfer timeout must be positive")
        if self.load_full_weights or self.config["zero_optimization"]["stage"] != 2:
            raise ValueError("This filesystem strategy supports native rigid ZeRO-2 only")
        if self.config["zero_optimization"].get("elastic_checkpoint", False):
            raise ValueError("Elastic optimizer checkpoint loading is not supported")
        if self.config.get("checkpoint", {}).get("use_node_local_storage", False):
            raise ValueError("Native global-rank checkpoint filenames are required")

    def _validate_teardown_mode(self) -> None:
        if not self.teardown_model_to_cpu and self._layer_sync is not None:
            raise ValueError("Exit-only teardown without model.cpu() does not support layer sync or sync batchnorm")

    def setup_environment(self) -> None:
        # Trainer injects its layer-sync plugin after strategy construction.
        self._validate_teardown_mode()
        super().setup_environment()

    def teardown(self) -> None:
        if self.teardown_model_to_cpu:
            return super().teardown()
        self._validate_teardown_mode()
        model, lightning_module = self.model, self._lightning_module
        try:
            # Lightning 2.5.3's lightning_module getter uses _lightning_module,
            # independently of model. Detach both only during native cleanup:
            # optimizer/precision/accelerator/checkpoint cleanup still runs.
            self.model = None
            self._lightning_module = None
            super().teardown()
        finally:
            # Exit-only callers retain the live GPU model until process exit.
            # Preserve these references even when native cleanup raises.
            self.model = model
            self._lightning_module = lightning_module

    def local_checkpoint_path(self, filepath: str) -> Path:
        return self.checkpoint_local_root / checkpoint_relative_flow2act(
            str(filepath), self.checkpoint_namespace_root)

    def _transfer_optimizer(self, source: Path, relative: Path) -> None:
        destination = f"{self.checkpoint_rsync_url}/{relative.as_posix()}/checkpoint/"
        # rsync's default temporary file + rename publishes a complete shard.
        subprocess.run(["rsync", "-t", "--", str(source), destination],
                       check=True, capture_output=True, text=True,
                       timeout=self.checkpoint_transfer_timeout)

    def save_checkpoint(self, checkpoint: dict, filepath: str,
                        storage_options: Any = None) -> None:
        if storage_options is not None:
            raise TypeError("DeepSpeedStrategy does not support storage_options")
        # Native Lightning already broadcasts the chief path. Include the
        # chief's exclusive-directory reservation error in that same broadcast.
        request = None
        if self.is_global_zero:
            try:
                if not self.checkpoint_chief_host:
                    raise ValueError("Global rank zero must run on the checkpoint chief host")
                relative = checkpoint_relative_flow2act(
                    str(filepath), self.checkpoint_namespace_root)
                self.local_checkpoint_path(str(filepath)).mkdir(parents=True, exist_ok=False)
                request = (str(filepath), None)
            except Exception as error:
                request = (str(filepath), f"{type(error).__name__}: {error}")
        canonical, reservation_error = self.broadcast(request)
        if reservation_error:
            raise RuntimeError(f"Checkpoint save refused: {reservation_error}")
        relative = checkpoint_relative_flow2act(canonical, self.checkpoint_namespace_root)
        local_path = (self.local_checkpoint_path(canonical) if self.checkpoint_chief_host
                      else self.checkpoint_local_root / "save_spool_flow2act")
        client_state = {key: value for key, value in checkpoint.items()
                        if key not in ("state_dict", "optimizer_states")}
        # Native save includes its own final collective barrier after all files
        # are closed. No additional pre-transfer barrier is needed.
        self.deepspeed_engine.save_checkpoint(str(local_path), client_state=client_state,
                                             tag="checkpoint")
        transfer_error = None
        try:
            if not self.checkpoint_chief_host:
                source = Path(self.deepspeed_engine._get_zero_ckpt_name(
                    str(local_path), "checkpoint"))
                if not source.is_file():
                    raise FileNotFoundError(source)
                self._transfer_optimizer(source, relative)
        except Exception as error:
            details = getattr(error, "stderr", None) or str(error)
            transfer_error = f"rank {dist.get_rank()}: {type(error).__name__}: {details}"
        errors = [None] * dist.get_world_size()
        # All ranks see a failed rsync and fail together, rather than one rank
        # raising while its peers enter a later training/checkpoint barrier.
        dist.all_gather_object(errors, transfer_error)
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError("Checkpoint shard transfer failed collectively: " + "; ".join(failures))

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        return super().load_checkpoint(str(self.local_checkpoint_path(str(checkpoint_path))))

    def remove_checkpoint(self, filepath: str) -> None:
        # Native retention runs the deletion only on global zero. Worker spool
        # and separately prepared resume directories are never pruned here.
        super().remove_checkpoint(str(self.local_checkpoint_path(str(filepath))))
