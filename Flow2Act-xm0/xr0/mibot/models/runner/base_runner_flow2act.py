# Copyright (C) 2026 Xiaomi Corporation.
from importlib import import_module
import json
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from safetensors.torch import load_file
from lightning import LightningModule, seed_everything
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from transformers.utils import logging

from mibot.models import MIMODEL
from mibot.utils.pretrain_base_flow2act import pretrain_base_directory_flow2act

logger = logging.get_logger(__name__)


@MIMODEL.register_module()
class BaseRunnerFlow2Act(LightningModule):
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the BaseRunner module.

        Args:
            params (Dict[str, Any]): A dictionary containing configurations for the model,
                                    optimizer, scheduler, and optionally pretrained weights.
        """
        super().__init__()
        self._training_seed = int(params.get("training_seed", 42))
        self._resume_requested = bool(params.get("resume_requested", False))
        self._teacher_dir = params.get("teacher_weights_dir")
        self._teacher_cpu_offload = params.get("teacher_cpu_offload", False)
        if type(self._teacher_cpu_offload) is not bool:
            raise ValueError("teacher_cpu_offload must be boolean")
        self._teacher_compute_device = params.get("teacher_compute_device", "cuda")
        if self._teacher_compute_device not in ("cpu", "cuda"):
            raise ValueError("teacher_compute_device must be cpu or cuda")
        if self._teacher_compute_device == "cpu" and self._teacher_cpu_offload:
            raise ValueError("CPU teacher compute cannot also enable GPU-between-batch offload")
        self._teacher_mmap_weights = params.get("teacher_mmap_weights", False)
        if type(self._teacher_mmap_weights) is not bool:
            raise ValueError("teacher_mmap_weights must be boolean")
        if self._teacher_mmap_weights and (self._teacher_compute_device != "cpu" or self._teacher_cpu_offload):
            raise ValueError("Shared DA3 weights require CPU compute without GPU offload")
        self._expected_world_size = int(params.get("expected_world_size", 0))
        self._checkpoint_experiment_config = params.get("checkpoint_experiment_config", {})
        self._checkpoint_kind = params.get("checkpoint_kind", "trained_policy")
        self._checkpoint_action_stats = params.get("checkpoint_action_stats")
        self._exact_batch = dict(params.get("exact_effective_batch", {"enabled": False}))
        self._batch_micro = 0
        self._batch_metric_sums = {}
        self._batch_valid_count = 0
        object.__setattr__(self, "_teacher", None)
        self._pretrained: Optional[str] = params.get("pretrained")
        self._model: Dict[str, Any] = params.get("model")
        self._optimizer: Dict[str, Any] = params.get("optimizer")
        self._max_optimizer_group_elements = params.get("max_optimizer_group_elements", 0)
        if type(self._max_optimizer_group_elements) is not int or self._max_optimizer_group_elements < 0:
            raise ValueError("max_optimizer_group_elements must be a nonnegative integer")
        self._scheduler: Optional[Dict[str, Any]] = params.get("scheduler")

    def configure_model(self) -> None:
        """
        Builds the model architecture and loads pretrained weights if provided.
        """
        if hasattr(self, "model"):
            return
        if self._pretrained:
            pretrain_base_directory_flow2act(self._pretrained)
        self.model = MIMODEL.build(self._model)

        if self._pretrained:
            self.load_pretrained(self._pretrained)

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Initialize new experiments exclusively from the native Pretrain Base."""
        path = pretrain_base_directory_flow2act(checkpoint_path)

        index_path = path / "model.safetensors.index.json"
        shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values())) if index_path.exists() else ["model.safetensors"]
        loaded_keys = set()
        unexpected_keys = set()
        for shard in shards:
            weights = {f"model.{key}": value for key, value in load_file(str(path / shard)).items()}
            info = self.load_state_dict(weights, strict=False)
            loaded_keys.update(weights)
            unexpected_keys.update(info.unexpected_keys)
            del weights

        missing_keys = set(self.state_dict()) - loaded_keys
        # Safetensors stores the tied token embedding only once.
        embedding = self.model.vlm.get_input_embeddings().weight
        if self.model.vlm.lm_head.weight is embedding:
            missing_keys.discard("model.vlm.lm_head.weight")
        allowed = tuple("model." + name for name in self.model.new_parameter_prefixes)
        missing_base = {key for key in missing_keys if not key.startswith(allowed)}
        if missing_base or unexpected_keys:
            raise RuntimeError(f"Native base checkpoint mismatch: missing={sorted(missing_base)}, unexpected={sorted(unexpected_keys)}")
        logger.info(f"Loaded {len(loaded_keys)} native pretrained tensors; initialized {len(missing_keys)} new tensors")

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer and scheduler configuration.
        """
        optimizer = self.build_optimizer(self._optimizer, self.named_parameters(),
                                         max_group_elements=self._max_optimizer_group_elements)
        scheduler = self.build_scheduler(self._scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @staticmethod
    def build_optimizer(cfg: Dict[str, Any], parameters: Iterator[Tuple[str, torch.Tensor]],
                        max_group_elements: int = 0) -> Optimizer:
        """
        Builds an optimizer with weight decay applied conditionally based on parameter names.

        Args:
            cfg (Dict[str, Any]): Optimizer configuration dictionary, including 'type' and 'params'.
            parameters (Iterator[Tuple[str, torch.Tensor]]): Model parameters with their names.

        Returns:
            Optimizer: Constructed optimizer instance.
        """
        module_params = cfg.get("params", {})
        from mibot.utils.optimizer_groups_flow2act import native_optimizer_groups_flow2act
        optimizer_grouped_parameters,_ = native_optimizer_groups_flow2act(parameters,
            weight_decay=module_params.get("weight_decay",0.1),max_group_elements=max_group_elements)

        module_type = cfg["type"]
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        optimizer_class = getattr(module, class_name)

        return optimizer_class(optimizer_grouped_parameters, **module_params)

    @staticmethod
    def build_scheduler(cfg: Optional[Dict[str, Any]], optimizer: Optimizer) -> Optional[LRScheduler]:
        """
        Builds a learning rate scheduler from the given configuration.

        Args:
            cfg (Optional[Dict[str, Any]]): Scheduler configuration dictionary.
            optimizer (Optimizer): The optimizer to be used with the scheduler.

        Returns:
            Optional[LRScheduler]: Constructed learning rate scheduler, or None if no config is given.
        """
        if cfg is None:
            return None

        module_type = cfg["type"]
        module_params = cfg.get("params", {})
        module_name, class_name = module_type.rsplit(".", 1)
        module = import_module(module_name)
        scheduler_class = getattr(module, class_name)

        return scheduler_class(optimizer, **module_params)

    def _build_teacher(self):
        if self._teacher is None:
            if not self._teacher_dir:
                raise ValueError("teacher_weights_dir is required for frozen DA3 supervision")
            # Lazy teacher construction must not change the policy/noise/augmentation streams.
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=devices):
                    from flow2act.teacher import FrozenDA3TeacherFlow2Act
                    teacher = FrozenDA3TeacherFlow2Act(
                        self._teacher_dir, device="cpu" if self._teacher_cpu_offload or self._teacher_compute_device == "cpu" else self.device,
                        mmap_weights=self._teacher_mmap_weights).eval()
                    teacher.requires_grad_(False)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
            object.__setattr__(self, "_teacher", teacher)
        return self._teacher

    def on_fit_start(self):
        # Extra stage parameters consume RNG during construction. Align fresh runs
        # after DeepSpeed setup. Explicit resumes skip this fresh-run alignment;
        # this is not a claim that native checkpoints restore every RNG stream.
        if not self._resume_requested and not getattr(self.trainer, "ckpt_path", None):
            seed_everything(self._training_seed, workers=True)
        if self._expected_world_size and int(self.trainer.world_size) != self._expected_world_size:
            raise ValueError(f"Actual world_size={self.trainer.world_size} differs from TOTAL_GPUS={self._expected_world_size}")
        self._effective_batch_contract()

    def _effective_batch_contract(self):
        from mibot.utils.global_batch_padding_flow2act import batch_contract_flow2act
        cfg = self._exact_batch
        if not cfg.get("enabled", False):
            return {"enabled": False}
        trainer = getattr(self, "_trainer", None)
        configured_world = int(cfg.get("world_size", self._expected_world_size))
        configured_ga = int(cfg.get("gradient_accumulation_steps", 0))
        world = int(trainer.world_size) if trainer is not None else configured_world
        accumulation = int(trainer.accumulate_grad_batches) if trainer is not None else configured_ga
        if (configured_world and configured_world != world) or (configured_ga and configured_ga != accumulation):
            raise ValueError("Configured effective batch differs from actual Trainer world/accumulation")
        return batch_contract_flow2act(enabled=cfg["enabled"],
            target=int(cfg["target_global_batch"]), world_size=world, accumulation=accumulation,
            micro_batch=int(cfg["micro_batch_size"]))

    def _checkpoint_contract(self):
        if not hasattr(self,"model"):
            self.configure_model()
        from mibot.utils.checkpoint_contract_flow2act import (
            build_checkpoint_contract_flow2act,resolved_model_config_flow2act)
        from mibot.utils.optimizer_groups_flow2act import native_optimizer_groups_flow2act
        _,group_contract=native_optimizer_groups_flow2act(self.named_parameters(),
            weight_decay=(self._optimizer or {}).get("params",{}).get("weight_decay",0.1),
            max_group_elements=self._max_optimizer_group_elements)
        return build_checkpoint_contract_flow2act(
            model_config=resolved_model_config_flow2act(self._model,type(self.model).__init__),
            method_config=self.model.geo_motion.config.to_dict(),vlm_config=self.model.vlm.config.to_dict(),
            experiment_config=self._checkpoint_experiment_config,action_stats=self._checkpoint_action_stats,
            kind=self._checkpoint_kind,batch_contract=self._effective_batch_contract(),
            optimizer_group_contract=group_contract)

    def on_save_checkpoint(self, checkpoint):
        from mibot.utils.checkpoint_contract_flow2act import CHECKPOINT_KEY_FLOW2ACT
        if self._batch_micro:
            raise ValueError("Exact effective-batch checkpoint requires an optimizer-update boundary")
        checkpoint[CHECKPOINT_KEY_FLOW2ACT] = self._checkpoint_contract()

    def on_load_checkpoint(self, checkpoint):
        from mibot.utils.checkpoint_contract_flow2act import (
            read_checkpoint_contract_flow2act,validate_checkpoint_contract_flow2act)
        saved=read_checkpoint_contract_flow2act(checkpoint)
        validate_checkpoint_contract_flow2act(self._checkpoint_contract(),saved)

    @torch.no_grad()
    def _teacher_features(self, batch):
        if self._model.get("stage", "baseline") == "baseline":
            return None
        raw = batch["teacher_images"]
        if raw.dtype != torch.uint8 or raw.ndim != 5 or raw.shape[1:3] != (2, 3):
            raise ValueError("Frozen DA3 requires independent uint8 raw joint-view RGB")
        raw = raw.to("cpu" if self._teacher_compute_device == "cpu" else self.device).float().div(255.0)
        rgb = torch.nn.functional.interpolate(raw.flatten(0,1),size=(252,252),
            mode="bilinear",align_corners=False,antialias=True).reshape(raw.shape[0],2,3,252,252)
        # Convex bilinear interpolation can exceed [0,1] by a few FP32 ULPs
        # (real LIBERO ep0/frame0 reaches 1 + 1 ULP). Reject substantive range
        # errors before removing this numerical resize overshoot only.
        tolerance = 8 * torch.finfo(torch.float32).eps
        if not torch.isfinite(rgb).all() or (rgb < -tolerance).any() or (rgb > 1 + tolerance).any():
            raise ValueError("Raw RGB resize exceeded the FP32 rounding tolerance")
        rgb = rgb.clamp(0.0, 1.0)
        from mibot.utils.teacher_runtime_flow2act import teacher_features_flow2act
        return teacher_features_flow2act(self._build_teacher(),rgb,
            device=self.device,cpu_offload=self._teacher_cpu_offload,
            compute_device=self._teacher_compute_device)

    def training_step(self, batch, batch_idx):
        teacher_features = self._teacher_features(batch)
        losses = self.model(batch, return_loss=True, teacher_features=teacher_features,
                            capture_visualization=(self._model.get("stage", "baseline") != "baseline"
                                and bool(getattr(self, "_capture_visualization", False))))
        snapshot = losses.pop("_visualization", None)
        if snapshot is not None:
            snapshot["teacher_features"] = tuple(value[:1].detach().cpu().float() for value in teacher_features)
            snapshot["current_rgb"] = batch["teacher_images"][:1].detach().cpu()
            snapshot["metadata"] = {
                "suite": batch["suite"][0],
                "episode_index": int(batch["episode_index"][0]),
                "frame_index": int(batch["frame_index"][0]),
            }
            object.__setattr__(self, "_visualization_payload", snapshot)
        contract = self._effective_batch_contract()
        weight = 1.0
        if contract["enabled"]:
            from mibot.utils.global_batch_padding_flow2act import slot_weight_flow2act
            if batch["input_ids"].shape[0] != 1:
                raise ValueError("Exact effective batch only supports actual microbatch one")
            accumulation = contract["gradient_accumulation_steps"]
            engine = self.trainer.model
            if hasattr(engine, "micro_steps") and int(engine.micro_steps) % accumulation != self._batch_micro:
                raise RuntimeError("DeepSpeed and effective-batch microstep counters differ")
            weight = slot_weight_flow2act(target=contract["target_global_batch"],
                world_size=contract["world_size"], accumulation=accumulation,
                rank=int(self.global_rank), micro_step=self._batch_micro, global_step=int(self.global_step))
            # Inputs/forward/graph are identical for neutral slots; only the total
            # scalar loss is weighted. NaNs are deliberately not hidden by masking.
            for name, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    self._batch_metric_sums[name] = self._batch_metric_sums.get(name, 0) + value.detach().float() * weight
            self._batch_valid_count += int(weight > 0)
            self._batch_micro += 1
            if self._batch_micro == accumulation:
                # Per-rank /GA then native rank mean gives exactly the mean over
                # the target contributing observations, including neutral slots.
                for name, value in self._batch_metric_sums.items():
                    self.log(f"train/{name}", value / accumulation, on_step=True, on_epoch=False,
                             sync_dist=True, batch_size=1)
                self.log("train/effective_sample_count", losses["loss"].new_tensor(float(self._batch_valid_count)),
                         on_step=True, on_epoch=False, sync_dist=True, reduce_fx="sum", batch_size=1)
                self.log("train/native_padded_slots", float(contract["native_padded_slots"]),
                         on_step=True, on_epoch=False, batch_size=1)
                self._batch_micro = 0
                self._batch_metric_sums = {}
                self._batch_valid_count = 0
        else:
            for name, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.log(f"train/{name}", value.detach(), on_step=True, on_epoch=False,
                             batch_size=batch["input_ids"].shape[0])
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return losses["loss"] * weight

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Lightning's CSV `step` is zero-based. Expose its actual completed
        # update counter at the exact-batch boundary without predicting a step
        # before backward/optimizer execution.
        if self._exact_batch.get("enabled", False) and self._batch_micro == 0:
            self.log("train/optimizer_step", float(self.global_step), on_step=True,
                     on_epoch=False, batch_size=1)

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Performs a single validation step.

        Args:
            batch (Dict[str, Any]): A dictionary containing input tensors for the model.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader, useful when using multiple val datasets.

        Returns:
            Dict[str, torch.Tensor]: Loss dictionary for the current validation step.
        """
        loss_dict = self.model(batch, return_loss=True, teacher_features=self._teacher_features(batch))
        for loss_name, loss_value in loss_dict.items():
            self.log(f"val/{loss_name}", loss_value.detach(), sync_dist=True)
        return loss_dict
