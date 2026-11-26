# Copyright (C) 2026 Xiaomi Corporation.
# Modified for Flow2Act: use the caller's default W&B account.
"""Training entry point for the MiBot framework.

Uses Hydra for config management and PyTorch Lightning for training.
Configures the model, data module, trainer, and launches training.
"""

import logging
import json
from pathlib import Path
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from mmengine import Config, DATASETS
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
import torch.distributed as dist
from mibot.models import MIMODEL

from mibot.utils.cfg_utils_flow2act import helper
from mibot.utils.optimizer_progress_flow2act import OptimizerStepProgressFlow2Act

import mibot.data.datamodule.base_datamodule_flow2act
import mibot.models.runner.base_runner_flow2act
import mibot.models.VLA.flow2act

warnings.filterwarnings("ignore")


def prepare(cfg: Dict[str, Any]) -> Tuple[Config, LightningDataModule, LightningModule, List[Any]]:
    """Prepare the training environment: resolve config, build modules and trainer.

    Steps:
        1. Resolve the OmegaConf config and set seeds / strategy.
        2. Build the data module from the registry.
        3. Build the model runner from the registry.
        4. Configure callbacks (ModelSummary, ModelCheckpoint) and logger (Wandb).

    Args:
        cfg: Raw Hydra/OmegaConf configuration.

    Returns:
        Tuple of (resolved_config, datamodule, model, logger_list).
    """
    ######################### Process config ###########################
    # Main policy training exports scalar losses only. Dense depth images belong
    # to the separate frozen-policy probe; no feature/motion image callback runs.
    cfg.model.params.visualization.every_n_steps = 0
    cfg.model.params.visualization.output_dir = None
    checkpoint_experiment_config = OmegaConf.to_container(cfg, resolve=True)
    cfg: Config = helper(cfg)

    ########################## Build modules ###########################
    # dataset
    datamodule: LightningDataModule = DATASETS.build(cfg.data)

    # model
    cfg.model.params.checkpoint_experiment_config = checkpoint_experiment_config
    cfg.model.params.checkpoint_action_stats = json.loads(Path(cfg.data.params.train_datasets.stats_path).read_text())
    model: LightningModule = MIMODEL.build(cfg.model)

    ########################## Build trainer ###########################
    # callbacks
    cfg.trainer["callbacks"] = [ModelSummary(max_depth=2)]
    if cfg.trainer.get("enable_progress_bar", True):
        cfg.trainer["callbacks"].append(OptimizerStepProgressFlow2Act())
    save_interval = cfg.trainer.pop("save_interval", 10000)
    if cfg.trainer.get("enable_checkpointing", True):
        three_fs = hasattr(cfg.trainer.strategy,"checkpoint_namespace_root")
        cfg.trainer["callbacks"].append(ModelCheckpoint(
            filename="step-{step:07d}_flow2act",
            save_top_k=3 if three_fs else -1,
            # EFS supports symlinks; avoid writing the complete ZeRO optimizer
            # a second time merely to expose the same update as last.ckpt.
            save_last="link",
            monitor="step" if three_fs else None,
            mode="max",
            every_n_train_steps=save_interval,
            dirpath=cfg.trainer.default_root_dir,
            enable_version_counter=False,
        ))

    # logger
    logger_backend = cfg.trainer.pop("logger_backend", "wandb")
    if logger_backend == "wandb":
        logger: List[Any] = [WandbLogger(
            project=cfg.trainer.pop("project"),
            name=cfg.trainer.pop("exp_name"),
            save_dir=cfg.trainer.default_root_dir,
            config=cfg,
        )]
    elif logger_backend == "csv":
        cfg.trainer.pop("project")
        cfg.trainer.pop("exp_name")
        # Native version directories preserve earlier logs on explicit resume.
        logger = [CSVLogger(save_dir=cfg.trainer.default_root_dir, name="csv",
                            flush_logs_every_n_steps=1)]
    else:
        raise ValueError("trainer.logger_backend must be wandb or csv")

    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    return (cfg, datamodule, model, logger)


@hydra.main(config_path="../configs", config_name="flow2act", version_base=None)
@record
def main(cfg: DictConfig) -> None:
    """Hydra entry point: prepare and launch training."""
    try:
        cfg, datamodule, model, logger = prepare(cfg)
        ckpt_path: Optional[str] = cfg.trainer.pop("ckpt_path", None)
        trainer = Trainer(**cfg.trainer, logger=logger)
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    finally:
        training_failed = sys.exc_info()[0] is not None
        if dist.is_available() and dist.is_initialized():
            try:
                # Lightning 2.5.3 tears down the strategy without destroying
                # its distributed groups. Clean them after checkpoint/log hooks.
                dist.destroy_process_group()
            except Exception:
                if not training_failed:
                    raise
                logging.getLogger(__name__).exception(
                    "Distributed cleanup failed; preserving the original training exception."
                )


if __name__ == "__main__":
    main()
