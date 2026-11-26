# Copyright (C) 2026 Xiaomi Corporation.
import os
from pathlib import Path
from typing import Any

from lightning import seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy
from mmengine import Config
from omegaconf import DictConfig, OmegaConf
from transformers.utils import logging
from mibot.utils.precision_flow2act import DeepSpeedPrecisionFlow2Act

logger = logging.get_logger(__name__)


def helper(cfg: DictConfig) -> Config:
    """Process and finalize the training configuration.

    Resolves the OmegaConf config, uses one common sampler seed across ranks,
    configures the training strategy, and injects
    optimizer/scheduler configs into the model params.

    Args:
        cfg: Raw Hydra/OmegaConf configuration object.

    Returns:
        Fully resolved ``mmengine.Config`` ready for trainer construction.
    """
    # Resolve config
    cfg = Config(OmegaConf.to_container(cfg, resolve=True))
    # DeepSpeed restores after on_fit_start; Trainer.ckpt_path can still be None there.
    cfg.model.params.resume_requested = bool(cfg.trainer.get("ckpt_path"))
    process_save_cfg(cfg)

    # Keep PL_GLOBAL_SEED common across ranks so DistributedSampler partitions one permutation.
    seed_everything(cfg.trainer.pop("seed", 42), workers=True)

    # Configure training strategy
    strategy_helper(cfg)

    # Inject optimizer & scheduler into model params so BaseRunner can access them
    cfg.model.params.optimizer = cfg.trainer.pop("optimizer")
    cfg.model.params.scheduler = cfg.trainer.pop("scheduler")

    return cfg


def strategy_helper(cfg: Config) -> Config:
    """Instantiate and attach the distributed training strategy.

    Currently only DeepSpeed is supported.

    Args:
        cfg: The configuration object to modify in-place.

    Returns:
        The configuration object with ``cfg.trainer.strategy`` replaced by
        an instantiated strategy object.
    """
    strategy_type: str = cfg.trainer.strategy.type
    if strategy_type in ("deepspeed", "deepspeed_3fs_flow2act"):
        strategy_class = DeepSpeedStrategy
        if strategy_type == "deepspeed_3fs_flow2act":
            from mibot.utils.strategy_flow2act import DeepSpeedStrategyFlow2Act
            strategy_class = DeepSpeedStrategyFlow2Act
        strategy = strategy_class(precision_plugin=DeepSpeedPrecisionFlow2Act(
            cfg.trainer.get("precision","32-true")),**cfg.trainer.strategy.params)
    else:
        raise TypeError("Unsupported strategy.")
    cfg.trainer.strategy = strategy
    return cfg


def fill_num_nodes(cfg: Config) -> None:
    """Ensure ``cfg.trainer.num_nodes`` is a positive integer.

    If the configured value is <= 0, it is auto-detected from the
    ``MLP_WORKER_NUM`` environment variable.

    Args:
        cfg: The configuration object to modify in-place.
    """
    if cfg.trainer.num_nodes > 0:
        assert cfg.trainer.num_nodes <= int(os.environ.get("MLP_WORKER_NUM", 1)), (
            "Number of nodes exceeds available workers"
        )
    else:
        cfg.trainer.num_nodes = int(os.environ.get("MLP_WORKER_NUM", 1))


def process_save_cfg(cfg: Config) -> None:
    """Create the output directory, persist the resolved config, and expose
    ``max_steps`` via an environment variable for downstream data code.

    The resolved config is dumped only inside this run's output directory as
    ``config_flow2act.yaml`` and ``config_flow2act.py``.

    Args:
        cfg: The configuration object to modify and save.
    """
    root_dir = cfg.trainer.pop("output_dir", None) or os.path.join(
        cfg.trainer.default_root_dir,
        f"project_{cfg.trainer.project}",
        cfg.trainer.exp_name,
    )
    cfg.trainer.default_root_dir = root_dir

    fill_num_nodes(cfg)

    # Expose max_steps so dataset code can read it if needed
    os.environ["_max_steps"] = str(cfg.trainer.max_steps)

    output = Path(root_dir)
    has_checkpoint = os.path.lexists(output / "last.ckpt") or any(output.glob("step-*_flow2act.ckpt"))
    if not cfg.trainer.get("ckpt_path") and has_checkpoint:
        raise FileExistsError(f"Checkpoint already exists in {root_dir}; choose a new output directory or set trainer.ckpt_path.")
    os.makedirs(root_dir, exist_ok=True)
    if int(os.environ.get("RANK", 0)) == 0:
        cfg.dump(os.path.join(root_dir, "config_flow2act.yaml"))
        cfg.dump(os.path.join(root_dir, "config_flow2act.py"))

