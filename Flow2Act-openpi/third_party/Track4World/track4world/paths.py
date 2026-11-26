"""Repository-relative paths for the vendored Track4World checkout."""

import os
from pathlib import Path


TRACK4WORLD_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_ROOT = Path(os.environ.get(
    "TRACK4WORLD_CHECKPOINTS_ROOT", str(TRACK4WORLD_ROOT / "checkpoints")
)).expanduser()
DA3_NESTED_CHECKPOINT = CHECKPOINTS_ROOT / "DA3NESTED-GIANT-LARGE-1.1"
CONVNEXT_TINY_CHECKPOINT = CHECKPOINTS_ROOT / "torchvision" / "convnext_tiny-983f1562.pth"


def require_local_path(path: Path) -> Path:
    """Return an existing repository-local path with a useful failure message."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing local Track4World dependency: {path}. "
            f"See {TRACK4WORLD_ROOT / 'WEIGHTS.md'}."
        )
    return path

