# Copyright (C) 2026 Xiaomi Corporation.
# Modified for Flow2Act: omit registration of unused legacy training routes.
"""Registry shared by the Flow2Act model and Lightning runner."""
from mmengine import Registry

MIMODEL = Registry("MIMODEL")
__all__ = ["MIMODEL"]
