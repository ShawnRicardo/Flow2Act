"""Editable installation for the Flow2Act policy and geometry modules."""
from pathlib import Path
from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).resolve().parent
setup(
    name="flow2act-xm0",
    version="0.1.0",
    description="Flow2Act on Xiaomi-Robotics-0",
    long_description=(ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else "Flow2Act on XR0",
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    license="Apache-2.0",
    packages=(find_namespace_packages(where="xr0", include=["mibot", "mibot.*"])
              + find_namespace_packages(include=["flow2act", "flow2act.*"])),
    package_dir={"": "xr0", "flow2act": "flow2act"},
)
