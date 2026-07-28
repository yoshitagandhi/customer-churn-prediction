"""Minimal setuptools configuration for installing the project as a package."""

from pathlib import Path

from setuptools import find_packages, setup

_REQUIREMENTS_FILE = Path(__file__).resolve().parent / "requirements.txt"
_requirements = _REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()

setup(
    name="customer-churn-prediction",
    version="0.1.0",
    description="End-to-end machine learning project for predicting customer churn.",
    python_requires=">=3.12",
    packages=find_packages(include=["src", "src.*", "configs", "configs.*", "app", "app.*"]),
    install_requires=_requirements,
)
