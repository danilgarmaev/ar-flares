"""Compatibility wrapper for classifier_NN.experiments.run_experiments_A1."""

from __future__ import annotations

import runpy

from ..experiments.run_experiments_A1 import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("classifier_NN.experiments.run_experiments_A1", run_name="__main__")
