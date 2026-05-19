"""Compatibility wrapper for classifier_NN.experiments.run_experiments_v2."""

from __future__ import annotations

import runpy

from ..experiments.run_experiments_v2 import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("classifier_NN.experiments.run_experiments_v2", run_name="__main__")
