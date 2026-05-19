"""Compatibility wrapper for classifier_NN.experiments.run_experiments_A3_framecount."""

from __future__ import annotations

import runpy

from ..experiments.run_experiments_A3_framecount import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("classifier_NN.experiments.run_experiments_A3_framecount", run_name="__main__")
