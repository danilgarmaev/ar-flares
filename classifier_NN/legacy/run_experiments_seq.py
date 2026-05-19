"""Compatibility wrapper for classifier_NN.experiments.run_experiments_seq."""

from __future__ import annotations

import runpy

from ..experiments.run_experiments_seq import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("classifier_NN.experiments.run_experiments_seq", run_name="__main__")
