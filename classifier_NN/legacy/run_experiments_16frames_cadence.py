"""Compatibility wrapper for classifier_NN.experiments.run_experiments_16frames_cadence."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("classifier_NN.experiments.run_experiments_16frames_cadence", run_name="__main__")
else:
    from ..experiments.run_experiments_16frames_cadence import *  # noqa: F401,F403
