"""Compatibility wrapper for classifier_NN.experiments.run_resume_A1_20_epochs."""

from __future__ import annotations

import runpy

from ..experiments.run_resume_A1_20_epochs import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("classifier_NN.experiments.run_resume_A1_20_epochs", run_name="__main__")
