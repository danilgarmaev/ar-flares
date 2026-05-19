"""Compatibility wrapper for classifier_NN.experiments.run_physics_attention."""

from __future__ import annotations

import runpy

from ..experiments.run_physics_attention import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("classifier_NN.experiments.run_physics_attention", run_name="__main__")
