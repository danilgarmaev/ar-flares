"""Official experiment runner entrypoint.

This package accumulated a number of one-off experiment scripts over time.
To keep the top-level namespace tidy while preserving existing workflows,
the supported implementations now live under `classifier_NN.experiments`.

This module stays as the stable entrypoint used by Compute Canada SLURM:
    python -m classifier_NN.run_experiments
"""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("classifier_NN.experiments.run_experiments", run_name="__main__")
