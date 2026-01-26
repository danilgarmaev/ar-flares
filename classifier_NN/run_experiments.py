"""Official experiment runner entrypoint.

This package accumulated a number of one-off experiment scripts over time.
To keep the top-level namespace tidy while preserving existing workflows,
the implementations live under `classifier_NN.legacy`.

This module stays as the stable entrypoint used by Compute Canada SLURM:
    python -m classifier_NN.run_experiments
"""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    # Run the legacy module as if it were executed directly.
    runpy.run_module("classifier_NN.legacy.run_experiments", run_name="__main__")
