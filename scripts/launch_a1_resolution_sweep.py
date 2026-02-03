"""Launcher helper for A1 resolution sweep.

Runs backbones x resolutions x seeds with early stopping and writes a summary
JSON via run_sweep().
"""

from classifier_NN.legacy.run_experiments_A1 import BACKBONES, RESOLUTIONS, run_sweep


def main():
    seeds = list(range(5))
    run_sweep(backbones=list(BACKBONES), resolutions=list(RESOLUTIONS), seeds=seeds)


if __name__ == "__main__":
    main()
