# classifier_NN

`classifier_NN` is the active training package for this repo.

## Supported Commands

```bash
python -m classifier_NN.smoke_test
python -m classifier_NN.train
python -m classifier_NN.run_experiments
```

Supported experiment modules live under `classifier_NN.experiments`.

The `classifier_NN.legacy` namespace remains available as a compatibility layer for older scripts and SLURM commands.

## Package Layout

- `config.py`: path resolution and training defaults
- `datasets.py`: WebDataset loading, labels, and temporal sampling
- `models.py`: active model builders and normalization hooks
- `train.py`: training and evaluation loop
- `experiments/`: supported experiment runners
- `legacy/`: compatibility wrappers plus historical runner docs

## Notes

- The default smoke test is data-free.
- If local shards exist, the smoke test will also validate one real dataloader batch.
- Dataset paths should be provided through the `AR_FLARES_*` environment variables documented in the root README and `docs/COLLABORATOR_SETUP.md`.
