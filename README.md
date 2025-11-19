# AR-Flares

A modular PyTorch pipeline for classifying Active Region (AR) solar flare activity using deep neural networks (ConvNeXt, ViT, custom CNNs) with optional optical flow and temporal sequences.

## Highlights
- Modular refactored training loop now in `classifier_refactored/`
- Fast 112x112 training mode (~4× speedup)
- Class balancing only on training split (probabilistic negative downsampling)
- Rich metrics: ROC AUC, PR AUC, Precision, Recall, F1, TSS, HSS
- Per-epoch JSONL logging + loss & accuracy curves
- Easily switch backbone (e.g. `convnext_base`, `vit_base_patch16_224`)

## Quick Start
```bash
cd classifier_refactored
python train.py
```
Edit `config.py` to change backbone, image size, epochs, etc.

## Configuration Snippet
```python
CFG = {
  "image_size": 112,
  "backbone": "convnext_base",
  "balance_classes": True,
  "neg_keep_prob": 0.25,
  "epochs": 5,
  "lr": 1e-4,
}
```

## Data
WebDataset-style tar shards expected under `data/wds_out/{train,val,test}`. Optical flow shards optional at `data/wds_flow/`.

## Results
Training outputs saved under `results/DATE_MODELNAME/`. Curves: `training_curves.png`. Per-epoch metrics: `epoch_metrics.jsonl`.

## License
MIT – see `LICENSE`.

## Citation
If you use this code, please cite the repository (add DOI later).

## Sampling Modes
Training split only:
- `balance_mode='prob'`: per-epoch probabilistic negative downsampling (diverse negatives each epoch)
- `balance_mode='fixed'`: deterministic hashed subset of negatives (stable epoch length)
Validation & Test remain imbalanced.

## Roadmap
- Sequence modeling improvements (attention aggregation)
- LoRA / lightweight fine-tuning for large backbones
- Experiment registry & reproducibility bundle

Contributions welcome!
