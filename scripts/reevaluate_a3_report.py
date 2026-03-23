#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifier_NN.reevaluate_checkpoint import reevaluate_with_fixed_threshold


def _is_a3_cadence_run(run_dir: Path) -> bool:
    return run_dir.is_dir() and 'A3-16-cadence' in run_dir.name


def _discover_runs(results_dir: Path, mode: str) -> list[Path]:
    runs: list[Path] = []
    for run_dir in sorted(results_dir.iterdir()):
        if not _is_a3_cadence_run(run_dir):
            continue
        if not (run_dir / 'config.json').exists():
            continue
        if not (run_dir / 'best_tss.pt').exists():
            continue
        if mode == 'incomplete' and (run_dir / 'metrics.json').exists():
            continue
        runs.append(run_dir)
    return runs


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\\n')


def _best_val_threshold_from_history(run_dir: Path) -> tuple[float, int, float]:
    history_path = run_dir / 'epoch_metrics.jsonl'
    if not history_path.exists():
        raise FileNotFoundError(f'Missing epoch history: {history_path}')

    best_row: dict[str, Any] | None = None
    best_key: tuple[float, int] | None = None
    with history_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tss = float(row.get('val_best_tss', float('-inf')))
            epoch = int(row.get('epoch', -1))
            key = (tss, epoch)
            if best_key is None or key > best_key:
                best_key = key
                best_row = row

    if best_row is None:
        raise ValueError(f'No usable rows found in {history_path}')

    thr = best_row.get('val_best_threshold')
    if thr is None:
        raise ValueError(f'No val_best_threshold found in best row for {run_dir}')
    return float(thr), int(best_row['epoch']), float(best_row['val_best_tss'])


def _report_block(prefix: str, *, threshold: float, tss: float, precision: float, recall: float, f1: float, accuracy: float, tpr: float, tnr: float, hss: float) -> dict[str, Any]:
    return {
        f'{prefix}_threshold': float(threshold),
        f'{prefix}_TSS': float(tss),
        f'{prefix}_Precision': float(precision),
        f'{prefix}_Recall': float(recall),
        f'{prefix}_F1': float(f1),
        f'{prefix}_Accuracy': float(accuracy),
        f'{prefix}_TPR': float(tpr),
        f'{prefix}_TNR': float(tnr),
        f'{prefix}_FPR': float(1.0 - tnr),
        f'{prefix}_HSS': float(hss),
    }


def _report_from_metrics(prefix: str, metrics: dict[str, Any], *, threshold_key: str, tss_key: str, precision_key: str, recall_key: str, f1_key: str, accuracy_key: str, tpr_key: str, tnr_key: str, hss_key: str) -> dict[str, Any]:
    threshold = metrics.get(threshold_key)
    if threshold is None:
        raise ValueError(f'Missing {threshold_key} in metrics payload')
    return _report_block(
        prefix,
        threshold=float(threshold),
        tss=float(metrics[tss_key]),
        precision=float(metrics[precision_key]),
        recall=float(metrics[recall_key]),
        f1=float(metrics[f1_key]),
        accuracy=float(metrics[accuracy_key]),
        tpr=float(metrics[tpr_key]),
        tnr=float(metrics[tnr_key]),
        hss=float(metrics[hss_key]),
    )


def _load_existing_threshold_eval(run_dir: Path, suffix: str, expected_threshold: float) -> dict[str, Any] | None:
    metrics_path = run_dir / f'metrics_{suffix}.json'
    payload = _load_json(metrics_path)
    if payload is None:
        return None
    actual = payload.get('fixed_threshold')
    if actual is None:
        return None
    if abs(float(actual) - float(expected_threshold)) > 1e-9:
        return None
    return payload


def _val_report(run_dir: Path, *, force: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    base_metrics = _load_json(run_dir / 'metrics.json')
    val_threshold, best_epoch, best_val_tss = _best_val_threshold_from_history(run_dir)

    if (not force) and base_metrics is not None and base_metrics.get('fixed_TSS') is not None:
        payload = {
            'source': 'metrics.json',
            'best_val_epoch': int(best_epoch),
            'best_val_tss': float(best_val_tss),
        }
        payload.update(
            _report_from_metrics(
                'report_val',
                base_metrics,
                threshold_key='val_threshold_for_test' if base_metrics.get('val_threshold_for_test') is not None else 'fixed_threshold',
                tss_key='fixed_TSS',
                precision_key='fixed_Precision',
                recall_key='fixed_Recall',
                f1_key='fixed_F1',
                accuracy_key='fixed_Accuracy',
                tpr_key='fixed_TPR',
                tnr_key='fixed_TNR',
                hss_key='fixed_HSS',
            )
        )
        return payload, {'reused_metrics_json': True}

    cached = None if force else _load_existing_threshold_eval(run_dir, 'report_valthr', val_threshold)
    if cached is None:
        cached = reevaluate_with_fixed_threshold(str(run_dir), 'best_tss.pt', 'report_valthr', val_threshold)

    payload = {
        'source': 'metrics_report_valthr.json',
        'best_val_epoch': int(best_epoch),
        'best_val_tss': float(best_val_tss),
    }
    payload.update(
        _report_from_metrics(
            'report_val',
            cached,
            threshold_key='fixed_threshold',
            tss_key='fixed_TSS',
            precision_key='fixed_Precision',
            recall_key='fixed_Recall',
            f1_key='fixed_F1',
            accuracy_key='fixed_Accuracy',
            tpr_key='fixed_TPR',
            tnr_key='fixed_TNR',
            hss_key='fixed_HSS',
        )
    )
    return payload, {'reused_metrics_json': False}


def _thr05_report(run_dir: Path, *, force: bool) -> dict[str, Any]:
    cached = None if force else _load_existing_threshold_eval(run_dir, 'report_thr05', 0.5)
    if cached is None:
        cached = reevaluate_with_fixed_threshold(str(run_dir), 'best_tss.pt', 'report_thr05', 0.5)
    payload = {'source': 'metrics_report_thr05.json'}
    payload.update(
        _report_from_metrics(
            'report_0p5',
            cached,
            threshold_key='fixed_threshold',
            tss_key='fixed_TSS',
            precision_key='fixed_Precision',
            recall_key='fixed_Recall',
            f1_key='fixed_F1',
            accuracy_key='fixed_Accuracy',
            tpr_key='fixed_TPR',
            tnr_key='fixed_TNR',
            hss_key='fixed_HSS',
        )
    )
    return payload


def process_run(run_dir: Path, *, force: bool) -> Path:
    cfg = _load_json(run_dir / 'config.json') or {}
    val_payload, meta = _val_report(run_dir, force=force)
    thr05_payload = _thr05_report(run_dir, force=force)

    report = {
        'exp_dir': str(run_dir),
        'run_id': cfg.get('run_id', run_dir.name),
        'backbone': cfg.get('backbone'),
        'min_flare_class': cfg.get('min_flare_class'),
        'use_aug': cfg.get('use_aug'),
        'report_ready': True,
        **meta,
        **val_payload,
        **thr05_payload,
    }
    out_path = run_dir / 'report_metrics.json'
    _write_json(out_path, report)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--mode', choices=['all', 'incomplete'], default='all')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--count', action='store_true')
    ap.add_argument('--task-id', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    runs = _discover_runs(results_dir, args.mode)

    if args.list:
        for run_dir in runs:
            print(run_dir)
        return
    if args.count:
        print(len(runs))
        return

    if args.task_id is None:
        raise SystemExit('--task-id is required unless using --list or --count')
    if args.task_id < 0 or args.task_id >= len(runs):
        raise SystemExit(f'task-id {args.task_id} out of range for {len(runs)} runs')

    run_dir = runs[args.task_id]
    out_path = process_run(run_dir, force=args.force)
    print(f'Processed {run_dir}')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
