#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


RE_TS_SPACE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})_")
RE_TS_COMPACT = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{6})_")


@dataclass(frozen=True)
class RunRecord:
    regime: str
    run_dir: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    run_dt: datetime | None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Some historical outputs have trailing extra text after a valid JSON
            # object. Parse the first JSON value and ignore the rest.
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text.lstrip())
            return obj if isinstance(obj, dict) else None
    except FileNotFoundError:
        return None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _parse_dt(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _dt_from_folder_name(folder_name: str) -> datetime | None:
    m = RE_TS_SPACE.match(folder_name)
    if m:
        return _parse_dt(m.group(1))
    m = RE_TS_COMPACT.match(folder_name)
    if m:
        dt_str = f"{m.group(1)} {m.group(2)[0:2]}:{m.group(2)[2:4]}:{m.group(2)[4:6]}"
        return _parse_dt(dt_str)
    return None


def _dt_from_metrics(metrics: dict[str, Any]) -> datetime | None:
    date_str = metrics.get("date")
    if not isinstance(date_str, str):
        return None
    return _parse_dt(date_str)


def _classify_regime(cfg: dict[str, Any]) -> str:
    min_flare_class = str(cfg.get("min_flare_class") or "C").upper()
    balance_classes = bool(cfg.get("balance_classes"))

    if min_flare_class == "M":
        target_neg_none = cfg.get("target_neg_none")
        target_neg_c = cfg.get("target_neg_c")
        if isinstance(target_neg_none, int) or isinstance(target_neg_c, int):
            none_k = "NA" if target_neg_none is None else str(int(target_neg_none / 1000))
            c_k = "NA" if target_neg_c is None else str(int(target_neg_c / 1000))
            return f"mplus_mild_none{none_k}k_c{c_k}k"
        return "mplus_bal" if balance_classes else "mplus_full"

    # Default: C+
    target_neg_total = cfg.get("target_neg_total")
    if isinstance(target_neg_total, int):
        return f"cplus_mild{int(target_neg_total / 1000)}k"
    return "cplus_bal" if balance_classes else "cplus_full"


def _is_swin_a1_224_seed0(cfg: dict[str, Any], metrics: dict[str, Any]) -> bool:
    if cfg.get("backbone") != "swin_tiny_patch4_window7_224":
        return False
    if int(cfg.get("image_size", -1)) != 224:
        return False
    if int(cfg.get("seed", -1)) != 0:
        return False
    # A1 runs are single-frame by default; keep this permissive.
    run_id = cfg.get("run_id")
    if isinstance(run_id, str) and "A1-224" not in run_id:
        return False
    # Basic sanity check: metrics should at least have AUC.
    if "AUC" not in metrics:
        return False
    return True


def _iter_run_records(results_dir: Path) -> Iterable[RunRecord]:
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        cfg = _load_json(run_dir / "config.json")
        metrics = _load_json(run_dir / "metrics.json")
        if cfg is None or metrics is None:
            continue

        if not _is_swin_a1_224_seed0(cfg, metrics):
            continue

        regime = _classify_regime(cfg)
        run_dt = _dt_from_metrics(metrics) or _dt_from_folder_name(run_dir.name)
        yield RunRecord(regime=regime, run_dir=run_dir, config=cfg, metrics=metrics, run_dt=run_dt)


def _pick_latest_per_regime(records: Iterable[RunRecord]) -> dict[str, RunRecord]:
    latest: dict[str, RunRecord] = {}
    for rec in records:
        cur = latest.get(rec.regime)
        if cur is None:
            latest[rec.regime] = rec
            continue

        cur_dt = cur.run_dt or datetime.min
        rec_dt = rec.run_dt or datetime.min
        if rec_dt >= cur_dt:
            latest[rec.regime] = rec
    return latest


def _tsv_cell(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    s = str(v)
    return s.replace("\t", " ").replace("\n", " ").strip() or "NA"


def _tsv_cell_excel_metrics(v: Any) -> str:
    """Excel-friendly cell formatting for *metrics* columns.

    Uses comma as decimal separator for floats while keeping other types stable.
    """
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Keep full precision; Excel can still parse with comma decimals.
        return ("%s" % v).replace(".", ",")
    if isinstance(v, (list, dict)):
        return json.dumps(v, sort_keys=True)
    s = str(v)
    return s.replace("\t", " ").replace("\n", " ").strip() or "NA"


def _write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(_tsv_cell(row.get(c)) for c in columns) + "\n")


def _write_tsv_excel_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
    metric_columns: set[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            cells: list[str] = []
            for c in columns:
                v = row.get(c)
                if c in metric_columns:
                    cells.append(_tsv_cell_excel_metrics(v))
                else:
                    cells.append(_tsv_cell(v))
            f.write("\t".join(cells) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    out_path = results_dir / "summary.tsv"
    out_path_excel = results_dir / "summary_excel.tsv"

    records = list(_iter_run_records(results_dir))
    by_regime = _pick_latest_per_regime(records)

    expected_regimes = [
        "cplus_full",
        "cplus_mild450k",
        "cplus_bal",
        "mplus_full",
        "mplus_mild_none150k_c125k",
        "mplus_bal",
    ]
    extra_regimes = sorted(set(by_regime.keys()) - set(expected_regimes))
    ordered_regimes = expected_regimes + extra_regimes

    metric_columns = {
        "AUC",
        "PR_AUC",
        "TPR",
        "TNR",
        "TSS",
        "HSS",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "Best_threshold",
        "val_threshold_for_test",
        "fixed_threshold",
        "fixed_TSS",
        "fixed_TPR",
        "fixed_TNR",
        "fixed_HSS",
        "fixed_Precision",
        "fixed_Recall",
        "fixed_F1",
        "fixed_Accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "date",
    }

    # Put metrics early for quick scanning in spreadsheets.
    columns = [
        "regime",
        "run_dir",
        "run_id",
        "date",
        "seed",
        "min_flare_class",
        "backbone",
        "image_size",
        # --- metrics block ---
        "AUC",
        "PR_AUC",
        "TSS",
        "HSS",
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "Best_threshold",
        "val_threshold_for_test",
        "fixed_threshold",
        "fixed_TSS",
        # --- hyperparams / setup ---
        "batch_size",
        "lr",
        "epochs",
        "optimizer",
        "scheduler",
        "loss_type",
        "balance_classes",
        "balance_mode",
        "neg_keep_prob",
        "neg_keep_prob_none",
        "neg_keep_prob_c",
        "auto_set_neg_keep_probs",
        "target_neg_total",
        "target_neg_none",
        "target_neg_c",
    ]

    rows: list[dict[str, Any]] = []
    for regime in ordered_regimes:
        rec = by_regime.get(regime)
        if rec is None:
            rows.append({"regime": regime})
            continue

        rel_run_dir = rec.run_dir.relative_to(repo_root)
        row: dict[str, Any] = {"regime": regime, "run_dir": str(rel_run_dir)}
        row.update({k: rec.config.get(k) for k in columns if k in rec.config})
        row.update({k: rec.metrics.get(k) for k in columns if k in rec.metrics})
        # Prefer the metrics timestamp if present.
        row["date"] = rec.metrics.get("date") or (rec.run_dt.isoformat(sep=" ") if rec.run_dt else None)
        rows.append(row)

    _write_tsv(out_path, rows, columns)
    _write_tsv_excel_metrics(out_path_excel, rows, columns, metric_columns)

    missing = [r for r in expected_regimes if r not in by_regime]
    print(f"Wrote {out_path} ({len(rows)} rows).")
    print(f"Wrote {out_path_excel} ({len(rows)} rows).")
    if missing:
        print("Missing regimes:")
        for r in missing:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
