#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from itertools import chain
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


RE_TS_SPACE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})_")
RE_TS_COMPACT = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{6})_")
RE_CADENCE = re.compile(r"cadence[-_]?([0-9]+)min", re.IGNORECASE)
# Supports both array logs (<job>_<task>) and non-array logs (<job>)
# for either .out or .err files.
RE_LOG_JOB_TASK = re.compile(r"_(\d+)(?:_(\d+))?\.(?:out|err)$")


@dataclass(frozen=True)
class RunRecord:
    regime: str
    run_dir: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    run_dt: datetime | None
    array_job_id: int | None
    array_task_id: int | None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
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


def _is_video_run(cfg: dict[str, Any], metrics: dict[str, Any]) -> bool:
    # Keep final results only.
    if not isinstance(metrics.get("AUC"), (int, float)):
        return False
    if not bool(cfg.get("use_seq", False)):
        return False

    run_id = str(cfg.get("run_id") or "")
    model_name = str(cfg.get("model_name") or "")
    haystack = " ".join([run_id, model_name]).lower()

    # Cadence-focused video experiments only.
    if "cadence" in haystack or "16frames" in haystack or "a3-16" in haystack:
        return True
    return False


def _interval_min(cfg: dict[str, Any]) -> int | None:
    stride = cfg.get("seq_stride")
    if isinstance(stride, int) and stride > 0:
        return int(stride * 12)

    run_id = str(cfg.get("run_id") or cfg.get("model_name") or "")
    m = RE_CADENCE.search(run_id)
    if m:
        return int(m.group(1))
    return None


def _temporal_context_hours(cfg: dict[str, Any]) -> float | None:
    t = cfg.get("seq_T")
    interval = _interval_min(cfg)
    if isinstance(t, int) and t > 1 and isinstance(interval, int):
        return ((t - 1) * interval) / 60.0
    return None


def _regime(cfg: dict[str, Any], interval_min: int | None) -> str:
    bb = str(cfg.get("backbone") or "unknown")
    min_class = str(cfg.get("min_flare_class") or "C").upper()
    aug = "aug" if bool(cfg.get("use_aug", False)) else "noaug"
    interval = "NA" if interval_min is None else str(interval_min)

    target_neg_none = cfg.get("target_neg_none")
    target_neg_c = cfg.get("target_neg_c")
    if min_class == "M":
        ds_tag = f"mplus_none{target_neg_none if isinstance(target_neg_none, int) else 'NA'}"
        if isinstance(target_neg_c, int):
            ds_tag += f"_c{target_neg_c}"
    else:
        ds_tag = f"cplus_none{target_neg_none if isinstance(target_neg_none, int) else 'NA'}"

    return f"{bb}_c{interval}_{min_class.lower()}plus_{aug}_{ds_tag}"


def _build_run_dir_to_slurm_map(repo_root: Path) -> dict[str, tuple[int | None, int | None]]:
    logs_dir = repo_root / "logs"
    mapping: dict[str, tuple[int | None, int | None]] = {}
    if not logs_dir.exists():
        return mapping

    log_paths = sorted(chain(logs_dir.glob("arfl-*.out"), logs_dir.glob("arfl-*.err")))

    for log_path in log_paths:
        m = RE_LOG_JOB_TASK.search(log_path.name)
        if not m:
            continue

        job_id = int(m.group(1))
        task_id = int(m.group(2)) if m.group(2) is not None else None
        run_dir_name: str | None = None

        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Experiment directory:" in line:
                        run_dir = line.split("Experiment directory:", 1)[1].strip()
                        run_dir_name = Path(run_dir).name
                        break
        except OSError:
            continue

        if run_dir_name:
            # Prefer mappings that include an array task id over ones that do not.
            prev = mapping.get(run_dir_name)
            if prev is None or (prev[1] is None and task_id is not None):
                mapping[run_dir_name] = (job_id, task_id)

    return mapping


def _iter_run_records(results_dir: Path, run_to_slurm: dict[str, tuple[int | None, int | None]]) -> Iterable[RunRecord]:
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue

        cfg = _load_json(run_dir / "config.json")
        metrics = _load_json(run_dir / "metrics.json")
        if cfg is None or metrics is None:
            continue

        if not _is_video_run(cfg, metrics):
            continue

        interval = _interval_min(cfg)
        regime = _regime(cfg, interval)
        run_dt = _dt_from_metrics(metrics) or _dt_from_folder_name(run_dir.name)
        jt = run_to_slurm.get(run_dir.name)

        yield RunRecord(
            regime=regime,
            run_dir=run_dir,
            config=cfg,
            metrics=metrics,
            run_dt=run_dt,
            array_job_id=jt[0] if jt else None,
            array_task_id=jt[1] if jt else None,
        )


def _sort_key(rec: RunRecord) -> tuple[Any, ...]:
    dt = rec.run_dt or datetime.min
    job = rec.array_job_id if rec.array_job_id is not None else 10**18
    task = rec.array_task_id if rec.array_task_id is not None else 10**18
    return (dt, job, task, rec.run_dir.name)


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
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
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
    out_path = results_dir / "summary_video.tsv"
    out_path_excel = results_dir / "summary_video_excel.tsv"

    run_to_slurm = _build_run_dir_to_slurm_map(repo_root)
    records = sorted(list(_iter_run_records(results_dir, run_to_slurm)), key=_sort_key)

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
        "report_val_threshold",
        "report_val_TSS",
        "report_val_TPR",
        "report_val_TNR",
        "report_val_FPR",
        "report_val_HSS",
        "report_val_Precision",
        "report_val_Recall",
        "report_val_F1",
        "report_val_Accuracy",
        "report_0p5_threshold",
        "report_0p5_TSS",
        "report_0p5_TPR",
        "report_0p5_TNR",
        "report_0p5_FPR",
        "report_0p5_HSS",
        "report_0p5_Precision",
        "report_0p5_Recall",
        "report_0p5_F1",
        "report_0p5_Accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
    }

    columns = [
        "regime",
        "run_dir",
        "run_id",
        "date",
        "array_job_id",
        "array_task_id",
        "seed",
        "min_flare_class",
        "backbone",
        "image_size",
        "seq_T",
        "seq_stride",
        "interval_min",
        "temporal_context_hours",
        "seq_offsets",
        "use_aug",
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
        "report_val_threshold",
        "report_val_TSS",
        "report_val_TPR",
        "report_val_FPR",
        "report_val_Precision",
        "report_val_Recall",
        "report_val_F1",
        "report_0p5_threshold",
        "report_0p5_TSS",
        "report_0p5_TPR",
        "report_0p5_FPR",
        "report_0p5_Precision",
        "report_0p5_Recall",
        "report_0p5_F1",
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
    for rec in records:
        rel_run_dir = rec.run_dir.relative_to(repo_root)
        report_metrics = _load_json(rec.run_dir / 'report_metrics.json') or {}
        row: dict[str, Any] = {
            "regime": rec.regime,
            "run_dir": str(rel_run_dir),
            "array_job_id": rec.array_job_id,
            "array_task_id": rec.array_task_id,
        }
        row.update({k: rec.config.get(k) for k in columns if k in rec.config})
        row.update({k: rec.metrics.get(k) for k in columns if k in rec.metrics})
        row.update({k: report_metrics.get(k) for k in columns if k in report_metrics})
        row["interval_min"] = _interval_min(rec.config)
        row["temporal_context_hours"] = _temporal_context_hours(rec.config)
        row["date"] = rec.metrics.get("date") or (rec.run_dt.isoformat(sep=" ") if rec.run_dt else None)
        rows.append(row)

    _write_tsv(out_path, rows, columns)
    _write_tsv_excel_metrics(out_path_excel, rows, columns, metric_columns)

    print(f"Wrote {out_path} ({len(rows)} rows).")
    print(f"Wrote {out_path_excel} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
