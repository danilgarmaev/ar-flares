from __future__ import annotations

import json


def add_norm_cli_args(ap) -> None:
    ap.add_argument(
        "--norm-2d",
        type=str,
        default=None,
        help="Normalization for repo-defined 2D CNN blocks: batchnorm, groupnorm, identity, custom, or custom:<name>",
    )
    ap.add_argument(
        "--norm-3d",
        type=str,
        default=None,
        help="Normalization for repo-defined 3D CNN blocks: batchnorm, groupnorm, identity, custom, or custom:<name>",
    )
    ap.add_argument(
        "--norm-token",
        type=str,
        default=None,
        help="Normalization for token-feature layers: layernorm, identity, custom, or custom:<name>",
    )
    ap.add_argument(
        "--norm-group-count",
        type=int,
        default=None,
        help="Group count when using groupnorm for repo-defined CNN blocks",
    )
    ap.add_argument(
        "--norm-custom-name",
        type=str,
        default=None,
        help="Registered custom norm name to use when any norm selection is 'custom'",
    )
    ap.add_argument(
        "--norm-custom-params-json",
        type=str,
        default=None,
        help="Optional JSON object passed to registered custom norm builders",
    )
    ap.add_argument(
        "--learnable-scaler",
        action="store_true",
        default=None,
        help="Replace model normalization layers using an external learnable_scaler.py module",
    )
    ap.add_argument(
        "--learnable-scaler-module",
        type=str,
        default=None,
        help="Python module containing LearnableScaler and replace_norm (default: learnable_scaler)",
    )
    ap.add_argument(
        "--learnable-scaler-class",
        type=str,
        default=None,
        help="Class name in the learnable scaler module (default: LearnableScaler)",
    )
    ap.add_argument(
        "--learnable-scaler-lr",
        type=float,
        default=None,
        help="Learning rate for LearnableScaler parameters (default: 1e-3 when enabled)",
    )


def apply_norm_cli_overrides(args, overrides: dict) -> None:
    for key in ("norm_2d", "norm_3d", "norm_token", "norm_group_count", "norm_custom_name"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value

    if getattr(args, "learnable_scaler", None) is not None:
        overrides["use_learnable_scaler"] = bool(args.learnable_scaler)
    for key in ("learnable_scaler_module", "learnable_scaler_class"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "learnable_scaler_lr", None) is not None:
        overrides["learnable_scaler_lr"] = float(args.learnable_scaler_lr)

    params_json = getattr(args, "norm_custom_params_json", None)
    if params_json is not None:
        parsed = json.loads(params_json)
        if not isinstance(parsed, dict):
            raise ValueError("--norm-custom-params-json must decode to a JSON object")
        overrides["norm_custom_params"] = parsed
