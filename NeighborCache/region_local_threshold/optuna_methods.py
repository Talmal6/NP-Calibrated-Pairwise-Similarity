"""Convenience Optuna runner for tuning specific method families.

This wrapper reuses the existing validation-based Optuna search in
`NeighborCache.region_local_threshold.cli` and runs one study per requested
method family.

The default method list covers:

- W.Cosine -> PCAWhitenedCosine
- MLP -> Tiny MLP
- XGBoost
- LDA

Example:

    python -m NeighborCache.region_local_threshold.optuna_methods \
      --data NeighborCache/data/h1h0_final.npz \
      --region_key global_cluster \
      --alpha 0.05 \
      --tau_mode global \
      --n_trials 20 \
      --seed 42 \
      --n_train 5270 \
      --n_calib 5240 \
      --n_eval 5270 \
      --hadamard_preprocess
"""
from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence

from . import cli as threshold_cli


METHOD_ALIASES = {
    "w.cosine": "PCAWhitenedCosine",
    "wcosine": "PCAWhitenedCosine",
    "whitenedcosine": "PCAWhitenedCosine",
    "pcawhitenedcosine": "PCAWhitenedCosine",
    "w.cosine.pca": "PCAWhitenedCosine_PCA",
    "wcosine_pca": "PCAWhitenedCosine_PCA",
    "pcawhitenedcosine_pca": "PCAWhitenedCosine_PCA",
    "w.cosine.zcacor": "PCAWhitenedCosine_ZCAcor",
    "wcosine_zcacor": "PCAWhitenedCosine_ZCAcor",
    "pcawhitenedcosine_zcacor": "PCAWhitenedCosine_ZCAcor",
    "w.cosine.pcacor": "PCAWhitenedCosine_PCAcor",
    "wcosine_pcacor": "PCAWhitenedCosine_PCAcor",
    "pcawhitenedcosine_pcacor": "PCAWhitenedCosine_PCAcor",
    "mlp": "Tiny MLP",
    "tiny mlp": "Tiny MLP",
    "xgboost": "XGBoost",
    "lda": "LDA",
}

DEFAULT_METHODS = ["W.Cosine", "MLP", "XGBoost", "LDA"]


def _normalize_method_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("Method names in --methods must be non-empty")
    normalized = METHOD_ALIASES.get(text.lower(), text)
    return normalized


def _parse_methods(raw: str | None) -> List[str]:
    if raw is None or str(raw).strip() == "":
        raw = ",".join(DEFAULT_METHODS)
    methods = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            methods.append(_normalize_method_name(text))
    if not methods:
        raise ValueError("--methods must contain at least one method name")
    return methods


def _replace_or_append(argv: Sequence[str], flag: str, value: str) -> List[str]:
    out = list(argv)
    for idx, token in enumerate(out):
        if token == flag:
            if idx + 1 >= len(out):
                out.append(value)
            else:
                out[idx + 1] = value
            return out
        if token.startswith(flag + "="):
            out[idx] = f"{flag}={value}"
            return out
    out.extend([flag, value])
    return out


def _ensure_flag(argv: Sequence[str], flag: str) -> List[str]:
    out = list(argv)
    if flag not in out and not any(token.startswith(flag + "=") for token in out):
        out.append(flag)
    return out


def _build_method_argv(base_argv: Sequence[str], method_name: str, *, optuna_trials: int) -> List[str]:
    method_slug = method_name.lower().replace(" ", "_").replace(".", "_")
    method_argv = _ensure_flag(base_argv, "--enable_optuna")
    method_argv = _replace_or_append(method_argv, "--optuna_target_method", method_name)
    method_argv = _replace_or_append(method_argv, "--optuna_trials", str(int(optuna_trials)))
    method_argv = _replace_or_append(method_argv, "--run_name", f"optuna_{method_slug}")
    method_argv = _replace_or_append(method_argv, "--optuna_study_name", f"optuna_{method_slug}")
    return method_argv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method list. Use W.Cosine, MLP, XGBoost, LDA.",
    )
    extra_args, remaining = extra.parse_known_args(argv)

    base_args = threshold_cli.parse_args(remaining)
    setattr(base_args, "methods", _parse_methods(extra_args.methods))
    setattr(base_args, "_base_argv", list(remaining))
    return base_args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    base_argv = list(getattr(args, "_base_argv", []))
    optuna_trials = int(getattr(args, "n_trials", 20))
    if any(token == "--optuna_trials" or token.startswith("--optuna_trials=") for token in base_argv):
        optuna_trials = int(getattr(args, "optuna_trials", optuna_trials))

    print("[INFO] Optuna method tuner starting")
    print(f"[INFO] Methods: {', '.join(args.methods)}")

    for method_name in args.methods:
        method_argv = _build_method_argv(base_argv, method_name, optuna_trials=optuna_trials)
        print(f"[INFO] Running Optuna for {method_name}")
        threshold_cli.main(method_argv)


if __name__ == "__main__":
    main()
