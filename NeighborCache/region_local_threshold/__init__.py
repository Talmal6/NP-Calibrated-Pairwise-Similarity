"""Region-local threshold benchmark package."""

from .io_helpers import resolve_npz_path, load_npz, resolve_features
from .methods import build_methods, try_fit_method, method_train_required
from .splits import RegionSplit, GlobalSplit, split_indices_per_region, split_global
from .evaluation import apply_threshold, evaluate_methods, evaluate_methods_global, aggregate_ranking
from .display import print_trial_table, print_ranking
from .cli import main

__all__ = [
    "resolve_npz_path",
    "load_npz",
    "resolve_features",
    "build_methods",
    "try_fit_method",
    "method_train_required",
    "RegionSplit",
    "GlobalSplit",
    "split_indices_per_region",
    "split_global",
    "apply_threshold",
    "evaluate_methods",
    "evaluate_methods_global",
    "aggregate_ranking",
    "print_trial_table",
    "print_ranking",
    "main",
]
