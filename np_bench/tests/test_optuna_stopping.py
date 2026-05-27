from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from NeighborCache.region_local_threshold import optuna_stopping as tuner
from NeighborCache.region_local_threshold.stopping_mechanism import StopConfig


class _FakeTrial:
    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


def _write_toy_npz(path: Path, *, n_per_class: int = 160, dim: int = 8) -> None:
    rng = np.random.default_rng(123)
    h0 = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, dim))
    h1 = rng.normal(loc=0.6, scale=1.0, size=(n_per_class, dim))
    X = np.vstack([h0, h1]).astype(np.float32)
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=np.int32), np.ones(n_per_class, dtype=np.int32)]
    )
    region = np.zeros(2 * n_per_class, dtype=np.int64)
    np.savez(path, emb=X, label=y, global_cluster=region)


def _small_params() -> dict:
    return {
        "stop_check_every": 1,
        "stop_window": 3,
        "stop_patience": 2,
        "stop_eps_tpr": 1e-3,
        "stop_eps_fpr": 1e-3,
        "stop_eps_tau": 1e-3,
        "stop_fpr_margin": 0.0,
        "n_monitor_h0": 8,
        "n_monitor_h1": 8,
        "online_init_h0": 10,
        "online_init_h1": 10,
        "online_batch_size": 16,
        "online_mem_cap": 100,
        "online_update_mode": "refit",
        "online_hill_lr": 0.1,
    }


def test_sample_stopping_params_covers_required_fields() -> None:
    params = tuner.sample_stopping_params(_FakeTrial())
    assert set(tuner.STOPPING_PARAM_NAMES) <= set(params)

    fixed_margin_params = tuner.sample_stopping_params(
        _FakeTrial(),
        tune_stop_fpr_margin=False,
        fixed_stop_fpr_margin=0.007,
    )
    assert fixed_margin_params["stop_fpr_margin"] == pytest.approx(0.007)


def _score_args(selection_mode: str = "strict_np") -> argparse.Namespace:
    return argparse.Namespace(
        alpha=0.05,
        constraint_metric="p95_fpr",
        selection_mode=selection_mode,
        hard_constraint=True,
        lambda_cost=0.05,
        lambda_tau=0.1,
        lambda_failure=0.25,
        lambda_fpr=100.0,
        lambda_margin=1.0,
    )


def test_strict_np_penalizes_raw_alpha_violation_even_with_margin() -> None:
    args = _score_args("strict_np")
    metrics = {
        "mean_tpr": 0.9,
        "mean_fpr": 0.0571,
        "max_fpr": 0.0571,
        "p95_fpr": 0.0571,
        "mean_normalized_steps_until_stop": 0.0,
        "tau_p95_abs_slope": 0.0,
        "stop_success_rate": 1.0,
    }
    value = tuner.score_aggregate_metrics(args, {"stop_fpr_margin": 0.00928}, metrics)
    assert value == -1e9
    assert metrics["raw_alpha_constraint_passed"] is False
    assert metrics["relaxed_margin_constraint_passed"] is True
    assert metrics["constraint_passed"] is False
    assert metrics["final_constraint_used"] == "p95_fpr <= alpha"


def test_relaxed_margin_mode_allows_margin_but_penalizes_it() -> None:
    args = _score_args("relaxed_margin")
    metrics = {
        "mean_tpr": 0.9,
        "mean_fpr": 0.0571,
        "max_fpr": 0.0571,
        "p95_fpr": 0.0571,
        "mean_normalized_steps_until_stop": 0.0,
        "tau_p95_abs_slope": 0.0,
        "stop_success_rate": 1.0,
    }
    value = tuner.score_aggregate_metrics(args, {"stop_fpr_margin": 0.00928}, metrics)
    assert value == pytest.approx(0.9 - 0.00928)
    assert metrics["raw_alpha_constraint_passed"] is False
    assert metrics["relaxed_margin_constraint_passed"] is True
    assert metrics["constraint_passed"] is True
    assert metrics["final_constraint_used"] == "p95_fpr <= alpha + stop_fpr_margin"


def test_strict_np_penalizes_large_fpr_violation() -> None:
    args = _score_args("strict_np")
    metrics = {
        "mean_tpr": 1.0,
        "mean_fpr": 0.20,
        "max_fpr": 0.20,
        "p95_fpr": 0.20,
        "mean_normalized_steps_until_stop": 0.0,
        "tau_p95_abs_slope": 0.0,
        "stop_success_rate": 1.0,
    }
    value = tuner.score_aggregate_metrics(args, {"stop_fpr_margin": 0.0}, metrics)
    assert value == -1e9
    assert metrics["constraint_passed"] is False


def test_monitor_samples_are_not_reused_in_final_eval(tmp_path: Path) -> None:
    data = tmp_path / "toy.npz"
    _write_toy_npz(data, n_per_class=160)
    args = tuner.parse_args(
        [
            "--data",
            str(data),
            "--region_key",
            "global_cluster",
            "--alpha",
            "0.05",
            "--tau_mode",
            "global",
            "--n_train",
            "60",
            "--n_calib",
            "40",
            "--n_eval",
            "50",
            "--out_dir",
            str(tmp_path / "out"),
        ]
    )
    args.hard_constraint = True
    result = tuner.run_single_seed_eval(args, _small_params(), seed=0)
    assert result["monitor_final_eval_overlap"] == 0
    assert result["train_final_eval_overlap"] == 0
    assert result["calib_final_eval_overlap"] == 0
    assert result["n_eval_h0"] > 0
    assert result["n_eval_h1"] > 0


def test_optuna_stopping_smoke_writes_outputs(tmp_path: Path) -> None:
    pytest.importorskip("optuna")

    data = tmp_path / "toy.npz"
    _write_toy_npz(data, n_per_class=220, dim=4)
    out_dir = tmp_path / "optuna_out"
    storage = f"sqlite:///{out_dir / 'study.db'}"

    tuner.main(
        [
            "--data",
            str(data),
            "--region_key",
            "global_cluster",
            "--alpha",
            "0.05",
            "--tau_mode",
            "global",
            "--n_trials",
            "2",
            "--seed",
            "42",
            "--eval_seeds",
            "0",
            "--n_train",
            "80",
            "--n_calib",
            "40",
            "--n_eval",
            "80",
            "--out_dir",
            str(out_dir),
            "--study_name",
            "smoke_stopping",
            "--storage",
            storage,
        ]
    )

    assert out_dir.exists()
    assert (out_dir / "optuna_trials.csv").exists()
    assert (out_dir / "best_params.json").exists()
    assert (out_dir / "best_strict_params.json").exists()
    assert (out_dir / "best_strict_trial_metrics.json").exists()
    assert (out_dir / "best_relaxed_params.json").exists()
    assert (out_dir / "best_relaxed_trial_metrics.json").exists()

    best_params = json.loads((out_dir / "best_params.json").read_text())
    for field in StopConfig.__dataclass_fields__:
        assert field in best_params

    with (out_dir / "optuna_trials.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "constraint_passed" in rows[0]
    assert "raw_alpha_constraint_passed" in rows[0]
    assert "relaxed_margin_constraint_passed" in rows[0]
    assert "final_constraint_used" in rows[0]
    assert "alpha_plus_margin" in rows[0]
    assert rows[0]["selection_mode"] == "strict_np"

    best_metrics = json.loads((out_dir / "best_trial_metrics.json").read_text())
    assert best_metrics["monitor_final_eval_overlap"] == 0

    relaxed_metrics = json.loads((out_dir / "best_relaxed_trial_metrics.json").read_text())
    if relaxed_metrics.get("relaxed_margin_constraint_passed"):
        assert relaxed_metrics["selection_mode"] == "relaxed_margin"
        assert relaxed_metrics["final_constraint_used"] == "p95_fpr <= alpha + stop_fpr_margin"

    summary = json.loads((out_dir / "study_summary.json").read_text())
    assert summary["selection_mode"] == "strict_np"
    assert summary["primary_constraint"] == "p95_fpr <= alpha"
    assert "best_strict_trial_number" in summary
    assert "best_relaxed_trial_number" in summary
