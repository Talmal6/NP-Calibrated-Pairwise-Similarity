"""Reusable online stopping simulation for region-local threshold experiments."""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import numpy as np

from np_bench.methods.base import OnlineBaseMethod

from .evaluation import _select_tau, apply_threshold
from .splits import GlobalSplit
from .stopping_mechanism import OnlineStopper, StopConfig


def sample_monitor_from_global_eval(
    gs: GlobalSplit,
    *,
    n_monitor_h0: int,
    n_monitor_h1: int,
    seed: int,
) -> tuple[GlobalSplit, np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    h0_eval = np.asarray(gs.H0_eval, dtype=np.int64).copy()
    h1_eval = np.asarray(gs.H1_eval, dtype=np.int64).copy()
    rng.shuffle(h0_eval)
    rng.shuffle(h1_eval)

    n0 = int(min(max(n_monitor_h0, 0), h0_eval.size))
    n1 = int(min(max(n_monitor_h1, 0), h1_eval.size))

    h0_monitor = h0_eval[:n0]
    h1_monitor = h1_eval[:n1]
    h0_eval_rem = h0_eval[n0:]
    h1_eval_rem = h1_eval[n1:]

    gs_out = GlobalSplit(
        H0_train=gs.H0_train,
        H1_train=gs.H1_train,
        H0_calib=gs.H0_calib,
        H1_calib=gs.H1_calib,
        H0_eval=h0_eval_rem,
        H1_eval=h1_eval_rem,
    )
    stats = {
        "monitor_h0": int(h0_monitor.size),
        "monitor_h1": int(h1_monitor.size),
        "eval_h0_remaining": int(h0_eval_rem.size),
        "eval_h1_remaining": int(h1_eval_rem.size),
    }
    return gs_out, h0_monitor, h1_monitor, stats


def sample_monitor_from_region_splits(
    splits: List[Any],
    *,
    n_monitor_h0: int,
    n_monitor_h1: int,
    min_h0_eval: int,
    min_h1_eval: int,
    seed: int,
) -> tuple[List[Any], np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)

    h0_candidates: List[int] = []
    h1_candidates: List[int] = []
    for s in splits:
        h0_ev = np.asarray(s.H0_eval, dtype=np.int64)
        h1_ev = np.asarray(s.H1_eval, dtype=np.int64)
        h0_excess = max(0, int(h0_ev.size) - int(min_h0_eval))
        h1_excess = max(0, int(h1_ev.size) - int(min_h1_eval))
        if h0_excess > 0:
            idx0 = h0_ev.copy()
            rng.shuffle(idx0)
            h0_candidates.extend(idx0[:h0_excess].tolist())
        if h1_excess > 0:
            idx1 = h1_ev.copy()
            rng.shuffle(idx1)
            h1_candidates.extend(idx1[:h1_excess].tolist())

    rng.shuffle(h0_candidates)
    rng.shuffle(h1_candidates)

    n0 = int(min(max(n_monitor_h0, 0), len(h0_candidates)))
    n1 = int(min(max(n_monitor_h1, 0), len(h1_candidates)))

    h0_monitor = np.asarray(h0_candidates[:n0], dtype=np.int64)
    h1_monitor = np.asarray(h1_candidates[:n1], dtype=np.int64)

    h0_monitor_set = set(int(i) for i in h0_monitor.tolist())
    h1_monitor_set = set(int(i) for i in h1_monitor.tolist())

    splits_out: List[Any] = []
    for s in splits:
        h0_ev = np.asarray(s.H0_eval, dtype=np.int64)
        h1_ev = np.asarray(s.H1_eval, dtype=np.int64)
        h0_keep = h0_ev[~np.isin(h0_ev, list(h0_monitor_set))]
        h1_keep = h1_ev[~np.isin(h1_ev, list(h1_monitor_set))]
        splits_out.append(
            type(s)(
                rid=int(s.rid),
                H0_train=s.H0_train,
                H1_train=s.H1_train,
                H0_calib=s.H0_calib,
                H1_calib=s.H1_calib,
                H0_eval=h0_keep,
                H1_eval=h1_keep,
            )
        )

    stats = {
        "monitor_h0": int(h0_monitor.size),
        "monitor_h1": int(h1_monitor.size),
        "eval_h0_remaining": int(sum(int(s.H0_eval.size) for s in splits_out)),
        "eval_h1_remaining": int(sum(int(s.H1_eval.size) for s in splits_out)),
    }
    return splits_out, h0_monitor, h1_monitor, stats


def run_online_stopping(
    X_main: np.ndarray,
    y: np.ndarray,
    *,
    h0_train_idx: np.ndarray,
    h1_train_idx: np.ndarray,
    h0_calib_idx: np.ndarray,
    h0_monitor_idx: np.ndarray,
    h1_monitor_idx: np.ndarray,
    alpha: float,
    tie_mode: str,
    tau_guardrail: str,
    tau_guardrail_delta: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Optional[OnlineBaseMethod], List[Dict[str, Any]], Dict[str, Any], np.ndarray, np.ndarray]:
    empty_idx = np.array([], dtype=np.int64)
    if h0_train_idx.size == 0 or h1_train_idx.size == 0:
        return None, [], {
            "stopped": False,
            "reason": "empty_train",
        }, empty_idx, empty_idx
    if h0_calib_idx.size == 0:
        return None, [], {
            "stopped": False,
            "reason": "empty_calib_h0",
        }, empty_idx, empty_idx
    if h0_monitor_idx.size == 0 or h1_monitor_idx.size == 0:
        return None, [], {
            "stopped": False,
            "reason": "empty_monitor",
        }, empty_idx, empty_idx

    rng = np.random.default_rng(seed)
    h0_idx = np.asarray(h0_train_idx, dtype=np.int64).copy()
    h1_idx = np.asarray(h1_train_idx, dtype=np.int64).copy()
    rng.shuffle(h0_idx)
    rng.shuffle(h1_idx)

    n_init_h0 = int(min(max(args.online_init_h0, 0), h0_idx.size))
    n_init_h1 = int(min(max(args.online_init_h1, 0), h1_idx.size))
    if n_init_h0 == 0 or n_init_h1 == 0:
        return None, [], {
            "stopped": False,
            "reason": "empty_init",
        }, empty_idx, empty_idx

    init_h0_idx = h0_idx[:n_init_h0]
    init_h1_idx = h1_idx[:n_init_h1]
    rem_h0_idx = h0_idx[n_init_h0:]
    rem_h1_idx = h1_idx[n_init_h1:]
    used_h0_parts: List[np.ndarray] = [init_h0_idx.astype(np.int64, copy=False)]
    used_h1_parts: List[np.ndarray] = [init_h1_idx.astype(np.int64, copy=False)]

    online_update_mode = str(args.online_update_mode)
    method_update_mode = "refit" if online_update_mode == "reservoir" else online_update_mode
    online = OnlineBaseMethod(
        mem_cap_H0=int(args.online_mem_cap),
        mem_cap_H1=int(args.online_mem_cap),
        update_mode=method_update_mode,
        update_every=1,
        hill_lr=float(args.online_hill_lr),
        seed=seed,
    )
    online.initialize(X_main[init_h0_idx], X_main[init_h1_idx])

    check_every = max(1, int(args.stop_check_every))
    stopper = OnlineStopper(
        StopConfig(
            stop_check_every=check_every,
            stop_window=max(1, int(args.stop_window)),
            stop_patience=max(1, int(args.stop_patience)),
            stop_eps_tpr=float(args.stop_eps_tpr),
            stop_eps_fpr=float(args.stop_eps_fpr),
            stop_eps_tau=float(args.stop_eps_tau),
            stop_fpr_margin=float(args.stop_fpr_margin),
            alpha=float(alpha),
        )
    )

    rem_idx = (
        np.concatenate([rem_h0_idx, rem_h1_idx])
        if rem_h0_idx.size + rem_h1_idx.size > 0
        else np.array([], dtype=np.int64)
    )
    rng.shuffle(rem_idx)

    history_rows: List[Dict[str, Any]] = []
    total_updates = 0
    stopped = False
    stop_reason = "stream_exhausted"
    last_tau = float("inf")
    samples_streamed = 0
    samples_init = int(init_h0_idx.size + init_h1_idx.size)
    total_stream_available = int(rem_idx.size)

    def _checkpoint(checkpoint_idx: int) -> bool:
        nonlocal last_tau
        sc0_cal = np.asarray(online.score(X_main[h0_calib_idx]), dtype=np.float32).reshape(-1)
        tau = _select_tau(
            sc0_cal,
            alpha=float(alpha),
            tie_mode=tie_mode,
            guardrail=tau_guardrail,
            guardrail_delta=tau_guardrail_delta,
        )
        if not np.isfinite(tau):
            tau = float(np.quantile(sc0_cal, 1.0 - float(alpha)))
        last_tau = float(tau)

        sc0_mon = np.asarray(online.score(X_main[h0_monitor_idx]), dtype=np.float32).reshape(-1)
        sc1_mon = np.asarray(online.score(X_main[h1_monitor_idx]), dtype=np.float32).reshape(-1)
        p0 = apply_threshold(sc0_mon, tau, tie_mode)
        p1 = apply_threshold(sc1_mon, tau, tie_mode)
        fpr_monitor = float(np.mean(p0 == 1))
        tpr_monitor = float(np.mean(p1 == 1))

        entry = stopper.update(checkpoint_idx, tpr_monitor, fpr_monitor, float(tau))
        history_rows.append(
            {
                "checkpoint": int(entry.checkpoint),
                "tpr_monitor": float(entry.tpr_monitor),
                "fpr_monitor": float(entry.fpr_monitor),
                "tau": float(entry.tau),
                "slope_tpr": float(entry.slope_tpr),
                "slope_fpr": float(entry.slope_fpr),
                "slope_tau": float(entry.slope_tau),
                "condition_passed": bool(entry.condition_passed),
                "stop_streak": int(entry.stop_streak),
                "should_stop": bool(entry.should_stop),
            }
        )
        return bool(entry.should_stop)

    if rem_idx.size == 0:
        stopped = _checkpoint(0)
        stop_reason = "no_stream_data"
    else:
        batch_size = int(max(1, args.online_batch_size))
        for start in range(0, rem_idx.size, batch_size):
            end = min(start + batch_size, rem_idx.size)
            idx = rem_idx[start:end]
            online.update(X_main[idx], y[idx])
            batch_y = np.asarray(y[idx], dtype=np.int32).reshape(-1)
            used_h0_parts.append(idx[batch_y == 0].astype(np.int64, copy=False))
            used_h1_parts.append(idx[batch_y == 1].astype(np.int64, copy=False))
            total_updates += 1
            samples_streamed += int(idx.size)
            if total_updates % check_every == 0 or end == rem_idx.size:
                if _checkpoint(total_updates):
                    stopped = True
                    stop_reason = "stability_reached"
                    break

    used_h0_train_idx = (
        np.concatenate([p for p in used_h0_parts if p.size > 0]).astype(np.int64, copy=False)
        if any(p.size > 0 for p in used_h0_parts)
        else empty_idx
    )
    used_h1_train_idx = (
        np.concatenate([p for p in used_h1_parts if p.size > 0]).astype(np.int64, copy=False)
        if any(p.size > 0 for p in used_h1_parts)
        else empty_idx
    )
    original_h0_train = int(h0_train_idx.size)
    original_h1_train = int(h1_train_idx.size)
    used_h0_train = int(used_h0_train_idx.size)
    used_h1_train = int(used_h1_train_idx.size)
    original_total_train = int(original_h0_train + original_h1_train)
    used_total_train = int(used_h0_train + used_h1_train)
    summary = {
        "stopped": bool(stopped),
        "reason": str(stop_reason),
        "updates": int(total_updates),
        "final_tau": float(last_tau),
        "history_len": int(len(history_rows)),
        "samples_init": int(samples_init),
        "samples_streamed": int(samples_streamed),
        "samples_total_used": int(samples_init + samples_streamed),
        "samples_stream_available": int(total_stream_available),
        "used_h0_train": used_h0_train,
        "used_h1_train": used_h1_train,
        "used_total_train": used_total_train,
        "original_h0_train": original_h0_train,
        "original_h1_train": original_h1_train,
        "original_total_train": original_total_train,
        "train_fraction": float(used_total_train / original_total_train) if original_total_train > 0 else float("nan"),
    }
    return online, history_rows, summary, used_h0_train_idx, used_h1_train_idx


_sample_monitor_from_global_eval = sample_monitor_from_global_eval
_sample_monitor_from_region_splits = sample_monitor_from_region_splits
_run_online_stopping = run_online_stopping
