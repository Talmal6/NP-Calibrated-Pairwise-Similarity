from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.launchers import benchmark_registry as registry
from experiments.launchers.build_benchmark_tasks import (
    BenchmarkTask,
    build_main_suite,
    is_complete,
    task_to_cli_args,
    task_to_command_line,
)


def _touch_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo_root = tmp_path / "repo"
    data_dir = repo_root / "NeighborCache" / "data"
    data_dir.mkdir(parents=True)
    wildchat = data_dir / "h1h0_final.npz"
    lmsys = data_dir / "lmsys.npz"
    wildchat.touch()
    lmsys.touch()
    monkeypatch.setattr(
        registry,
        "NPZ_PATHS",
        {
            ("wildchat_final", "default"): wildchat,
            ("lmsys_cluster", "default"): lmsys,
        },
    )
    return repo_root


def _task(output_dir: Path, extra_cli_args: tuple[str, ...] = ()) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=1,
        group="main",
        dataset="wildchat_final",
        embedder="default",
        seed=0,
        alpha=0.05,
        tau_mode="global",
        n_train=200,
        n_calib=200,
        n_eval=200,
        hadamard=True,
        n_trials=1,
        extra_cli_args=extra_cli_args,
        output_dir=output_dir,
        npz_path=Path("input data.npz"),
    )


def test_main_suite_cardinality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = _touch_inputs(tmp_path, monkeypatch)
    tasks = build_main_suite(
        tmp_path / "exp",
        ["wildchat_final", "lmsys_cluster"],
        ["default"],
        [0, 1, 2],
        [0.01, 0.05],
        1,
        200,
        200,
        200,
        repo_root,
    )
    assert len(tasks) == 12
    assert [task.task_id for task in tasks] == list(range(1, 13))
    assert len({task.output_dir for task in tasks}) == 12


def test_missing_npz_raises_with_full_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry,
        "NPZ_PATHS",
        {
            ("wildchat_final", "default"): tmp_path / "missing_wildchat.npz",
            ("lmsys_cluster", "default"): tmp_path / "missing_lmsys.npz",
        },
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        build_main_suite(
            tmp_path / "exp",
            ["wildchat_final", "lmsys_cluster"],
            ["default"],
            [0],
            [0.05],
            1,
            200,
            200,
            200,
            tmp_path,
        )
    message = str(exc_info.value)
    assert "missing_wildchat.npz" in message
    assert "missing_lmsys.npz" in message


def test_task_to_command_line_quotes_paths(tmp_path: Path) -> None:
    task = _task(tmp_path / "output with space")
    command = task_to_command_line("/usr/bin/python", task)
    parts = shlex.split(command)
    assert str(task.output_dir) in parts
    assert str(task.output_dir / "run.log") in parts
    assert str(task.output_dir / "_COMPLETE") in parts


def test_is_complete_roundtrip(tmp_path: Path) -> None:
    task = _task(tmp_path)
    assert not is_complete(task)
    (tmp_path / "_COMPLETE").touch()
    assert is_complete(task)


def test_extra_cli_args_passthrough(tmp_path: Path) -> None:
    task = _task(tmp_path, ("--enable_online_stopping", "--alpha", "0.02"))
    args = task_to_cli_args(task)
    assert args[-3:] == ["--enable_online_stopping", "--alpha", "0.02"]
