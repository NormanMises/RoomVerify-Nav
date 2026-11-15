# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import subprocess
import tempfile
import platform


_RUN_META_CACHE: Optional[Dict[str, Any]] = None


def _find_git_root(start: Path) -> Optional[Path]:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _run_git(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    # Preserve empty stdout for successful commands such as
    # `git status --porcelain` on a clean worktree; only failures return None.
    return (proc.stdout or "").strip()


def _collect_run_meta() -> Dict[str, Any]:
    global _RUN_META_CACHE
    if _RUN_META_CACHE is not None:
        return _RUN_META_CACHE

    meta: Dict[str, Any] = {
        "created_at_unix": time.time(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python_executable": None,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    try:
        import sys

        meta["python_executable"] = sys.executable
        meta["python_version"] = sys.version
    except Exception:
        meta["python_executable"] = None

    git_root = _find_git_root(Path(__file__).resolve())
    git: Dict[str, Any] = {
        "root": str(git_root) if git_root else None,
        "commit": None,
        "branch": None,
        "describe": None,
        "commit_time": None,
        "is_dirty": None,
    }
    if git_root:
        git["commit"] = _run_git(["rev-parse", "HEAD"], cwd=git_root)
        git["branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=git_root)
        git["describe"] = _run_git(["describe", "--tags", "--always", "--dirty"], cwd=git_root)
        git["commit_time"] = _run_git(["show", "-s", "--format=%cI", "HEAD"], cwd=git_root)
        status = _run_git(["status", "--porcelain"], cwd=git_root)
        if status is not None:
            git["is_dirty"] = bool(status.strip())
    meta["git"] = git

    _RUN_META_CACHE = meta
    return meta


def _atomic_write_json(path: str, payload: Dict[str, Any], indent: int) -> None:
    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        return

    try:
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=indent)
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        return


def _ensure_run_meta_written(log_dir: str) -> None:
    # Only write once; keep it best-effort and non-fatal.
    meta_path = os.path.join(log_dir, "run_meta.json")
    try:
        if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
            return
    except Exception:
        return

    meta = _collect_run_meta()
    _atomic_write_json(meta_path, meta, indent=2)


def log_episode(episode_id: Union[str, int], scene_id: str, data: Dict) -> None:
    log_dir = os.environ["ZSOS_LOG_DIR"]
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    _ensure_run_meta_written(log_dir)

    run_meta = _collect_run_meta()
    git = run_meta.get("git", {}) if isinstance(run_meta, dict) else {}
    base = f"{episode_id}_{scene_id}.json"
    filename = os.path.join(log_dir, base)

    # Skip if the filename already exists AND it isn't empty
    if not (os.path.exists(filename) and os.path.getsize(filename) > 0):
        print(f"Logging episode {int(episode_id):04d} to {filename}")
        with open(filename, "w") as f:
            json.dump(
                {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "git_commit": git.get("commit"),
                    "git_describe": git.get("describe"),
                    "git_is_dirty": git.get("is_dirty"),
                    **data,
                },
                f,
                indent=4,
            )


def is_evaluated(episode_id: Union[str, int], scene_id: str) -> bool:
    log_dir = os.environ["ZSOS_LOG_DIR"]
    base = f"{episode_id}_{scene_id}.json"
    filename = os.path.join(log_dir, base)

    # Return false if the directory doesn't exist
    if not os.path.exists(log_dir):
        return False

    # Delete any empty files that are older than 5 minutes
    for f in os.listdir(log_dir):
        try:
            if os.path.getsize(os.path.join(log_dir, f)) == 0 and (
                time.time() - os.path.getmtime(os.path.join(log_dir, f)) > 300
            ):
                os.remove(os.path.join(log_dir, f))
        except Exception:
            pass

    return os.path.exists(filename)
