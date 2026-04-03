from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        joined_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, joined_key))
        else:
            flattened[joined_key] = value
    return flattened


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_gpu_info() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {
            "gpu_count": len(lines),
            "gpu_devices": json.dumps(lines),
        }
    except Exception:
        return {
            "gpu_count": 0,
            "gpu_devices": json.dumps([]),
        }


def collect_environment_info() -> dict[str, Any]:
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count() or 0,
    }
    info.update(get_gpu_info())
    return info
