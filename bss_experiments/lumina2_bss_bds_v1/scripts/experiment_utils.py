from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EXPERIMENT_ID = "lumina2_bss_bds_v1"
MODEL_ID = "Lumina-Image-2.0"
MODEL_REPO_ID = "Alpha-VLLM/Lumina-Image-2.0"

REQUIRED_DIRS = [
    "reports",
    "tables",
    "metrics",
    "figures",
    "figures/side_by_side",
    "schedules",
    "manifests",
    "logs",
    "scripts",
    "patches",
    "splits",
    "prompt_suites",
    "notebooks",
]


def as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def ensure_dir(path: Any) -> Path:
    path = as_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_experiment_dirs(experiment_root: Any) -> Path:
    root = ensure_dir(experiment_root)
    for rel in REQUIRED_DIRS:
        ensure_dir(root / rel)
    return root


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha1_text(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def write_json(path: Any, data: Any) -> Path:
    path = as_path(path)
    ensure_dir(path.parent)
    path.write_text(json_dumps(data) + "\n", encoding="utf-8")
    return path


def read_json(path: Any) -> Any:
    return json.loads(as_path(path).read_text(encoding="utf-8"))


def write_text(path: Any, text: str) -> Path:
    path = as_path(path)
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return path


def read_csv_rows(path: Any) -> List[Dict[str, str]]:
    path = as_path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Any, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> Path:
    path = as_path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def run_capture(cmd: Sequence[str], cwd: Optional[Any] = None, timeout: int = 30) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def package_version(package_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "not_installed"


def repo_state(path: Any) -> Dict[str, str]:
    path = as_path(path)
    if not (path / ".git").exists():
        return {
            "repo_path": str(path),
            "is_git_repo": "false",
            "branch": "not_git",
            "commit": "not_git",
            "remote": "not_git",
            "dirty_status": "not_git",
        }
    branch = run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
    commit = run_capture(["git", "rev-parse", "HEAD"], cwd=path)
    remote = run_capture(["git", "remote", "get-url", "origin"], cwd=path)
    dirty = run_capture(["git", "status", "--short"], cwd=path)
    dirty_status = dirty["stdout"] if dirty["stdout"] else "clean"
    return {
        "repo_path": str(path),
        "is_git_repo": "true",
        "branch": branch["stdout"] or "unknown",
        "commit": commit["stdout"] or "unknown",
        "remote": remote["stdout"] or "none",
        "dirty_status": dirty_status,
    }


def env_flag(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return "false"
    return "true" if str(value).strip() else "false"


def python_runtime_audit() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "diffusers": package_version("diffusers"),
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "safetensors": package_version("safetensors"),
        "sentencepiece": package_version("sentencepiece"),
    }


def torch_cuda_audit() -> Dict[str, str]:
    audit = {
        "torch_cuda_available": "unknown",
        "torch_cuda_version": "unknown",
        "gpu_name": "unknown",
        "gpu_vram_gb": "unknown",
    }
    try:
        import torch

        audit["torch_cuda_available"] = str(torch.cuda.is_available()).lower()
        audit["torch_cuda_version"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            audit["gpu_name"] = props.name
            audit["gpu_vram_gb"] = f"{props.total_memory / (1024 ** 3):.1f}"
    except Exception as exc:
        audit["torch_cuda_available"] = f"error: {exc!r}"
    smi = run_capture(["nvidia-smi"], timeout=15)
    audit["nvidia_smi_returncode"] = str(smi["returncode"])
    audit["nvidia_smi"] = smi["stdout"] or smi["stderr"]
    return audit


def linspace(start: float, end: float, n: int) -> List[float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return [float(start)]
    return [float(start + (end - start) * i / (n - 1)) for i in range(n)]


def native_time_shift(t: float, factor: float) -> float:
    if factor is None or factor == 1:
        return float(t)
    denominator = t + factor - factor * t
    if denominator == 0:
        return float(t)
    return float(t / denominator)


def build_uniform_coords(
    actual_nfe: int,
    coordinate_type: str = "native_time",
    time_shifting_factor: float = 6.0,
    direction: str = "ascending",
) -> List[float]:
    if actual_nfe <= 0:
        raise ValueError("actual_nfe must be positive")
    if direction not in {"ascending", "descending"}:
        raise ValueError("direction must be ascending or descending")
    raw = linspace(0.0, 1.0, actual_nfe)
    if coordinate_type in {"native_time", "shifted_native_time", "sigma"}:
        coords = [native_time_shift(t, time_shifting_factor) for t in raw]
    else:
        coords = raw
    if direction == "descending":
        coords = list(reversed(coords))
    return [round(float(x), 10) for x in coords]


def midpoint(a: float, b: float) -> float:
    return round((float(a) + float(b)) / 2.0, 10)


def build_bss_coords(
    actual_nfe: int,
    coordinate_type: str = "native_time",
    time_shifting_factor: float = 6.0,
    split_pairs: Tuple[int, int] = (0, -1),
    direction: str = "ascending",
    coordinate_mode: str = "native",
) -> Dict[str, Any]:
    if actual_nfe < 4:
        raise ValueError("BSS needs actual_nfe >= 4")
    base_nfe = actual_nfe - 2
    if coordinate_mode == "index":
        raw_base = linspace(0.0, 1.0, base_nfe)
        first_mid_raw = midpoint(raw_base[0], raw_base[1])
        last_mid_raw = midpoint(raw_base[-2], raw_base[-1])
        raw_final = raw_base[:1] + [first_mid_raw] + raw_base[1:-1] + [last_mid_raw] + raw_base[-1:]
        coords = [native_time_shift(t, time_shifting_factor) for t in raw_final]
        base_coords = [native_time_shift(t, time_shifting_factor) for t in raw_base]
        inserted = [
            {"interval_index": 0, "midpoint": native_time_shift(first_mid_raw, time_shifting_factor)},
            {"interval_index": base_nfe - 2, "midpoint": native_time_shift(last_mid_raw, time_shifting_factor)},
        ]
    else:
        base_coords = build_uniform_coords(base_nfe, coordinate_type, time_shifting_factor, "ascending")
        first_mid = midpoint(base_coords[0], base_coords[1])
        last_mid = midpoint(base_coords[-2], base_coords[-1])
        coords = base_coords[:1] + [first_mid] + base_coords[1:-1] + [last_mid] + base_coords[-1:]
        inserted = [
            {
                "interval_index": 0,
                "left": base_coords[0],
                "right": base_coords[1],
                "midpoint": first_mid,
            },
            {
                "interval_index": base_nfe - 2,
                "left": base_coords[-2],
                "right": base_coords[-1],
                "midpoint": last_mid,
            },
        ]
    if direction == "descending":
        coords = list(reversed(coords))
        base_coords = list(reversed(base_coords))
    return {
        "base_sample_steps": base_nfe,
        "base_coords": [round(float(x), 10) for x in base_coords],
        "final_coords": [round(float(x), 10) for x in coords],
        "inserted_midpoints": inserted,
        "split_pairs": list(split_pairs),
    }


def monotonicity(coords: Sequence[float]) -> str:
    if len(coords) < 2:
        return "trivial"
    increasing = all(coords[i] <= coords[i + 1] for i in range(len(coords) - 1))
    decreasing = all(coords[i] >= coords[i + 1] for i in range(len(coords) - 1))
    if increasing:
        return "nondecreasing"
    if decreasing:
        return "nonincreasing"
    return "not_monotonic"


def coordinate_gaps(coords: Sequence[float]) -> List[float]:
    return [abs(float(coords[i + 1]) - float(coords[i])) for i in range(len(coords) - 1)]


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def bootstrap_mean_ci(values: Sequence[float], seed: int = 0, samples: int = 2000) -> Dict[str, float]:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return {"mean": math.nan, "lcb95": math.nan, "ucb95": math.nan}
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return {"mean": sum(clean) / len(clean), "lcb95": lo, "ucb95": hi}


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"

