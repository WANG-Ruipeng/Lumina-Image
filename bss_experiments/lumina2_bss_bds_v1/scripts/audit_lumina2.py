from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict

from experiment_utils import (
    MODEL_REPO_ID,
    ensure_experiment_dirs,
    env_flag,
    json_dumps,
    package_version,
    python_runtime_audit,
    repo_state,
    run_capture,
    torch_cuda_audit,
    write_json,
    write_text,
)


NATIVE_FILES = [
    "sample.py",
    "demo.py",
    "transport",
    "transport/integrators.py",
    "transport/transport.py",
    "transport/dpm_solver.py",
    "scripts/sample.sh",
]


def parse_sample_sh(path: Path) -> Dict[str, str]:
    defaults: Dict[str, str] = {}
    if not path.exists():
        return defaults
    text = path.read_text(encoding="utf-8", errors="ignore")
    for key in ["steps", "solver", "time_shifting_factor", "cfg_scale", "res", "seed"]:
        match = re.search(rf"\b{re.escape(key)}=([^\s]+)", text)
        if match:
            defaults[key] = match.group(1).strip("\"'")
    return defaults


def file_contains(path: Path, needle: str) -> str:
    if not path.exists() or path.is_dir():
        return "missing"
    try:
        return "true" if needle in path.read_text(encoding="utf-8", errors="ignore") else "false"
    except Exception as exc:
        return f"error: {exc!r}"


def diffusers_audit() -> Dict[str, str]:
    result = {"import_lumina2_pipeline": "unknown", "pipeline_call_support": "unknown", "scheduler_class": "unknown"}
    try:
        from diffusers import Lumina2Pipeline

        result["import_lumina2_pipeline"] = "true"
        import inspect

        sig = inspect.signature(Lumina2Pipeline.__call__)
        result["pipeline_call_support"] = ", ".join(sig.parameters.keys())
        try:
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

            result["scheduler_class"] = FlowMatchEulerDiscreteScheduler.__name__
        except Exception:
            result["scheduler_class"] = "FlowMatchEulerDiscreteScheduler import unavailable"
    except Exception as exc:
        result["import_lumina2_pipeline"] = f"false: {exc!r}"
    return result


def build_audit(args: argparse.Namespace) -> Dict[str, Any]:
    experiment_root = ensure_experiment_dirs(args.experiment_root)
    lumina_root = Path(args.lumina_root)
    weights_root = Path(args.weights_root)
    current_repo = Path.cwd()
    sample_py = lumina_root / "sample.py"
    transport_py = lumina_root / "transport/transport.py"
    integrators_py = lumina_root / "transport/integrators.py"
    sample_sh = lumina_root / "scripts/sample.sh"

    native_file_status = {rel: str((lumina_root / rel).exists()) for rel in NATIVE_FILES}
    sample_text = sample_py.read_text(encoding="utf-8", errors="ignore") if sample_py.exists() else ""
    audit: Dict[str, Any] = {
        "experiment_root": str(experiment_root),
        "current_repo": repo_state(current_repo),
        "lumina_repo": repo_state(lumina_root) if lumina_root.exists() else {"repo_path": str(lumina_root), "exists": "false"},
        "runtime": python_runtime_audit(),
        "cuda": torch_cuda_audit(),
        "environment": {
            "is_colab": "true" if "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists() else "false",
            "google_drive_mounted": str(Path("/content/drive/MyDrive").exists()).lower(),
            "hf_token_env": "true" if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") else "false",
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "paths": {
            "lumina_root": str(lumina_root),
            "weights_root": str(weights_root),
            "weights_root_exists": str(weights_root.exists()).lower(),
            "model_repo_id": MODEL_REPO_ID,
        },
        "native_files": native_file_status,
        "native_solver_support": {
            "euler_in_sample": str("euler" in sample_text).lower(),
            "midpoint_in_sample": str("midpoint" in sample_text).lower(),
            "dpm_in_sample": str("dpm" in sample_text).lower(),
            "custom_time_grid_patch_applied": file_contains(sample_py, "custom_time_grid_json"),
            "integrator_time_shift_formula": file_contains(integrators_py, "time_shifting_factor"),
            "dpm_time_uniform_flow": file_contains(lumina_root / "transport/dpm_solver.py", "time_uniform_flow"),
        },
        "sample_sh_defaults": parse_sample_sh(sample_sh),
        "diffusers": {
            **diffusers_audit(),
            "diffusers_version": package_version("diffusers"),
            "transformers_version": package_version("transformers"),
            "accelerate_version": package_version("accelerate"),
        },
        "weight_format_hints": {
            "native_model_args": str((weights_root / "model_args.pth").exists()).lower(),
            "native_consolidated_00": str((weights_root / "consolidated.00-of-01.pth").exists()).lower(),
            "diffusers_model_index": str((weights_root / "model_index.json").exists()).lower(),
            "text_encoder_gemma_access_needed": "true",
            "vae": "FLUX-VAE-16CH/native flux vae route or diffusers model folder",
        },
    }
    if args.include_commands:
        audit["commands"] = {
            "which_python": run_capture(["which", "python"]),
            "nvidia_smi": run_capture(["nvidia-smi"], timeout=15),
        }
    return audit


def write_markdown_report(path: Path, audit: Dict[str, Any]) -> None:
    lines = [
        "# Lumina-Image 2.0 Repo, Model, and Hardware Audit",
        "",
        "## Summary",
        "",
        f"- Experiment root: `{audit['experiment_root']}`",
        f"- Lumina root: `{audit['paths']['lumina_root']}`",
        f"- Weights root: `{audit['paths']['weights_root']}`",
        f"- Weights root exists: `{audit['paths']['weights_root_exists']}`",
        f"- Current repo commit: `{audit['current_repo'].get('commit', '')}`",
        f"- Current repo dirty status: `{audit['current_repo'].get('dirty_status', '')}`",
        f"- Lumina repo commit: `{audit['lumina_repo'].get('commit', '')}`",
        f"- GPU: `{audit['cuda'].get('gpu_name', 'unknown')}`",
        f"- VRAM GB: `{audit['cuda'].get('gpu_vram_gb', 'unknown')}`",
        f"- Torch: `{audit['runtime'].get('torch', '')}`",
        f"- CUDA: `{audit['cuda'].get('torch_cuda_version', '')}`",
        f"- Diffusers: `{audit['diffusers'].get('diffusers_version', '')}`",
        "",
        "## Native Files",
        "",
    ]
    for key, value in audit["native_files"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Native Solver Support",
            "",
        ]
    )
    for key, value in audit["native_solver_support"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## sample.sh Defaults",
            "",
        ]
    )
    for key, value in audit["sample_sh_defaults"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Diffusers",
            "",
        ]
    )
    for key, value in audit["diffusers"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Weight Status",
            "",
        ]
    )
    for key, value in audit["weight_format_hints"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## If Weights Are Missing",
            "",
            "Download to Google Drive, not to the GitHub repo:",
            "",
            "```bash",
            "huggingface-cli download Alpha-VLLM/Lumina-Image-2.0 \\",
            "  --local-dir /content/drive/MyDrive/ModelWeights/Lumina-Image-2.0 \\",
            "  --local-dir-use-symlinks False",
            "```",
            "",
            "If Gemma-2-2B requires authentication, provide `HF_TOKEN` through the environment or Colab secrets.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumina_root", required=True)
    parser.add_argument("--weights_root", required=True)
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--include_commands", action="store_true")
    args = parser.parse_args()

    audit = build_audit(args)
    root = ensure_experiment_dirs(args.experiment_root)
    write_json(root / "reports/00_repo_model_hardware_audit.json", audit)
    write_markdown_report(root / "reports/00_repo_model_hardware_audit.md", audit)
    print(json_dumps({"audit_report": str(root / "reports/00_repo_model_hardware_audit.md")}))


if __name__ == "__main__":
    main()

