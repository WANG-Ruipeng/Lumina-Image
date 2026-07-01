from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from experiment_utils import (
    MODEL_ID,
    ensure_experiment_dirs,
    repo_state,
    sha1_text,
    write_csv_rows,
)


MANIFEST_FIELDS = [
    "run_id",
    "model_id",
    "model_variant",
    "setting",
    "few_step_prior",
    "task",
    "modality",
    "protocol",
    "backend",
    "solver",
    "bss_variant",
    "case_id",
    "category",
    "prompt",
    "prompt_hash",
    "method",
    "method_family",
    "sampler_mode",
    "actual_nfe",
    "num_sampling_steps",
    "base_sample_steps",
    "split_pairs",
    "time_shifting_factor",
    "cfg_scale",
    "cfg_trunc_ratio",
    "cfg_normalization",
    "seed",
    "height",
    "width",
    "max_sequence_length",
    "reference_method",
    "reference_nfe",
    "low_baseline_method",
    "output_path",
    "schedule_json_path",
    "stdout_log_path",
    "stderr_log_path",
    "runtime_json_path",
    "status",
    "error_message",
    "git_commit",
    "dirty_status",
]


def load_prompt_suite(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def method_spec(method: str) -> Dict[str, Any]:
    if method == "reference_uniform50":
        return {"actual_nfe": 50, "method_family": "reference", "sampler_mode": "uniform", "bss_variant": "none"}
    if method.startswith("uniform"):
        return {
            "actual_nfe": int(method.replace("uniform", "")),
            "method_family": "uniform",
            "sampler_mode": "uniform",
            "bss_variant": "none",
        }
    if method.startswith("bss") and method.endswith("_native"):
        nfe = int(method.replace("bss", "").replace("_native", ""))
        return {"actual_nfe": nfe, "method_family": "bss", "sampler_mode": "bss", "bss_variant": "native"}
    if method.startswith("bss") and method.endswith("_index"):
        nfe = int(method.replace("bss", "").replace("_index", ""))
        return {"actual_nfe": nfe, "method_family": "bss", "sampler_mode": "bss", "bss_variant": "index"}
    raise ValueError(f"Unsupported method: {method}")


def build_row(
    experiment_root: Path,
    repo: Dict[str, str],
    prompt_case: Dict[str, str],
    method: str,
    backend: str,
    solver: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    spec = method_spec(method)
    prompt_hash = sha1_text(prompt_case["prompt"])
    case_id = prompt_case["case_id"]
    run_id = f"{case_id}_{backend}_{solver}_{method}_seed{args.seed}_{args.height}x{args.width}"
    actual_nfe = int(spec["actual_nfe"])
    base_sample_steps = actual_nfe - 2 if spec["sampler_mode"] == "bss" else actual_nfe
    num_sampling_steps = base_sample_steps if spec["sampler_mode"] == "bss" else actual_nfe
    rel_stem = f"{case_id}/{backend}_{solver}/{method}_seed{args.seed}_{args.height}x{args.width}"
    return {
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_variant": "2.6B",
        "setting": "Image Generation",
        "few_step_prior": "None / inference-acceleration, not explicit few-step distillation",
        "task": "t2i",
        "modality": "image",
        "protocol": "fixed_prompt_suite_lumina2",
        "backend": backend,
        "solver": solver,
        "bss_variant": spec["bss_variant"],
        "case_id": case_id,
        "category": prompt_case.get("category", ""),
        "prompt": prompt_case["prompt"],
        "prompt_hash": prompt_hash,
        "method": method,
        "method_family": spec["method_family"],
        "sampler_mode": spec["sampler_mode"],
        "actual_nfe": actual_nfe,
        "num_sampling_steps": num_sampling_steps,
        "base_sample_steps": base_sample_steps,
        "split_pairs": "[0,-1]" if spec["sampler_mode"] == "bss" else "",
        "time_shifting_factor": args.time_shifting_factor,
        "cfg_scale": args.cfg_scale,
        "cfg_trunc_ratio": args.cfg_trunc_ratio,
        "cfg_normalization": args.cfg_normalization,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "max_sequence_length": args.max_sequence_length,
        "reference_method": "reference_uniform50",
        "reference_nfe": 50,
        "low_baseline_method": "uniform8",
        "output_path": str(experiment_root / "figures/generated" / f"{rel_stem}.png"),
        "schedule_json_path": str(experiment_root / "schedules" / f"{rel_stem}.json"),
        "stdout_log_path": str(experiment_root / "logs" / f"{rel_stem}.stdout.log"),
        "stderr_log_path": str(experiment_root / "logs" / f"{rel_stem}.stderr.log"),
        "runtime_json_path": str(experiment_root / "logs" / f"{rel_stem}.runtime.json"),
        "status": "pending",
        "error_message": "",
        "git_commit": repo.get("commit", "not_git"),
        "dirty_status": repo.get("dirty_status", "not_git"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--prompt_suite", default="")
    parser.add_argument("--backend", default="native", choices=["native", "diffusers", "dry_run"])
    parser.add_argument("--solver", default="euler", choices=["euler", "midpoint", "dpm", "flowmatch_euler"])
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--time_shifting_factor", type=float, default=6.0)
    parser.add_argument("--cfg_trunc_ratio", default="")
    parser.add_argument("--cfg_normalization", default="")
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mini_prompts", type=int, default=16)
    args = parser.parse_args()

    experiment_root = ensure_experiment_dirs(args.experiment_root)
    default_suite = Path(__file__).resolve().parents[1] / "prompt_suites/lumina2_prompt_suite_v1.json"
    suite_path = Path(args.prompt_suite) if args.prompt_suite else default_suite
    suite = load_prompt_suite(suite_path)
    repo = repo_state(Path.cwd())

    smoke_methods = ["uniform8", "uniform10", "bss10_native", "reference_uniform50"]
    mini_methods = [
        "uniform8",
        "uniform10",
        "uniform20",
        "uniform30",
        "uniform40",
        "reference_uniform50",
        "bss10_native",
        "bss20_native",
        "bss30_native",
        "bss40_native",
    ]

    smoke_case = suite["smoke_prompt"]
    smoke_rows = [
        build_row(experiment_root, repo, smoke_case, method, args.backend, args.solver, args)
        for method in smoke_methods
    ]
    prompt_cases = suite["prompts"][: args.mini_prompts]
    mini_rows = [
        build_row(experiment_root, repo, prompt_case, method, args.backend, args.solver, args)
        for prompt_case in prompt_cases
        for method in mini_methods
    ]

    smoke_path = experiment_root / "manifests/lumina2_smoke_manifest.csv"
    mini_path = experiment_root / "manifests/lumina2_prompt_suite_manifest.csv"
    write_csv_rows(smoke_path, smoke_rows, MANIFEST_FIELDS)
    write_csv_rows(mini_path, mini_rows, MANIFEST_FIELDS)
    print(f"wrote {smoke_path}")
    print(f"wrote {mini_path}")


if __name__ == "__main__":
    main()

