from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from experiment_utils import (
    MODEL_ID,
    MODEL_REPO_ID,
    build_bss_coords,
    build_uniform_coords,
    ensure_dir,
    now_iso,
    repo_state,
    safe_float,
    safe_int,
    write_json,
    write_text,
)


def tail_text(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


class Lumina2Adapter:
    def __init__(
        self,
        backend: str,
        lumina_root: str,
        weights_root: str,
        experiment_root: str,
        device: str,
        dtype: str,
        solver: str,
        resolution: int,
        cfg_scale: float,
        time_shifting_factor: float,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
    ):
        self.backend = backend
        self.lumina_root = Path(lumina_root) if lumina_root else Path("")
        self.weights_root = Path(weights_root) if weights_root else Path(MODEL_REPO_ID)
        self.experiment_root = Path(experiment_root)
        self.device = device
        self.dtype = dtype
        self.solver = solver
        self.resolution = resolution
        self.cfg_scale = cfg_scale
        self.time_shifting_factor = float(time_shifting_factor)
        self.hf_token = hf_token
        self.cpu_offload = cpu_offload
        self.pipeline = None

    def load_pipeline(self):
        if self.backend == "diffusers":
            return self._load_diffusers_pipeline()
        if self.backend == "native":
            return None
        if self.backend == "dry_run":
            return None
        raise ValueError(f"Unsupported backend: {self.backend}")

    def get_official_base_schedule(self, num_steps: int, solver: str, **kwargs: Any) -> Dict[str, Any]:
        coordinate_type = kwargs.get("coordinate_type") or self._coordinate_type_for_solver(solver)
        direction = kwargs.get("direction") or self._direction_for_backend()
        coords = build_uniform_coords(
            num_steps,
            coordinate_type=coordinate_type,
            time_shifting_factor=float(kwargs.get("time_shifting_factor", self.time_shifting_factor)),
            direction=direction,
        )
        return {
            "solver": solver,
            "num_sampling_steps": num_steps,
            "coordinate_type": coordinate_type,
            "coords": coords,
            "source": "local_schedule_builder",
        }

    def build_uniform_schedule(self, actual_nfe: int, solver: str, **kwargs: Any) -> Dict[str, Any]:
        coordinate_type = kwargs.get("coordinate_type") or self._coordinate_type_for_solver(solver)
        direction = kwargs.get("direction") or self._direction_for_backend()
        coords = build_uniform_coords(
            actual_nfe,
            coordinate_type=coordinate_type,
            time_shifting_factor=float(kwargs.get("time_shifting_factor", self.time_shifting_factor)),
            direction=direction,
        )
        return {
            "sampler_mode": "uniform",
            "bss_variant": "none",
            "actual_nfe": actual_nfe,
            "base_sample_steps": actual_nfe,
            "num_sampling_steps": actual_nfe,
            "coordinate_type": coordinate_type,
            "base_coords": coords,
            "final_coords": coords,
            "inserted_midpoints": [],
            "split_pairs": [],
        }

    def build_bss_schedule(
        self,
        actual_nfe: int,
        solver: str,
        split_pairs: Tuple[int, int] = (0, -1),
        coordinate_mode: str = "native",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        coordinate_type = kwargs.get("coordinate_type") or self._coordinate_type_for_solver(solver)
        direction = kwargs.get("direction") or self._direction_for_backend()
        payload = build_bss_coords(
            actual_nfe,
            coordinate_type=coordinate_type,
            time_shifting_factor=float(kwargs.get("time_shifting_factor", self.time_shifting_factor)),
            split_pairs=split_pairs,
            direction=direction,
            coordinate_mode=coordinate_mode,
        )
        payload.update(
            {
                "sampler_mode": "bss",
                "bss_variant": coordinate_mode,
                "actual_nfe": actual_nfe,
                "num_sampling_steps": payload["base_sample_steps"],
                "coordinate_type": coordinate_type,
            }
        )
        return payload

    def run_one_case(self, manifest_row: Dict[str, str]) -> Dict[str, Any]:
        schedule_payload = self.schedule_for_manifest_row(manifest_row)
        schedule_path = Path(manifest_row["schedule_json_path"])
        self.dump_schedule_json(manifest_row, schedule_payload)

        output_path = Path(manifest_row["output_path"])
        ensure_dir(output_path.parent)
        if output_path.exists() and str(manifest_row.get("force", "")).lower() != "true":
            return {"status": "skipped_existing", "output_path": str(output_path), "schedule_json_path": str(schedule_path)}

        if self.backend == "dry_run":
            return {"status": "dry_run", "output_path": str(output_path), "schedule_json_path": str(schedule_path)}
        if self.backend == "diffusers":
            return self._run_diffusers_case(manifest_row, schedule_payload)
        if self.backend == "native":
            return self._run_native_case(manifest_row, schedule_payload)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def dump_schedule_json(self, manifest_row: Dict[str, str], schedule_payload: Dict[str, Any]) -> Path:
        path = Path(manifest_row["schedule_json_path"])
        payload = dict(schedule_payload)
        payload.update(
            {
                "model_id": MODEL_ID,
                "setting": "Image Generation",
                "backend": manifest_row.get("backend", self.backend),
                "solver": manifest_row.get("solver", self.solver),
                "method": manifest_row.get("method", ""),
                "reference_method": manifest_row.get("reference_method", "reference_uniform50"),
                "reference_nfe": safe_int(manifest_row.get("reference_nfe", 50), 50),
                "time_shifting_factor": safe_float(manifest_row.get("time_shifting_factor", self.time_shifting_factor), self.time_shifting_factor),
                "cfg_scale": safe_float(manifest_row.get("cfg_scale", self.cfg_scale), self.cfg_scale),
                "cfg_trunc_ratio": manifest_row.get("cfg_trunc_ratio", ""),
                "cfg_normalization": manifest_row.get("cfg_normalization", ""),
                "prompt_hash": manifest_row.get("prompt_hash", ""),
                "seed": safe_int(manifest_row.get("seed", 0), 0),
                "height": safe_int(manifest_row.get("height", self.resolution), self.resolution),
                "width": safe_int(manifest_row.get("width", self.resolution), self.resolution),
                "output_path": manifest_row.get("output_path", ""),
                "git_commit": manifest_row.get("git_commit", ""),
                "dirty_status": manifest_row.get("dirty_status", ""),
                "timestamp": now_iso(),
            }
        )
        write_json(path, payload)
        return path

    def validate_outputs(self, manifest_row: Dict[str, str]) -> Dict[str, Any]:
        output_path = Path(manifest_row["output_path"])
        schedule_path = Path(manifest_row["schedule_json_path"])
        return {
            "output_exists": output_path.exists(),
            "output_path": str(output_path),
            "schedule_exists": schedule_path.exists(),
            "schedule_json_path": str(schedule_path),
        }

    def schedule_for_manifest_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        actual_nfe = safe_int(row.get("actual_nfe"), 0)
        solver = row.get("solver") or self.solver
        sampler_mode = row.get("sampler_mode") or ("bss" if row.get("method", "").startswith("bss") else "uniform")
        bss_variant = row.get("bss_variant") or "none"
        if sampler_mode == "bss":
            coordinate_mode = bss_variant if bss_variant != "none" else "native"
            return self.build_bss_schedule(actual_nfe, solver, coordinate_mode=coordinate_mode)
        return self.build_uniform_schedule(actual_nfe, solver)

    def _coordinate_type_for_solver(self, solver: str) -> str:
        if self.backend == "diffusers" or solver == "flowmatch_euler":
            return "sigma"
        if solver in {"euler", "midpoint"}:
            return "native_time"
        if solver == "dpm":
            return "unknown"
        return "native_time"

    def _direction_for_backend(self) -> str:
        if self.backend == "diffusers":
            return "descending"
        return "ascending"

    def _torch_dtype(self):
        import torch

        if self.dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if self.dtype in {"fp16", "float16"}:
            return torch.float16
        return torch.float32

    def _load_diffusers_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        import torch
        from diffusers import Lumina2Pipeline

        model_path = str(self.weights_root)
        if not self.weights_root.exists():
            model_path = MODEL_REPO_ID
        kwargs: Dict[str, Any] = {"torch_dtype": self._torch_dtype()}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        pipe = Lumina2Pipeline.from_pretrained(model_path, **kwargs)
        if self.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device if torch.cuda.is_available() else "cpu")
        self.pipeline = pipe
        return pipe

    def _run_diffusers_case(self, row: Dict[str, str], schedule_payload: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        pipe = self.load_pipeline()
        signature = inspect.signature(pipe.__call__)
        prompt = row["prompt"]
        actual_nfe = safe_int(row.get("actual_nfe"), 0)
        seed = safe_int(row.get("seed"), 0)
        height = safe_int(row.get("height"), self.resolution)
        width = safe_int(row.get("width"), self.resolution)
        output_path = Path(row["output_path"])
        runtime_path = Path(row["runtime_json_path"])
        stdout_path = Path(row["stdout_log_path"])
        stderr_path = Path(row["stderr_log_path"])
        ensure_dir(stdout_path.parent)
        ensure_dir(stderr_path.parent)

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": actual_nfe,
            "guidance_scale": safe_float(row.get("cfg_scale"), self.cfg_scale),
            "generator": torch.Generator(device=self.device if torch.cuda.is_available() else "cpu").manual_seed(seed),
        }
        if "max_sequence_length" in signature.parameters and row.get("max_sequence_length"):
            call_kwargs["max_sequence_length"] = safe_int(row.get("max_sequence_length"), 256)
        if schedule_payload.get("sampler_mode") == "bss":
            if "sigmas" not in signature.parameters:
                raise RuntimeError("Installed Lumina2Pipeline does not expose sigmas; refusing to fake BSS.")
            call_kwargs["sigmas"] = schedule_payload["final_coords"]

        start = time.perf_counter()
        try:
            result = pipe(**call_kwargs)
            image = result.images[0]
            image.save(output_path)
            elapsed = time.perf_counter() - start
            write_text(stdout_path, f"saved {output_path}\n")
            write_text(stderr_path, "")
            write_json(
                runtime_path,
                {
                    "status": "completed",
                    "runtime_sec": elapsed,
                    "backend": "diffusers",
                    "device": self.device,
                    "actual_nfe": actual_nfe,
                    "timestamp": now_iso(),
                },
            )
            return {"status": "completed", "runtime_sec": elapsed, "output_path": str(output_path)}
        except Exception as exc:
            elapsed = time.perf_counter() - start
            write_text(stderr_path, repr(exc) + "\n")
            write_json(
                runtime_path,
                {
                    "status": "failed",
                    "runtime_sec": elapsed,
                    "backend": "diffusers",
                    "error": repr(exc),
                    "timestamp": now_iso(),
                },
            )
            raise

    def _run_native_case(self, row: Dict[str, str], schedule_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.lumina_root.exists():
            raise RuntimeError(f"Lumina root does not exist: {self.lumina_root}")
        sample_py = self.lumina_root / "sample.py"
        if not sample_py.exists():
            raise RuntimeError(f"Native sample.py missing: {sample_py}")
        sample_text = sample_py.read_text(encoding="utf-8", errors="ignore")
        needs_custom_grid = schedule_payload.get("sampler_mode") == "bss"
        if needs_custom_grid and "custom_time_grid_json" not in sample_text:
            raise RuntimeError(
                "Native BSS needs the custom time-grid patch. Run scripts/patch_lumina_native_custom_grid.py first."
            )

        temp_dir = ensure_dir(self.experiment_root / "logs/native_inputs")
        caption_path = temp_dir / f"{row['run_id']}.txt"
        write_text(caption_path, row["prompt"] + "\n")
        output_dir = ensure_dir(self.experiment_root / "native_outputs" / row["run_id"])
        output_path = Path(row["output_path"])
        runtime_path = Path(row["runtime_json_path"])
        stdout_path = Path(row["stdout_log_path"])
        stderr_path = Path(row["stderr_log_path"])

        cmd = [
            sys.executable,
            str(sample_py),
            "--ckpt",
            str(self.weights_root),
            "--image_save_path",
            str(output_dir),
            "--solver",
            row.get("solver", self.solver),
            "--num_sampling_steps",
            str(safe_int(row.get("num_sampling_steps"), safe_int(row.get("actual_nfe"), 0))),
            "--caption_path",
            str(caption_path),
            "--seed",
            str(safe_int(row.get("seed"), 0)),
            "--resolution",
            f"{row.get('height', self.resolution)}:{row.get('width', self.resolution)}x{row.get('height', self.resolution)}",
            "--time_shifting_factor",
            str(safe_float(row.get("time_shifting_factor"), self.time_shifting_factor)),
            "--cfg_scale",
            str(safe_float(row.get("cfg_scale"), self.cfg_scale)),
            "--batch_size",
            "1",
            "--precision",
            "bf16" if self.dtype in {"bf16", "bfloat16"} else "fp32",
        ]
        if self.hf_token:
            cmd.extend(["--hf_token", self.hf_token])
        if needs_custom_grid:
            cmd.extend(["--custom_time_grid_json", row["schedule_json_path"]])

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(self.lumina_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        elapsed = time.perf_counter() - start
        write_text(stdout_path, proc.stdout)
        write_text(stderr_path, proc.stderr)
        source_image = self._find_native_output_image(output_dir)
        status = "completed" if proc.returncode == 0 and source_image else "failed"
        if source_image:
            ensure_dir(output_path.parent)
            shutil.copy2(source_image, output_path)
        write_json(
            runtime_path,
            {
                "status": status,
                "runtime_sec": elapsed,
                "backend": "native",
                "returncode": proc.returncode,
                "source_image": str(source_image) if source_image else "",
                "timestamp": now_iso(),
            },
        )
        if status != "completed":
            details = [
                f"Native Lumina command failed for {row['run_id']} with returncode={proc.returncode}.",
                f"stdout_log_path={stdout_path}",
                f"stderr_log_path={stderr_path}",
            ]
            if proc.returncode == 0 and not source_image:
                details.append(f"No output image was found under {output_dir}.")
            stderr_tail = tail_text(proc.stderr)
            stdout_tail = tail_text(proc.stdout)
            if stderr_tail:
                details.append(f"STDERR tail:\n{stderr_tail}")
            if stdout_tail:
                details.append(f"STDOUT tail:\n{stdout_tail}")
            raise RuntimeError("\n".join(details))
        return {"status": status, "runtime_sec": elapsed, "output_path": str(output_path)}

    def _find_native_output_image(self, output_dir: Path) -> Optional[Path]:
        images_dir = output_dir / "images"
        candidates = list(images_dir.glob("*.png")) if images_dir.exists() else list(output_dir.glob("*.png"))
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]


def adapter_from_manifest_row(
    row: Dict[str, str],
    lumina_root: str,
    weights_root: str,
    experiment_root: str,
    device: str,
    dtype: str,
    hf_token: Optional[str],
    cpu_offload: bool = False,
) -> Lumina2Adapter:
    return Lumina2Adapter(
        backend=row.get("backend", "native"),
        lumina_root=lumina_root,
        weights_root=weights_root,
        experiment_root=experiment_root,
        device=device,
        dtype=dtype,
        solver=row.get("solver", "euler"),
        resolution=safe_int(row.get("height"), 1024),
        cfg_scale=safe_float(row.get("cfg_scale"), 4.0),
        time_shifting_factor=safe_float(row.get("time_shifting_factor"), 6.0),
        hf_token=hf_token,
        cpu_offload=cpu_offload,
    )

