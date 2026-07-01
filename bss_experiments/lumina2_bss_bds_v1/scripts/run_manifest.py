from __future__ import annotations

import argparse
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, List

from experiment_utils import ensure_dir, read_csv_rows, write_csv_rows
from lumina2_adapter import adapter_from_manifest_row
from make_manifest_lumina2_bds import MANIFEST_FIELDS


def sync_completed_artifacts(row: Dict[str, str], experiment_root: Path, drive_root: Path) -> None:
    for key in ["output_path", "schedule_json_path", "stdout_log_path", "stderr_log_path", "runtime_json_path"]:
        src = Path(row[key])
        if not src.exists():
            continue
        try:
            rel = src.relative_to(experiment_root)
        except ValueError:
            rel = Path(src.name)
        dst = drive_root / rel
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lumina_root", default="/content/Lumina-Image-2.0")
    parser.add_argument("--weights_root", default="/content/drive/MyDrive/ModelWeights/Lumina-Image-2.0")
    parser.add_argument("--experiment_root", default="")
    parser.add_argument("--drive_experiment_root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sync_drive", action="store_true")
    parser.add_argument("--dry_run_schedules_only", action="store_true")
    parser.add_argument("--backend_override", default="", choices=["", "native", "diffusers", "dry_run"])
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--allow_failures", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_csv_rows(manifest_path)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {manifest_path}")
    experiment_root = Path(args.experiment_root) if args.experiment_root else manifest_path.parents[1]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    updated: List[Dict[str, str]] = []

    for row in rows:
        if args.backend_override:
            row["backend"] = args.backend_override
        if args.dry_run_schedules_only:
            row["backend"] = "dry_run"
        if args.force:
            row["force"] = "true"
        output_path = Path(row["output_path"])
        if args.resume and row.get("status") == "completed" and output_path.exists():
            updated.append(row)
            continue
        adapter = adapter_from_manifest_row(
            row,
            lumina_root=args.lumina_root,
            weights_root=args.weights_root,
            experiment_root=str(experiment_root),
            device=args.device,
            dtype=args.dtype,
            hf_token=hf_token,
            cpu_offload=args.cpu_offload,
        )
        try:
            result = adapter.run_one_case(row)
            row["status"] = "completed" if result["status"] in {"completed", "skipped_existing", "dry_run"} else result["status"]
            row["error_message"] = ""
            if args.sync_drive and args.drive_experiment_root and row["status"] == "completed":
                sync_completed_artifacts(row, experiment_root, Path(args.drive_experiment_root))
        except Exception as exc:
            row["status"] = "failed"
            row["error_message"] = repr(exc)
            stderr_path = Path(row["stderr_log_path"])
            ensure_dir(stderr_path.parent)
            if not stderr_path.exists():
                stderr_path.write_text(traceback.format_exc(), encoding="utf-8")
            updated.append(row)
            write_csv_rows(manifest_path, updated + rows[len(updated) :], MANIFEST_FIELDS)
            if not args.allow_failures:
                raise
            continue
        updated.append(row)
        write_csv_rows(manifest_path, updated + rows[len(updated) :], MANIFEST_FIELDS)

    write_csv_rows(manifest_path, updated, MANIFEST_FIELDS)
    completed = sum(1 for row in updated if row.get("status") == "completed")
    failed = sum(1 for row in updated if row.get("status") == "failed")
    print(f"manifest done: completed={completed} failed={failed} total={len(updated)}")


if __name__ == "__main__":
    main()


