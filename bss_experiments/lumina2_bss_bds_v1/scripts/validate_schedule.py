from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from experiment_utils import (
    coordinate_gaps,
    ensure_dir,
    monotonicity,
    read_csv_rows,
    safe_int,
    write_csv_rows,
)
from lumina2_adapter import adapter_from_manifest_row


def validate_payload(row: Dict[str, str], payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    actual_nfe = safe_int(row.get("actual_nfe"), 0)
    final_coords = payload.get("final_coords", [])
    base_coords = payload.get("base_coords", [])
    if len(final_coords) != actual_nfe:
        errors.append(f"final_coords length {len(final_coords)} != actual_nfe {actual_nfe}")
    if row.get("sampler_mode") == "bss":
        expected_base = actual_nfe - 2
        if payload.get("base_sample_steps") != expected_base:
            errors.append(f"base_sample_steps {payload.get('base_sample_steps')} != {expected_base}")
        if len(base_coords) != expected_base:
            errors.append(f"base_coords length {len(base_coords)} != {expected_base}")
        if len(payload.get("inserted_midpoints", [])) != 2:
            errors.append("BSS must insert exactly two midpoints")
        if final_coords == base_coords:
            errors.append("BSS final_coords unexpectedly equal base_coords")
    else:
        if payload.get("base_sample_steps") != actual_nfe:
            errors.append(f"uniform base_sample_steps {payload.get('base_sample_steps')} != {actual_nfe}")
    mono = monotonicity(final_coords)
    if mono == "not_monotonic":
        errors.append("final_coords are not monotonic")
    gaps = coordinate_gaps(final_coords)
    if any(gap < 0 for gap in gaps):
        errors.append("negative coordinate gap found")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lumina_root", default="")
    parser.add_argument("--weights_root", default="")
    parser.add_argument("--experiment_root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--require_schedule_files", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_csv_rows(manifest_path)
    if not rows:
        raise RuntimeError(f"No rows in manifest: {manifest_path}")
    experiment_root = args.experiment_root or str(manifest_path.parents[1])
    failures: List[Dict[str, str]] = []

    for row in rows:
        adapter = adapter_from_manifest_row(
            row,
            lumina_root=args.lumina_root,
            weights_root=args.weights_root,
            experiment_root=experiment_root,
            device=args.device,
            dtype=args.dtype,
            hf_token=None,
        )
        schedule_path = Path(row["schedule_json_path"])
        if schedule_path.exists():
            payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        else:
            if args.require_schedule_files:
                failures.append({"run_id": row["run_id"], "error": f"missing schedule file: {schedule_path}"})
                continue
            payload = adapter.schedule_for_manifest_row(row)
            if not args.no_write:
                adapter.dump_schedule_json(row, payload)
        errors = validate_payload(row, payload)
        if errors:
            failures.append({"run_id": row["run_id"], "error": "; ".join(errors)})

    out_path = manifest_path.with_name(manifest_path.stem + "_schedule_validation.csv")
    write_csv_rows(out_path, failures, ["run_id", "error"])
    if failures:
        print(f"schedule validation failed: {len(failures)} rows")
        print(f"details: {out_path}")
        raise SystemExit(1)
    print(f"schedule validation passed for {len(rows)} rows")
    print(f"details: {out_path}")


if __name__ == "__main__":
    main()

