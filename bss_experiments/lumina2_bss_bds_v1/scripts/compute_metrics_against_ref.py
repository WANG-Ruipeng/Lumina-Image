from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from experiment_utils import ensure_experiment_dirs, read_csv_rows, safe_int, write_csv_rows


def image_array(path: Path):
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype("float32") / 255.0


def pair_metrics(method_path: Path, ref_path: Path) -> Dict[str, float]:
    import numpy as np

    a = image_array(method_path)
    b = image_array(ref_path)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {method_path} {a.shape} vs {ref_path} {b.shape}")
    diff = a - b
    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff * diff)))
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0 else float(20.0 * math.log10(1.0 / math.sqrt(mse)))
    return {"rgb_l1_to_ref": l1, "rgb_l2_to_ref": l2, "psnr_to_ref": psnr}


def key_for(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str]:
    return (
        row.get("case_id", ""),
        row.get("backend", ""),
        row.get("solver", ""),
        row.get("seed", ""),
        row.get("height", ""),
        row.get("width", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    rows = read_csv_rows(args.manifest)
    run_root = ensure_experiment_dirs(args.run_root)
    ref_by_key: Dict[Tuple[str, str, str, str, str, str], Dict[str, str]] = {}
    low_by_key: Dict[Tuple[str, str, str, str, str, str], Dict[str, str]] = {}
    for row in rows:
        if row.get("method") == row.get("reference_method", "reference_uniform50"):
            ref_by_key[key_for(row)] = row
        if row.get("method") == row.get("low_baseline_method", "uniform8"):
            low_by_key[key_for(row)] = row

    metric_rows: List[Dict[str, Any]] = []
    for row in rows:
        method_path = Path(row["output_path"])
        ref_row = ref_by_key.get(key_for(row))
        low_row = low_by_key.get(key_for(row))
        if not ref_row or not low_row:
            if args.allow_missing:
                continue
            raise RuntimeError(f"Missing reference or low baseline for {row['run_id']}")
        ref_path = Path(ref_row["output_path"])
        low_path = Path(low_row["output_path"])
        if not method_path.exists() or not ref_path.exists() or not low_path.exists():
            if args.allow_missing:
                continue
            raise RuntimeError(f"Missing image for {row['run_id']}")
        metrics = pair_metrics(method_path, ref_path)
        low_metrics = pair_metrics(low_path, ref_path)
        low_l1 = low_metrics["rgb_l1_to_ref"]
        closure = 1.0 - metrics["rgb_l1_to_ref"] / low_l1 if low_l1 > 0 else float("nan")
        metric_rows.append(
            {
                **{key: row.get(key, "") for key in [
                    "run_id",
                    "case_id",
                    "category",
                    "backend",
                    "solver",
                    "method",
                    "method_family",
                    "actual_nfe",
                    "seed",
                    "height",
                    "width",
                    "reference_method",
                    "reference_nfe",
                    "low_baseline_method",
                ]},
                **metrics,
                "rgb_l1_closure": closure,
                "compute_fraction": safe_int(row.get("actual_nfe"), 0) / max(safe_int(row.get("reference_nfe"), 50), 1),
                "output_path": row["output_path"],
                "reference_output_path": ref_row["output_path"],
            }
        )

    master_path = run_root / "metrics/master_long_metrics.csv"
    write_csv_rows(master_path, metric_rows)

    by_pair: Dict[Tuple[str, str, str, str, str], Dict[str, Dict[str, Any]]] = {}
    for row in metric_rows:
        if row["method_family"] not in {"uniform", "bss"}:
            continue
        key = (row["case_id"], row["backend"], row["solver"], row["actual_nfe"], row["seed"])
        by_pair.setdefault(key, {})[row["method_family"]] = row
    gain_rows: List[Dict[str, Any]] = []
    for key, pair in by_pair.items():
        if "uniform" not in pair or "bss" not in pair:
            continue
        uniform = pair["uniform"]
        bss = pair["bss"]
        gain_rows.append(
            {
                "case_id": key[0],
                "backend": key[1],
                "solver": key[2],
                "actual_nfe": key[3],
                "seed": key[4],
                "uniform_method": uniform["method"],
                "bss_method": bss["method"],
                "uniform_rgb_l1_closure": uniform["rgb_l1_closure"],
                "bss_rgb_l1_closure": bss["rgb_l1_closure"],
                "rgb_l1_closure_gain": bss["rgb_l1_closure"] - uniform["rgb_l1_closure"],
                "bss_win": bss["rgb_l1_closure"] > uniform["rgb_l1_closure"],
                "reference_method": bss["reference_method"],
                "reference_nfe": bss["reference_nfe"],
            }
        )
    write_csv_rows(run_root / "metrics/same_compute_gain_long.csv", gain_rows)
    write_csv_rows(run_root / "metrics/per_case_metrics.csv", metric_rows)
    print(f"wrote {master_path}")
    print(f"paired gains: {len(gain_rows)}")


if __name__ == "__main__":
    main()

