from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from experiment_utils import (
    bootstrap_mean_ci,
    ensure_experiment_dirs,
    markdown_table,
    read_csv_rows,
    safe_float,
    safe_int,
    write_csv_rows,
    write_text,
)


TARGET_NFE_ALL = [10, 20, 30, 40]


def split_cases(cases: Sequence[str], split_name: str) -> Tuple[List[str], List[str]]:
    cases = list(cases)
    if split_name == "first_half_split":
        mid = len(cases) // 2
        return cases[:mid], cases[mid:]
    if split_name == "alternating_split":
        return cases[::2], cases[1::2]
    if split_name == "random_split_seed0":
        rng = random.Random(0)
        shuffled = list(cases)
        rng.shuffle(shuffled)
        mid = len(shuffled) // 2
        return shuffled[:mid], shuffled[mid:]
    raise ValueError(split_name)


def filter_values(rows: Sequence[Dict[str, str]], cases: Sequence[str], nfe_set: Sequence[int]) -> List[float]:
    wanted_cases = set(cases)
    wanted_nfe = {str(nfe) for nfe in nfe_set}
    values = []
    for row in rows:
        if row.get("case_id") in wanted_cases and row.get("actual_nfe") in wanted_nfe:
            values.append(safe_float(row.get("rgb_l1_closure_gain")))
    return values


def win_rate(rows: Sequence[Dict[str, str]], cases: Sequence[str], nfe_set: Sequence[int]) -> float:
    wanted_cases = set(cases)
    wanted_nfe = {str(nfe) for nfe in nfe_set}
    vals = [row for row in rows if row.get("case_id") in wanted_cases and row.get("actual_nfe") in wanted_nfe]
    if not vals:
        return float("nan")
    wins = sum(1 for row in vals if str(row.get("bss_win")).lower() in {"true", "1"})
    return wins / len(vals)


def verdict_for(low_ci: Dict[str, float], all_ci: Dict[str, float]) -> str:
    low_lcb = low_ci["lcb95"]
    all_lcb = all_ci["lcb95"]
    if low_lcb != low_lcb:
        return "Need more data"
    if all_lcb == all_lcb and all_lcb > 0:
        return "Green"
    if low_lcb > 0:
        return "Low-only"
    return "Reject"


def format_cell(rows: Sequence[Dict[str, str]], nfe: int) -> str:
    selected = [row for row in rows if safe_int(row.get("actual_nfe")) == nfe]
    if not selected:
        return "--"
    bss_vals = [safe_float(row.get("bss_rgb_l1_closure")) for row in selected]
    gain_vals = [safe_float(row.get("rgb_l1_closure_gain")) for row in selected]
    bss_mean = sum(bss_vals) / len(bss_vals)
    gain_mean = sum(gain_vals) / len(gain_vals)
    return f"{bss_mean:.3f} / {gain_mean:+.3f} [{nfe} NFE]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--gain_csv", default="")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    args = parser.parse_args()

    run_root = ensure_experiment_dirs(args.run_root)
    gain_csv = Path(args.gain_csv) if args.gain_csv else run_root / "metrics/same_compute_gain_long.csv"
    rows = read_csv_rows(gain_csv)
    if not rows:
        raise RuntimeError(f"No gain rows found: {gain_csv}")
    cases = sorted({row["case_id"] for row in rows})
    available_nfe = sorted({safe_int(row["actual_nfe"]) for row in rows})
    all_set = [nfe for nfe in TARGET_NFE_ALL if nfe in available_nfe]

    split_rows: List[Dict[str, Any]] = []
    for split_name in ["first_half_split", "alternating_split", "random_split_seed0"]:
        cal_cases, hold_cases = split_cases(cases, split_name)
        for tset_name, tset in [("BDS_low", [10]), ("BDS_all", all_set)]:
            cal_values = filter_values(rows, cal_cases, tset)
            hold_values = filter_values(rows, hold_cases, tset)
            cal_ci = bootstrap_mean_ci(cal_values, samples=args.bootstrap_samples)
            hold_ci = bootstrap_mean_ci(hold_values, samples=args.bootstrap_samples)
            low_ci = bootstrap_mean_ci(filter_values(rows, cal_cases, [10]), samples=args.bootstrap_samples)
            all_ci = bootstrap_mean_ci(filter_values(rows, cal_cases, all_set), samples=args.bootstrap_samples)
            split_rows.append(
                {
                    "split": split_name,
                    "tset": tset_name,
                    "nfe_points": ",".join(str(x) for x in tset),
                    "calibration_cases": len(cal_cases),
                    "holdout_cases": len(hold_cases),
                    "calibration_mean_gain": f"{cal_ci['mean']:.6f}",
                    "calibration_lcb95": f"{cal_ci['lcb95']:.6f}",
                    "calibration_ucb95": f"{cal_ci['ucb95']:.6f}",
                    "calibration_win_rate": f"{win_rate(rows, cal_cases, tset):.6f}",
                    "holdout_mean_gain": f"{hold_ci['mean']:.6f}",
                    "holdout_win_rate": f"{win_rate(rows, hold_cases, tset):.6f}",
                    "predicted_verdict": verdict_for(low_ci, all_ci),
                    "low_confirmed": str(low_ci["lcb95"] > 0),
                    "all_green_confirmed": str(all_ci["lcb95"] > 0),
                    "low_only_confirmed": str(low_ci["lcb95"] > 0 and not all_ci["lcb95"] > 0),
                }
            )

    table_path = run_root / "tables/tableA_lumina2_bds_by_split.csv"
    write_csv_rows(table_path, split_rows)
    columns = [
        "split",
        "tset",
        "nfe_points",
        "calibration_mean_gain",
        "calibration_lcb95",
        "holdout_mean_gain",
        "predicted_verdict",
    ]
    write_text(run_root / "tables/tableA_lumina2_bds_by_split.md", markdown_table(split_rows, columns))

    low_ci_all_cases = bootstrap_mean_ci(filter_values(rows, cases, [10]), samples=args.bootstrap_samples)
    all_ci_all_cases = bootstrap_mean_ci(filter_values(rows, cases, all_set), samples=args.bootstrap_samples)
    final_verdict = verdict_for(low_ci_all_cases, all_ci_all_cases)
    gain_values = filter_values(rows, cases, all_set)
    mean_delta = sum(gain_values) / len(gain_values) if gain_values else float("nan")
    win = win_rate(rows, cases, all_set)
    cross_row = {
        "Model": "Lumina-Image 2.0",
        "Setting": "Image Generation",
        "Few-step prior": "None / inference-acceleration",
        "Ref.": "50",
        "Cases": str(len(cases)),
        "BDS": final_verdict,
        "Low": format_cell(rows, 10),
        "Middle Low": format_cell(rows, 20),
        "Middle High": format_cell(rows, 30),
        "High": format_cell(rows, 40),
        "Mean Delta": f"{mean_delta:+.3f}",
        "Win": f"{win:.3f}",
    }
    write_csv_rows(run_root / "tables/cross_model_bds_row.csv", [cross_row])
    write_text(
        run_root / "tables/cross_model_bds_row.md",
        markdown_table([cross_row], list(cross_row.keys())),
    )
    write_csv_rows(run_root / "tables/table_cross_model_same_compute_lumina2_row.csv", [cross_row])
    write_text(
        run_root / "tables/table_cross_model_same_compute_lumina2_row.md",
        markdown_table([cross_row], list(cross_row.keys())),
    )
    latex_cells = [
        cross_row["Model"],
        cross_row["Setting"],
        cross_row["Few-step prior"],
        cross_row["Ref."],
        cross_row["Cases"],
        cross_row["BDS"],
        cross_row["Low"],
        cross_row["Middle Low"],
        cross_row["Middle High"],
        cross_row["High"],
        cross_row["Mean Delta"],
        cross_row["Win"],
    ]
    write_text(
        run_root / "tables/table_cross_model_same_compute_lumina2_row.tex",
        " & ".join(latex_cells).replace("_", "\\_") + " \\\\\n",
    )
    write_text(
        run_root / "reports/03_bds_report.md",
        "\n".join(
            [
                "# Lumina-Image 2.0 BDS Report",
                "",
                f"- Gain source: `{gain_csv}`",
                f"- Cases: `{len(cases)}`",
                f"- Available NFE points: `{','.join(str(x) for x in available_nfe)}`",
                f"- Final verdict: `{final_verdict}`",
                f"- Low LCB95: `{low_ci_all_cases['lcb95']:.6f}`",
                f"- All LCB95: `{all_ci_all_cases['lcb95']:.6f}`",
                "",
                "This report uses exact same-NFE pairs only and bootstraps over prompt cases.",
                "",
            ]
        ),
    )
    print(f"wrote {table_path}")
    print(f"verdict {final_verdict}")


if __name__ == "__main__":
    main()


