from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from experiment_utils import (
    build_bss_coords,
    build_uniform_coords,
    coordinate_gaps,
    ensure_experiment_dirs,
    markdown_table,
    monotonicity,
    write_csv_rows,
    write_text,
)


def coord_type_for(solver: str, backend: str) -> str:
    if backend == "diffusers" or solver == "flowmatch_euler":
        return "sigma"
    if solver in {"euler", "midpoint"}:
        return "native_time"
    return "unknown"


def summarize_coords(coords: List[float]) -> Dict[str, Any]:
    gaps = coordinate_gaps(coords)
    first_gap = gaps[0] if gaps else 0.0
    terminal_gap = gaps[-1] if gaps else 0.0
    return {
        "monotonicity": monotonicity(coords),
        "first_gap": f"{first_gap:.10f}",
        "terminal_gap": f"{terminal_gap:.10f}",
        "tail_front_ratio": f"{(terminal_gap / first_gap):.6f}" if first_gap else "nan",
    }


def grid_distance(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(abs(float(a[i]) - float(b[i])) for i in range(n)) / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--backend", default="native", choices=["native", "diffusers"])
    parser.add_argument("--time_shifting_factor", type=float, default=6.0)
    args = parser.parse_args()

    root = ensure_experiment_dirs(args.experiment_root)
    rows: List[Dict[str, Any]] = []
    solvers = ["euler", "midpoint", "dpm"] if args.backend == "native" else ["flowmatch_euler"]
    direction = "descending" if args.backend == "diffusers" else "ascending"
    for solver in solvers:
        ctype = coord_type_for(solver, args.backend)
        for nfe in [10, 20, 30, 40, 50]:
            if solver == "dpm":
                uniform = build_uniform_coords(nfe, "native_time", args.time_shifting_factor, direction)
                bss_payload = {"final_coords": [], "base_coords": [], "inserted_midpoints": []}
                custom_safe = "false"
                note = "DPM custom solver-coordinate BSS is audit-only until safe lambda/logSNR injection is implemented."
            else:
                uniform = build_uniform_coords(nfe, ctype, args.time_shifting_factor, direction)
                bss_payload = build_bss_coords(nfe, ctype, args.time_shifting_factor, direction=direction)
                custom_safe = "true"
                note = ""
            summary = summarize_coords(uniform)
            bss_coords = bss_payload.get("final_coords", [])
            rows.append(
                {
                    "backend": args.backend,
                    "solver": solver,
                    "actual_nfe": nfe,
                    "coordinate_type": ctype,
                    "uniform_monotonicity": summary["monotonicity"],
                    "uniform_first_gap": summary["first_gap"],
                    "uniform_terminal_gap": summary["terminal_gap"],
                    "uniform_tail_front_ratio": summary["tail_front_ratio"],
                    "bss_custom_schedule_safe": custom_safe,
                    "bss_grid_distance_to_uniform": f"{grid_distance(uniform, bss_coords):.10f}" if bss_coords else "",
                    "inserted_midpoints": str(bss_payload.get("inserted_midpoints", "")),
                    "nfe_equals_model_eval_count": "solver-dependent" if solver == "midpoint" else "true",
                    "note": note,
                }
            )
    table_csv = root / "tables/table_scheduler_geometry_lumina2.csv"
    table_md = root / "tables/table_scheduler_geometry_lumina2.md"
    write_csv_rows(table_csv, rows)
    write_text(table_md, markdown_table(rows, [
        "backend",
        "solver",
        "actual_nfe",
        "coordinate_type",
        "uniform_first_gap",
        "uniform_terminal_gap",
        "bss_custom_schedule_safe",
        "bss_grid_distance_to_uniform",
        "nfe_equals_model_eval_count",
        "note",
    ]))
    try:
        import matplotlib.pyplot as plt

        for solver in solvers:
            if solver == "dpm":
                continue
            plt.figure(figsize=(7, 4))
            for nfe in [10, 20, 40, 50]:
                coords = build_uniform_coords(nfe, coord_type_for(solver, args.backend), args.time_shifting_factor, direction)
                plt.plot(range(len(coords)), coords, marker="o", label=f"uniform{nfe}")
            plt.title(f"Lumina2 {solver} scheduler coordinates")
            plt.xlabel("model evaluation index")
            plt.ylabel(coord_type_for(solver, args.backend))
            plt.legend()
            plt.tight_layout()
            plt.savefig(root / f"figures/scheduler_coords_{solver}.png", dpi=160)
            plt.close()
    except Exception as exc:
        write_text(root / "logs/scheduler_geometry_plot_error.log", repr(exc) + "\n")
    write_text(
        root / "reports/04_scheduler_geometry_audit.md",
        "\n".join(
            [
                "# Lumina-Image 2.0 Scheduler Geometry Audit",
                "",
                f"- Backend: `{args.backend}`",
                f"- Time shifting factor: `{args.time_shifting_factor}`",
                f"- Table: `{table_csv}`",
                "",
                "Euler and Midpoint use the native shifted transport-time approximation used by the scaffold.",
                "DPM remains audit-only until a safe solver-coordinate insertion is implemented.",
                "",
            ]
        ),
    )
    print(f"wrote {table_csv}")


if __name__ == "__main__":
    main()

