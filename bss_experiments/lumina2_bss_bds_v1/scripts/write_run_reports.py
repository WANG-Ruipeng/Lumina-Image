from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from experiment_utils import ensure_experiment_dirs, read_csv_rows, write_text


def status_counts(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.get("status", "")] = counts.get(row.get("status", ""), 0) + 1
    return counts


def fmt_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def first_existing(*paths: Path) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--smoke_manifest", required=True)
    parser.add_argument("--mini_manifest", required=True)
    parser.add_argument("--backend", default="")
    parser.add_argument("--solver", default="")
    args = parser.parse_args()

    run_root = ensure_experiment_dirs(args.run_root)
    smoke_rows = read_csv_rows(args.smoke_manifest)
    mini_rows = read_csv_rows(args.mini_manifest)
    smoke_counts = status_counts(smoke_rows)
    mini_counts = status_counts(mini_rows)
    completed_mini = sum(1 for row in mini_rows if row.get("status") == "completed")
    total_mini = len(mini_rows)

    bds_row_path = run_root / "tables/table_cross_model_same_compute_lumina2_row.csv"
    bds_rows = read_csv_rows(bds_row_path)
    verdict = bds_rows[0].get("BDS", "pending") if bds_rows else "pending"
    backend = args.backend or (smoke_rows[0].get("backend", "pending") if smoke_rows else "pending")
    solver = args.solver or (smoke_rows[0].get("solver", "pending") if smoke_rows else "pending")

    smoke_report = [
        "# Lumina-Image 2.0 Smoke Report",
        "",
        f"- Smoke manifest: `{args.smoke_manifest}`",
        f"- Status counts: `{fmt_counts(smoke_counts)}`",
        f"- Backend: `{backend}`",
        f"- Solver: `{solver}`",
        "",
    ]
    write_text(run_root / "reports/01_lumina2_smoke_report.md", "\n".join(smoke_report))

    mini_report = [
        "# Lumina-Image 2.0 Mini-Suite Run Report",
        "",
        f"- Mini manifest: `{args.mini_manifest}`",
        f"- Completed rows: `{completed_mini}/{total_mini}`",
        f"- Status counts: `{fmt_counts(mini_counts)}`",
        "",
    ]
    write_text(run_root / "reports/02_mini_suite_run_report.md", "\n".join(mini_report))

    final_lines = [
        "# Final Lumina-Image 2.0 BSS/BDS Report",
        "",
        "## 1. Purpose",
        "",
        "Scheduler-naturalness T2I probe after Sana Reject and FLUX Low-only. This is a same-compute BSS vs uniform experiment, not a universal BSS claim.",
        "",
        "## 2. Repo, Model, and Hardware Audit",
        "",
        f"- Audit report: `{first_existing(run_root / 'reports/00_repo_model_hardware_audit.md')}`",
        "",
        "## 3. Weight Location and Drive Setup",
        "",
        "- Expected Drive weight path: `/content/drive/MyDrive/ModelWeights/Lumina-Image-2.0/`",
        "- Expected Drive artifact mirror: `/content/drive/MyDrive/Colab_Projects/Lumina2-BSS-BDS/lumina2_bss_bds_v1/`",
        "",
        "## 4. Backend and Solver",
        "",
        f"- Backend selected: `{backend}`",
        f"- Solver selected: `{solver}`",
        "- Native Euler is preferred for the main scientific row. Diffusers is a fallback if native custom schedule injection is unavailable.",
        "",
        "## 5. BSS Schedule Implementation",
        "",
        "BSS constructs a base schedule with `T-2` scheduler coordinates, inserts midpoints into the first and last base intervals, and runs exactly `T` model evaluations. DPM remains audit-only unless a safe solver-coordinate injection is implemented.",
        "",
        "## 6. Smoke Result",
        "",
        f"- Smoke manifest status: `{fmt_counts(smoke_counts)}`",
        f"- Smoke report: `{run_root / 'reports/01_lumina2_smoke_report.md'}`",
        "",
        "## 7. Mini-Suite Completion",
        "",
        f"- Completed rows: `{completed_mini}/{total_mini}`",
        f"- Mini report: `{run_root / 'reports/02_mini_suite_run_report.md'}`",
        "",
        "## 8. Same-Compute RGB-L1 Closure Row",
        "",
        f"- CSV: `{first_existing(run_root / 'tables/table_cross_model_same_compute_lumina2_row.csv')}`",
        f"- Markdown: `{first_existing(run_root / 'tables/table_cross_model_same_compute_lumina2_row.md')}`",
        f"- LaTeX: `{first_existing(run_root / 'tables/table_cross_model_same_compute_lumina2_row.tex')}`",
        "",
        "## 9. BDS Calibration / Holdout",
        "",
        f"- BDS table: `{first_existing(run_root / 'tables/tableA_lumina2_bds_by_split.csv')}`",
        f"- BDS report: `{first_existing(run_root / 'reports/03_bds_report.md')}`",
        "",
        "## 10. Scheduler Geometry Audit",
        "",
        f"- Geometry table: `{first_existing(run_root / 'tables/table_scheduler_geometry_lumina2.csv')}`",
        f"- Geometry report: `{first_existing(run_root / 'reports/04_scheduler_geometry_audit.md')}`",
        "",
        "## 11. Optional Solver Ablation",
        "",
        f"- Solver ablation table: `{first_existing(run_root / 'tables/table_solver_ablation_lumina2.md')}`",
        "",
        "## 12. Figures",
        "",
        f"- Compute-quality figure: `{first_existing(run_root / 'figures/compute_quality_rgb_closure.png')}`",
        f"- Side-by-side index: `{first_existing(run_root / 'figures/side_by_side/index.html')}`",
        "",
        "## 13. Final Verdict",
        "",
        f"- Verdict: `{verdict}`",
        "- Include in cross-model paper table only after smoke, schedule validation, and mini-suite metrics are complete.",
        "",
        "## Caveats",
        "",
        "- Fixed prompt suite, not an official benchmark.",
        "- Reference is uniform50, not ground truth.",
        "- NFE is a compute proxy.",
        "- Native and Diffusers backends may differ.",
        "- Gemma/text encoder auth may be required.",
        "- Solver-specific schedule injection risk must be documented.",
        "",
    ]
    final_path = run_root / "reports/FINAL_LUMINA2_BSS_BDS_REPORT.md"
    write_text(final_path, "\n".join(final_lines))
    print(f"wrote {final_path}")


if __name__ == "__main__":
    main()

