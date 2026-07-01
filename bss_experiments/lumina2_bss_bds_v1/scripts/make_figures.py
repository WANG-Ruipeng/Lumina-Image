from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from experiment_utils import ensure_experiment_dirs, read_csv_rows, safe_float, safe_int, write_text


def make_quality_plot(run_root: Path) -> None:
    metrics_path = run_root / "metrics/master_long_metrics.csv"
    rows = read_csv_rows(metrics_path)
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_text(run_root / "logs/make_figures_error.log", repr(exc) + "\n")
        return
    grouped: Dict[str, Dict[int, List[float]]] = {"uniform": {}, "bss": {}}
    for row in rows:
        family = row.get("method_family")
        if family not in grouped:
            continue
        nfe = safe_int(row.get("actual_nfe"))
        grouped[family].setdefault(nfe, []).append(safe_float(row.get("rgb_l1_closure")))
    plt.figure(figsize=(6, 4))
    for family, points in grouped.items():
        xs = sorted(points)
        ys = [sum(points[x]) / len(points[x]) for x in xs]
        if xs:
            plt.plot([x / 50.0 for x in xs], ys, marker="o", label=family)
    plt.xlabel("NFE / reference NFE")
    plt.ylabel("RGB-L1 closure to uniform50")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_root / "figures/compute_quality_rgb_closure.png", dpi=160)
    plt.savefig(run_root / "figures/compute_quality_rgb_closure.pdf")
    plt.close()


def make_side_by_side_index(run_root: Path, manifest_path: Path) -> None:
    rows = read_csv_rows(manifest_path)
    by_case: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if Path(row.get("output_path", "")).exists():
            by_case.setdefault(row["case_id"], []).append(row)
    html = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>Lumina2 side by side</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.item img{max-width:100%;height:auto}.item{border:1px solid #ddd;padding:8px}</style>",
        "</head><body>",
        "<h1>Lumina-Image 2.0 Side by Side</h1>",
    ]
    for case_id, case_rows in sorted(by_case.items()):
        html.append(f"<h2>{case_id}</h2><div class=\"grid\">")
        for row in sorted(case_rows, key=lambda r: (safe_int(r.get("actual_nfe")), r.get("method", ""))):
            rel = Path(row["output_path"]).resolve().as_posix()
            html.append(f"<div class=\"item\"><div>{row['method']}</div><img src=\"{rel}\"></div>")
        html.append("</div>")
    html.append("</body></html>")
    write_text(run_root / "figures/side_by_side/index.html", "\n".join(html))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()
    run_root = ensure_experiment_dirs(args.run_root)
    make_quality_plot(run_root)
    make_side_by_side_index(run_root, Path(args.manifest))
    print(f"wrote figures under {run_root / 'figures'}")


if __name__ == "__main__":
    main()

