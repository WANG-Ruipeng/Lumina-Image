from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Patch anchor not found for {label}")
    return text.replace(old, new, 1)


def patch_integrators(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = replace_once(
        text,
        "time_shifting_factor=None,\n    ):",
        "time_shifting_factor=None,\n        custom_time_grid=None,\n    ):",
        "integrators.ode signature",
    )
    text = replace_once(
        text,
        "self.t = th.linspace(t0, t1, num_steps)\n        if time_shifting_factor:\n            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)",
        "if custom_time_grid is not None:\n            self.t = th.as_tensor(custom_time_grid, dtype=th.float32)\n        else:\n            self.t = th.linspace(t0, t1, num_steps)\n            if time_shifting_factor:\n                self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)",
        "integrators.ode time grid",
    )
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_transport(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = replace_once(
        text,
        "time_shifting_factor=None,\n    ):",
        "time_shifting_factor=None,\n        custom_time_grid=None,\n    ):",
        "transport.sample_ode signature",
    )
    text = replace_once(
        text,
        "time_shifting_factor=time_shifting_factor,\n        )",
        "time_shifting_factor=time_shifting_factor,\n            custom_time_grid=custom_time_grid,\n        )",
        "transport.sample_ode ode call",
    )
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_sample(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = replace_once(
        text,
        "import json\n",
        "import json\n",
        "sample import json presence",
    )
    if "custom_time_grid_json" not in text:
        text = replace_once(
            text,
            "sample_fn = sampler.sample_ode(\n                    sampling_method=args.solver,\n                    num_steps=args.num_sampling_steps,\n                    atol=args.atol,\n                    rtol=args.rtol,\n                    reverse=args.reverse,\n                    time_shifting_factor=args.t_shift,\n                )",
            "custom_time_grid = None\n                if args.custom_time_grid_json:\n                    with open(args.custom_time_grid_json, \"r\", encoding=\"utf-8\") as grid_file:\n                        custom_time_grid = json.load(grid_file)[\"final_coords\"]\n                sample_fn = sampler.sample_ode(\n                    sampling_method=args.solver,\n                    num_steps=args.num_sampling_steps,\n                    atol=args.atol,\n                    rtol=args.rtol,\n                    reverse=args.reverse,\n                    time_shifting_factor=args.t_shift,\n                    custom_time_grid=custom_time_grid,\n                )",
            "sample.py sample_ode call",
        )
        text = replace_once(
            text,
            "parser.add_argument(\"--hf_token\", type=str, default=None, help=\"huggingface read token for accessing gated repo.\")",
            "parser.add_argument(\"--hf_token\", type=str, default=None, help=\"huggingface read token for accessing gated repo.\")\n    parser.add_argument(\"--custom_time_grid_json\", type=str, default=\"\", help=\"Optional JSON schedule with final_coords for BSS.\")",
            "sample.py parser argument",
        )
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumina_root", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root = Path(args.lumina_root)
    targets = {
        "integrators": root / "transport/integrators.py",
        "transport": root / "transport/transport.py",
        "sample": root / "sample.py",
    }
    for label, path in targets.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} file missing: {path}")
    if args.dry_run:
        print("patch anchors will be checked without writing")
        return
    changed = {
        "integrators": patch_integrators(targets["integrators"]),
        "transport": patch_transport(targets["transport"]),
        "sample": patch_sample(targets["sample"]),
    }
    print(changed)


if __name__ == "__main__":
    main()

