from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_once(text: str, pattern: str, repl: str, label: str, flags: int = 0) -> str:
    new_text, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Patch anchor not found for {label}")
    return new_text


def add_param_after_time_shift(text: str, label: str) -> str:
    if "custom_time_grid=None" in text:
        return text
    pattern = r"(?P<indent>[ \t]*)time_shifting_factor=None,\n(?P=indent)\):"
    repl = r"\g<indent>time_shifting_factor=None,\n\g<indent>custom_time_grid=None,\n\g<indent>):"
    return replace_once(text, pattern, repl, label)


def patch_integrators(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = add_param_after_time_shift(text, "integrators.ode signature")
    if "custom_time_grid is not None" not in text:
        pattern = (
            r"(?P<indent>[ \t]*)self\.t = th\.linspace\(t0, t1, num_steps\)\n"
            r"(?P=indent)if time_shifting_factor:\n"
            r"(?P=indent)(?P<body_indent>[ \t]+)self\.t = self\.t / "
            r"\(self\.t \+ time_shifting_factor - time_shifting_factor \* self\.t\)"
        )
        repl = (
            r"\g<indent>if custom_time_grid is not None:\n"
            r"\g<indent>\g<body_indent>self.t = th.as_tensor(custom_time_grid, dtype=th.float32)\n"
            r"\g<indent>else:\n"
            r"\g<indent>\g<body_indent>self.t = th.linspace(t0, t1, num_steps)\n"
            r"\g<indent>\g<body_indent>if time_shifting_factor:\n"
            r"\g<indent>\g<body_indent>\g<body_indent>self.t = self.t / "
            r"(self.t + time_shifting_factor - time_shifting_factor * self.t)"
        )
        text = replace_once(text, pattern, repl, "integrators.ode time grid")
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_transport(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = add_param_after_time_shift(text, "transport.sample_ode signature")
    if "custom_time_grid=custom_time_grid" not in text:
        pattern = r"(?P<indent>[ \t]*)time_shifting_factor=time_shifting_factor,\n(?P=indent)\)"
        repl = (
            r"\g<indent>time_shifting_factor=time_shifting_factor,\n"
            r"\g<indent>custom_time_grid=custom_time_grid,\n"
            r"\g<indent>)"
        )
        text = replace_once(text, pattern, repl, "transport.sample_ode ode call")
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_sample(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    if "custom_time_grid=custom_time_grid" not in text:
        pattern = (
            r"(?P<indent>[ \t]*)sample_fn = sampler\.sample_ode\(\n"
            r"(?P<body>(?:(?!\n(?P=indent)\)).)*?time_shifting_factor=args\.t_shift,?\n)"
            r"(?P=indent)\)"
        )
        repl = (
            r"\g<indent>custom_time_grid = None\n"
            r"\g<indent>if args.custom_time_grid_json:\n"
            r'\g<indent>    with open(args.custom_time_grid_json, "r", encoding="utf-8") as grid_file:\n'
            r'\g<indent>        custom_time_grid = json.load(grid_file)["final_coords"]\n'
            r"\g<indent>sample_fn = sampler.sample_ode(\n"
            r"\g<body>"
            r"\g<indent>    custom_time_grid=custom_time_grid,\n"
            r"\g<indent>)"
        )
        text = replace_once(text, pattern, repl, "sample.py sample_ode call", flags=re.DOTALL)
        text = re.sub(
            r"(time_shifting_factor=args\.t_shift),?\n(?P<indent>[ \t]*custom_time_grid=custom_time_grid,)",
            r"\1,\n\g<indent>",
            text,
            count=1,
        )
    if "--custom_time_grid_json" not in text:
        pattern = (
            r"(?P<indent>[ \t]*)parser\.add_argument\("
            r"\"--hf_token\", type=str, default=None, "
            r"help=\"huggingface read token for accessing gated repo\.\"\)"
        )
        repl = (
            r'\g<indent>parser.add_argument("--hf_token", type=str, default=None, '
            r'help="huggingface read token for accessing gated repo.")\n'
            r'\g<indent>parser.add_argument("--custom_time_grid_json", type=str, default="", '
            r'help="Optional JSON schedule with final_coords for BSS.")'
        )
        text = replace_once(text, pattern, repl, "sample.py parser argument")
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
        for label, path in targets.items():
            print(label, "exists", path)
        return
    changed = {
        "integrators": patch_integrators(targets["integrators"]),
        "transport": patch_transport(targets["transport"]),
        "sample": patch_sample(targets["sample"]),
    }
    print(changed)


if __name__ == "__main__":
    main()
