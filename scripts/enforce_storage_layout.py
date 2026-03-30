#!/usr/bin/env python3
"""Enforce project storage layout by routing runtime dirs into storage/*.

Default mapping:
- artifacts -> storage/artifacts
- outputs -> storage/outputs
- releases -> storage/releases
- runs -> storage/runs
- runtime -> storage/runtime
- tmp_runs -> storage/tmp_runs

The script can:
1) migrate existing non-link source directories into storage/*
2) create source links (junctions on Windows) pointing at storage/*
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


MAPPING: Dict[str, str] = {
    "artifacts": "storage/artifacts",
    "outputs": "storage/outputs",
    "releases": "storage/releases",
    "runs": "storage/runs",
    "runtime": "storage/runtime",
    "tmp_runs": "storage/tmp_runs",
}


@dataclass
class Action:
    source: str
    target: str
    status: str
    note: str


def _is_reparse_point(path: Path) -> bool:
    try:
        return bool(path.lstat().st_file_attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except Exception:
        return path.is_symlink()


def _resolve_link_target(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return ""


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _move_children(src: Path, dst: Path) -> None:
    _ensure_dir(dst)
    for child in src.iterdir():
        dest = dst / child.name
        if dest.exists():
            if child.is_dir() and dest.is_dir():
                _move_children(child, dest)
                try:
                    child.rmdir()
                except OSError as exc:
                    # Keep moving other children; non-empty/locked dirs are handled later.
                    print(f"[warn] Could not remove merged directory {child}: {exc}")
            else:
                if child.is_file():
                    # Keep destination on collision.
                    continue
        else:
            shutil.move(str(child), str(dest))


def _create_link(source: Path, target: Path) -> None:
    if os.name == "nt":
        # Junction avoids admin requirement in common Windows setups.
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(source), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    source.symlink_to(target, target_is_directory=True)


def enforce_layout(root: Path, apply: bool, create_links: bool) -> List[Action]:
    actions: List[Action] = []
    for src_name, dst_name in MAPPING.items():
        src = root / src_name
        dst = root / dst_name
        _ensure_dir(dst)

        if src.exists():
            if _is_reparse_point(src):
                resolved = _resolve_link_target(src)
                if resolved and Path(resolved) == dst.resolve():
                    actions.append(Action(src_name, dst_name, "ok", "already linked"))
                else:
                    actions.append(Action(src_name, dst_name, "warn", f"link points to {resolved}"))
                continue

            if src.is_dir():
                if apply:
                    _move_children(src, dst)
                    try:
                        src.rmdir()
                    except OSError as exc:
                        print(f"[warn] Could not remove source directory {src}: {exc}")
                actions.append(Action(src_name, dst_name, "migrated" if apply else "plan", "directory merged to storage target"))
            else:
                if apply:
                    shutil.move(str(src), str(dst / src.name))
                actions.append(Action(src_name, dst_name, "migrated" if apply else "plan", "file moved into storage target"))
        else:
            actions.append(Action(src_name, dst_name, "missing", "source path absent"))

        if create_links and not src.exists():
            if apply:
                _create_link(src, dst)
            actions.append(Action(src_name, dst_name, "linked" if apply else "plan", "source linked to storage target"))

    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--create-links", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    actions = enforce_layout(root=root, apply=bool(args.apply), create_links=bool(args.create_links))
    for a in actions:
        print(f"[{a.status}] {a.source} -> {a.target} :: {a.note}")


if __name__ == "__main__":
    main()
