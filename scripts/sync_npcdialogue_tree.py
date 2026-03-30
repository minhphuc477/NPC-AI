#!/usr/bin/env python3
"""Validate or sync NPCDialogue dual trees.

Canonical source of truth is:
  ue5/Plugins/NPCInference/Source/NPCDialogue

Mirror tree:
  ue5/Source/NPCDialogue
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "ue5" / "Plugins" / "NPCInference" / "Source" / "NPCDialogue"
MIRROR = ROOT / "ue5" / "Source" / "NPCDialogue"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_canon_files() -> list[Path]:
    return sorted(p for p in CANON.rglob("*") if p.is_file())


def sync() -> tuple[int, int]:
    copied = 0
    deleted = 0
    MIRROR.mkdir(parents=True, exist_ok=True)

    canon_files = iter_canon_files()
    canon_rel = {p.relative_to(CANON) for p in canon_files}

    for src in canon_files:
        rel = src.relative_to(CANON)
        dst = MIRROR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256(src) != sha256(dst):
            shutil.copy2(src, dst)
            copied += 1

    for dst in sorted(p for p in MIRROR.rglob("*") if p.is_file()):
        rel = dst.relative_to(MIRROR)
        if rel not in canon_rel:
            dst.unlink()
            deleted += 1

    return copied, deleted


def check() -> list[str]:
    issues: list[str] = []
    canon_files = iter_canon_files()
    canon_rel = {p.relative_to(CANON) for p in canon_files}

    for src in canon_files:
        rel = src.relative_to(CANON)
        dst = MIRROR / rel
        if not dst.exists():
            issues.append(f"missing:{rel.as_posix()}")
            continue
        if sha256(src) != sha256(dst):
            issues.append(f"mismatch:{rel.as_posix()}")

    for dst in sorted(p for p in MIRROR.rglob("*") if p.is_file()):
        rel = dst.relative_to(MIRROR)
        if rel not in canon_rel:
            issues.append(f"extra:{rel.as_posix()}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync/check NPCDialogue dual trees")
    parser.add_argument("--fix", action="store_true", help="copy canonical tree into mirror and remove extras")
    args = parser.parse_args()

    if args.fix:
        copied, deleted = sync()
        print(f"synced copied={copied} deleted={deleted}")
        return 0

    issues = check()
    if not issues:
        print("sync_ok")
        return 0

    print("sync_mismatch")
    for item in issues:
        print(item)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
