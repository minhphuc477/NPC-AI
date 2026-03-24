#!/usr/bin/env python3
"""Persistent local daemon for UE5 <-> Python SAGE bridge.

Reads request JSON files and writes matching response JSON files so UE5 can
avoid process-spawn overhead for every single BT task call.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

import sage_bt_handlers


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_args_field(raw_args: str) -> List[str]:
    text = (raw_args or "").strip()
    if not text:
        return []
    tokens: List[str] = []
    current: List[str] = []
    in_quotes = False
    for ch in text:
        if ch == '"':
            in_quotes = not in_quotes
            continue
        if ch.isspace() and not in_quotes:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def parse_request(req_path: Path) -> tuple[str, List[str]]:
    payload = json.loads(req_path.read_text(encoding="utf-8"))
    req_id = str(payload.get("id", "")).strip() or req_path.stem
    argv_raw = payload.get("argv")
    if isinstance(argv_raw, Sequence) and not isinstance(argv_raw, (str, bytes)):
        argv = [str(item) for item in argv_raw]
    else:
        argv = parse_args_field(str(payload.get("args", "")))
    return req_id, argv


def cleanup_stale_files(directory: Path, pattern: str, max_age_seconds: float) -> int:
    if max_age_seconds <= 0:
        return 0
    now = time.time()
    deleted = 0
    for path in directory.glob(pattern):
        try:
            age = now - path.stat().st_mtime
            if age > max_age_seconds:
                path.unlink(missing_ok=True)
                deleted += 1
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return deleted


def process_one(req_path: Path, response_dir: Path) -> None:
    req_id = req_path.stem
    result: Dict[str, Any] = {
        "id": req_id,
        "return_code": 1,
        "stdout": "",
        "stderr": "",
    }
    try:
        req_id, argv = parse_request(req_path)
        payload, code = sage_bt_handlers.run(argv)
        result["id"] = req_id
        result["return_code"] = int(code)
        result["stdout"] = json.dumps(payload, ensure_ascii=False)
        result["stderr"] = ""
    except SystemExit as exc:
        result["return_code"] = int(getattr(exc, "code", 1) or 1)
        result["stderr"] = f"system_exit:{result['return_code']}"
    except Exception as exc:
        result["return_code"] = 1
        result["stderr"] = f"exception:{exc}\\n{traceback.format_exc(limit=3)}"

    resp_path = response_dir / f"resp_{result['id']}.json"
    write_json_atomic(resp_path, result)
    try:
        req_path.unlink(missing_ok=True)
    except Exception as exc:
        if result.get("stderr"):
            result["stderr"] = f"{result['stderr']}\\ncleanup_error:{exc}"
        else:
            result["stderr"] = f"cleanup_error:{exc}"
        write_json_atomic(resp_path, result)


def run_loop(args: argparse.Namespace) -> int:
    request_dir = Path(args.request_dir)
    response_dir = Path(args.response_dir)
    shutdown_file = Path(args.shutdown_file) if str(args.shutdown_file or "").strip() else None

    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    poll_seconds = max(0.001, float(args.poll_ms) / 1000.0)
    idle_seconds = max(0.001, float(args.idle_sleep_ms) / 1000.0)
    stale_seconds = max(0.0, float(args.stale_ms) / 1000.0)
    cleanup_interval_seconds = max(0.5, float(args.cleanup_interval_ms) / 1000.0)
    last_cleanup_at = 0.0

    while True:
        if shutdown_file and shutdown_file.exists():
            return 0

        now = time.time()
        if (now - last_cleanup_at) >= cleanup_interval_seconds:
            cleanup_stale_files(request_dir, "req_*.json", stale_seconds)
            cleanup_stale_files(response_dir, "resp_*.json", stale_seconds)
            cleanup_stale_files(request_dir, "*.tmp*", stale_seconds)
            cleanup_stale_files(response_dir, "*.tmp*", stale_seconds)
            last_cleanup_at = now

        req_files = sorted(request_dir.glob("req_*.json"))
        if not req_files:
            time.sleep(idle_seconds)
            continue

        for req_path in req_files:
            process_one(req_path, response_dir)
            if args.once:
                return 0
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent SAGE bridge daemon.")
    parser.add_argument("--request-dir", required=True)
    parser.add_argument("--response-dir", required=True)
    parser.add_argument("--shutdown-file", default="")
    parser.add_argument("--poll-ms", type=float, default=8.0)
    parser.add_argument("--idle-sleep-ms", type=float, default=16.0)
    parser.add_argument("--stale-ms", type=float, default=180000.0)
    parser.add_argument("--cleanup-interval-ms", type=float, default=5000.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_loop(args))


if __name__ == "__main__":
    main()
