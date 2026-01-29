"""Utility: convert gatekeeper CSV dataset to JSONL for annotation tasks.

Features:
- Supports writing to stdout by passing "-" as output path
- Robust parsing and normalization of headers (case-insensitive)
- Skips rows missing required fields and reports counts
"""
import csv
import json
import sys
from pathlib import Path
from typing import Optional


def csv_to_jsonl(csv_path: str, jsonl_path: Optional[str] = None) -> int:
    """Convert a CSV to JSONL.

    If jsonl_path is '-' or None, writes to stdout. Returns number of rows written.
    Expected CSV headers (case-insensitive): instruction, input, output, optional npc_state
    """
    csvp = Path(csv_path)
    if jsonl_path is None or jsonl_path == "-":
        wf = sys.stdout
        close_out = False
    else:
        wf = open(jsonl_path, "w", encoding="utf-8")
        close_out = True

    written = 0
    skipped = 0
    with csvp.open("r", encoding="utf-8") as rf:
        reader = csv.DictReader(rf)
        # normalize fieldnames to lowercase for convenience
        lower_map = {k: k for k in reader.fieldnames or []}
        # wrap reader iteration
        for row in reader:
            # create a case-insensitive view
            row_ci = {k.lower().strip(): (v.strip() if v is not None else "") for k, v in row.items()}
            instr = row_ci.get("instruction") or row_ci.get("prompt") or row_ci.get("context") or ""
            output = row_ci.get("output") or row_ci.get("response") or row_ci.get("utterance") or ""
            inp = row_ci.get("input") or ""
            npc_state = row_ci.get("npc_state") or None
            # skip clearly malformed rows
            if not instr or not output:
                skipped += 1
                continue
            # build a compact representation that is useful in annotation
            text = instr
            if inp:
                # some datasets separate input column; append with a separating space
                text = f"{text} {inp}".strip()
            text = f"{text}\n\nNPC: {output}"

            rec = {
                "instruction": instr,
                "input": inp,
                "output": output,
                "text": text,
            }
            if npc_state:
                rec["npc_state"] = npc_state

            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    if close_out:
        wf.close()
    # write a short report to stderr
    print(f"Wrote {written} rows, skipped {skipped} rows", file=sys.stderr)
    return written


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Convert a gatekeeper CSV to JSONL for annotation tasks")
    p.add_argument("csv", help="input CSV path")
    p.add_argument("jsonl", nargs="?", default="-", help="output JSONL path or '-' for stdout (default)")
    args = p.parse_args()
    count = csv_to_jsonl(args.csv, args.jsonl)
    # exit code 0 if at least one row written
    if count == 0:
        sys.exit(2)
    sys.exit(0)
