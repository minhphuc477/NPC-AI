#!/usr/bin/env python3
"""SAGE BT setup helper.

Usage:
  python scripts/BT_SETUP_GUIDE.py --print-guide
  python scripts/BT_SETUP_GUIDE.py --build-transition-matrix
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


GUIDE = """
SAGE BT Setup (UE5)
===================
1) Blackboard keys:
   Use `UNPCSageBlackboardKeyLibrary::GetDefaultSageBlackboardSchema()` as source of truth.
   Required new keys include:
   - SessionInitDone, StateHash, SessionTurnCount
   - MoodState, TrustScore, RelationshipScore, TrustEvent
   - EpisodicContext, PrefetchedPassages, InterruptFlag
   - GenerationTTFT, FallbackUsed

2) Recommended BT order:
   Service: SessionInit (once)
   Service: StateEmitter/StateTransitionDetector (tick)
   Service: RelationshipTracker (tick)
   Service: CrossNPCSync (0.5Hz)
   Task: PrefetchNextContext
   Task: Generate
   Task/Service: Memory writeback
   Decorator: ThreatInterrupt wraps dialogue sequence

3) Python bridge endpoints:
   Start local server:
   python scripts/sage_bt_http_server.py --host 127.0.0.1 --port 8000

   Implemented routes:
   - POST /api/generate
   - POST /api/warm_prefix
   - POST /api/invalidate_prefix
   - POST /api/episodic/recall
   - POST /api/episodic/write
   - POST /api/episodic/write_interrupt
   - POST /api/prefetch

4) Build transition matrix:
   python scripts/build_state_transition_matrix.py --input data/benchmark_scenarios.jsonl
   Then use `row_major` from output JSON for UE task config if needed.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE BT setup helper.")
    parser.add_argument("--print-guide", action="store_true")
    parser.add_argument("--build-transition-matrix", action="store_true")
    parser.add_argument("--input", default="data/proposal_eval_scenarios_large.jsonl")
    parser.add_argument("--out-json", default="storage/artifacts/datasets/state_transition_matrix.json")
    parser.add_argument("--out-md", default="storage/artifacts/datasets/state_transition_matrix.md")
    args = parser.parse_args()

    did_any = False
    if args.print_guide or (not args.print_guide and not args.build_transition_matrix):
        print(GUIDE.strip())
        did_any = True

    if args.build_transition_matrix:
        cmd = [
            sys.executable,
            "scripts/build_state_transition_matrix.py",
            "--input",
            str(args.input),
            "--out-json",
            str(args.out_json),
            "--out-md",
            str(args.out_md),
        ]
        print("running:", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "Failed to build transition matrix. "
                "Use --input <path_to_jsonl> with scenario turns/behavior_state."
            ) from exc
        did_any = True

    if not did_any:
        parser.print_help()


if __name__ == "__main__":
    main()
