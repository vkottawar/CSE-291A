"""Run fault specs several times each and save the faulted runs.

  python -m lineage.fault_batch                                   # every spec in faults/wrong_fact.json, 3 repeats
  python -m lineage.fault_batch --tasks 0 --stages researcher --repeats 1
  python -m lineage.fault_batch --ids t0-executor,t15-planner

Writes to runs/faults_<time>/:
  <fault id>_r<n>.json        the run record, with trace.faults saying what was changed
  <fault id>_r<n>.world.json  the banking world after the run; restart baselines start from this
  runs.csv                    one row per run
"""

import argparse
import csv
import json
import time
from pathlib import Path

from .batch import actions
from .faults import STAGES, load_specs
from .llm import LLM
from .runner import run_task
from .workflow import LAYOUTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="faults/wrong_fact.json")
    ap.add_argument("--ids", help="comma-separated fault ids; default: all")
    ap.add_argument("--tasks", help="comma-separated task numbers; default: all")
    ap.add_argument("--stages", help=f"comma-separated, from {','.join(STAGES)}; default: all")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", help="overrides LLM_MODEL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=LAYOUTS, default=LAYOUTS[0])
    args = ap.parse_args()

    specs = load_specs(args.specs)
    if args.ids:
        wanted = set(args.ids.split(","))
        specs = [s for s in specs if s.id in wanted]
    if args.tasks:
        specs = [s for s in specs if int(s.task.rsplit("_", 1)[1]) in {int(n) for n in args.tasks.split(",")}]
    if args.stages:
        specs = [s for s in specs if s.stage in args.stages.split(",")]
    if not specs:
        raise SystemExit("no fault specs match those filters")

    llm = LLM(model=args.model, seed=args.seed)
    out_dir = Path("runs") / f"faults_{time.strftime('%Y%m%d-%H%M%S')}_{args.layout}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in specs:
        for r in range(1, args.repeats + 1):
            _, trace, env = run_task(spec.task, llm, layout=args.layout, fault=spec, return_env=True)
            trace.save(out_dir / f"{spec.id}_r{r}.json")
            (out_dir / f"{spec.id}_r{r}.world.json").write_text(env.model_dump_json(indent=2))
            event = next((f for f in trace.faults if f["id"] == spec.id), {})
            row = {
                "fault_id": spec.id,
                "task": spec.task,
                "stage": spec.stage,
                "type": spec.type,
                "repeat": r,
                "applied": bool(event.get("applied")),
                "target": event.get("target", ""),
                "original": json.dumps(event.get("original")),
                "injected": json.dumps(event.get("injected")),
                "not_applied_reason": event.get("reason", ""),
                "passed": bool(trace.utility),
                "error": trace.error or "",
                "tokens": trace.tokens(),
                "seconds": round(trace.seconds, 1),
                "actions": " ; ".join(actions(trace)),
            }
            rows.append(row)
            placed = "placed" if row["applied"] else "NOT PLACED"
            status = "PASS" if row["passed"] else ("ERROR" if row["error"] else "fail")
            print(f"{spec.id:16} r{r}: {placed:10} {status:5} {row['original']}->{row['injected']}  {row['actions'][:70]}", flush=True)

    with open(out_dir / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    placed = sum(r["applied"] for r in rows)
    print(f"\n{len(rows)} runs, {placed} with the fault placed, {sum(r['passed'] for r in rows)} still passed, "
          f"{sum(bool(r['error']) for r in rows)} errors")
    (out_dir / "summary.json").write_text(
        json.dumps({"model": llm.model, "layout": args.layout, "seed": args.seed, "repeats": args.repeats,
                    "specs": args.specs, "fault_ids": [s.id for s in specs]}, indent=2)
    )
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
