"""Run many tasks several times and summarize pass rate, tokens, time, and whether repeats agree.

  python -m lineage.batch                          # all usable tasks, 3 repeats each
  python -m lineage.batch --tasks 0,3,11 --repeats 5

Writes every run record plus runs.csv and summary.json to runs/batch_<time>/.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from .llm import LLM
from .runner import USABLE_TASKS, run_task


def actions(trace) -> list[str]:
    """The world-changing calls with free-text fields (subject, date) removed, so repeats compare on
    what matters: tool, recipient, amount, ids. Sorted, because E1 and E2 may finish in either order."""
    out = []
    for s in trace.steps:
        if s.role != "executor":
            continue
        for c in s.tool_calls:
            args = {
                k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                for k, v in c.args.items()
                if k not in ("subject", "date")
            }
            out.append(f"{c.tool}({json.dumps(args, sort_keys=True)})")
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", help="comma-separated task numbers; default: all usable tasks")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", help="overrides LLM_MODEL")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    numbers = [int(n) for n in args.tasks.split(",")] if args.tasks else USABLE_TASKS
    llm = LLM(model=args.model, seed=args.seed)
    out_dir = Path("runs") / f"batch_{time.strftime('%Y%m%d-%H%M%S')}"

    rows = []
    for n in numbers:
        for r in range(1, args.repeats + 1):
            task_id = f"user_task_{n}"
            _, trace = run_task(task_id, llm)
            trace.save(out_dir / f"{task_id}_r{r}.json")
            row = {
                "task": n,
                "repeat": r,
                "passed": bool(trace.utility),
                "error": trace.error or "",
                "tokens": trace.tokens(),
                "seconds": round(trace.seconds, 1),
                "executors": sum(s.role == "executor" for s in trace.steps),
                "actions": " ; ".join(actions(trace)),
            }
            rows.append(row)
            status = "PASS" if row["passed"] else ("ERROR" if row["error"] else "fail")
            print(f"task {n:2} r{r}: {status:5} {row['tokens']:6} tok {row['seconds']:5}s  {row['actions'][:90]}", flush=True)

    with open(out_dir / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    print(f"\n{'task':>4}  {'passed':>7}  {'tokens (mean)':>13}  {'seconds (mean)':>14}  {'distinct action sets':>20}")
    for n in numbers:
        rs = [x for x in rows if x["task"] == n]
        s = {
            "task": n,
            "passed": sum(x["passed"] for x in rs),
            "runs": len(rs),
            "errors": sum(bool(x["error"]) for x in rs),
            "mean_tokens": round(statistics.mean(x["tokens"] for x in rs)),
            "mean_seconds": round(statistics.mean(x["seconds"] for x in rs), 1),
            "distinct_action_sets": len({x["actions"] for x in rs}),
        }
        summary.append(s)
        print(f"{n:>4}  {s['passed']:>3}/{s['runs']:<3}  {s['mean_tokens']:>13}  {s['mean_seconds']:>14}  {s['distinct_action_sets']:>20}")

    total_pass = sum(x["passed"] for x in rows)
    total_tokens = sum(x["tokens"] for x in rows)
    print(f"\noverall: {total_pass}/{len(rows)} passed, {sum(bool(x['error']) for x in rows)} errors, {total_tokens} tokens")
    (out_dir / "summary.json").write_text(
        json.dumps({"model": llm.model, "seed": args.seed, "repeats": args.repeats, "tasks": summary}, indent=2)
    )
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
