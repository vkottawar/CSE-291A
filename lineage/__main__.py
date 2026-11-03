"""Run one AgentDojo banking task through the 7-agent workflow.

  python -m lineage --task user_task_0     (settings come from .env)
"""

import argparse
import time
from pathlib import Path

from .llm import LLM
from .runner import run_task
from .workflow import LAYOUTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="user_task_0")
    ap.add_argument("--model", help="overrides LLM_MODEL")
    ap.add_argument("--base-url", help="overrides LLM_BASE_URL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=LAYOUTS, default=LAYOUTS[0])
    args = ap.parse_args()

    llm = LLM(model=args.model, base_url=args.base_url, seed=args.seed)
    task, trace = run_task(args.task, llm, layout=args.layout)

    print(f"TASK {args.task}: {task.PROMPT}\n")
    trace.print_summary()
    out = Path("runs") / f"{args.task}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    trace.save(out)
    print(f"trace saved to {out}")


if __name__ == "__main__":
    main()
