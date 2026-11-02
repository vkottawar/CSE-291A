"""Run one AgentDojo banking task through the 7-agent workflow.

  python -m lineage --task user_task_0     (settings come from .env)
"""

import argparse
import time
from pathlib import Path

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from .llm import LLM
from .trace import Trace
from .workflow import run_workflow

BENCHMARK_VERSION = "v1.2.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="user_task_0")
    ap.add_argument("--model", help="overrides LLM_MODEL")
    ap.add_argument("--base-url", help="overrides LLM_BASE_URL")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    llm = LLM(model=args.model, base_url=args.base_url, seed=args.seed)

    suite = get_suite(BENCHMARK_VERSION, "banking")
    task = suite.get_user_task_by_id(args.task)
    env = task.init_environment(suite.load_and_inject_default_environment({}))
    pre_env = env.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)

    trace = Trace(task_id=args.task)
    report = run_workflow(task.PROMPT, env, runtime, llm, trace)
    trace.utility = task.utility(report or "", pre_env, env)

    print(f"TASK {args.task}: {task.PROMPT}\n")
    trace.print_summary()
    out = Path("runs") / f"{args.task}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    trace.save(out)
    print(f"trace saved to {out}")


if __name__ == "__main__":
    main()
