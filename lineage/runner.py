"""Run one AgentDojo banking task through the workflow and grade it with AgentDojo's check."""

import time

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from .trace import Trace
from .workflow import run_workflow

BENCHMARK_VERSION = "v1.2.2"

# Skips 1, 7, 8 (answer-only: nothing to recover) and 10 (its reference solution makes no payment).
USABLE_TASKS = [0, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15]


def run_task(task_id: str, llm):
    suite = get_suite(BENCHMARK_VERSION, "banking")
    task = suite.get_user_task_by_id(task_id)
    env = task.init_environment(suite.load_and_inject_default_environment({}))
    pre_env = env.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)

    trace = Trace(task_id=task_id)
    start = time.time()
    try:
        report = run_workflow(task.PROMPT, env, runtime, llm, trace)
        trace.utility = task.utility(report or "", pre_env, env)
    except Exception as e:
        trace.error = f"{type(e).__name__}: {e}"
        trace.utility = False
    trace.seconds = time.time() - start
    return task, trace
