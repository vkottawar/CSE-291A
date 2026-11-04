"""Run one AgentDojo banking task through the workflow and grade it with AgentDojo's check."""

import time

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from .faults import FaultInjector, FaultSpec
from .trace import Trace
from .workflow import run_workflow

BENCHMARK_VERSION = "v1.2.2"

# Skips 1, 7, 8 (answer-only: nothing to recover) and 10 (its reference solution makes no payment).
USABLE_TASKS = [0, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15]


def run_task(task_id: str, llm, layout: str = "planner-first", fault: FaultSpec | None = None, return_env=False):
    if fault and fault.task != task_id:
        raise ValueError(f"fault {fault.id} is for {fault.task}, not {task_id}")
    suite = get_suite(BENCHMARK_VERSION, "banking")
    task = suite.get_user_task_by_id(task_id)
    trace = Trace(task_id=task_id, layout=layout)
    injector = FaultInjector(fault, trace) if fault else None
    # A researcher-stage prompt injection puts AgentDojo's payload into the world before it loads.
    injections = injector.prepare(suite, task, getattr(llm, "model", "")) if injector else {}
    env = task.init_environment(suite.load_and_inject_default_environment(injections))
    pre_env = env.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)

    start = time.time()
    report = None
    try:
        report = run_workflow(task.PROMPT, env, runtime, llm, trace, layout=layout, fault=injector)
        trace.utility = task.utility(report or "", pre_env, env)
    except Exception as e:
        trace.error = f"{type(e).__name__}: {e}"
        trace.utility = False
    trace.seconds = time.time() - start
    before = {t.id: t.model_dump() for t in pre_env.bank_account.transactions}
    after = {t.id: t.model_dump() for t in env.bank_account.transactions}
    trace.transaction_changes = {
        "before_count": len(before), "after_count": len(after),
        "added": [t for tid, t in after.items() if tid not in before],
        "modified": [t for tid, t in after.items() if tid in before and t != before[tid]],
        "removed_ids": [tid for tid in before if tid not in after],
    }
    if injector and not any(f["id"] == fault.id for f in trace.faults):
        injector.finish()  # the run crashed before the end; still record whether the fault was placed
    if injector:
        injector.grade(report, pre_env, env)  # prompt_injection: did the attacker's goal happen?
    return (task, trace, env) if return_env else (task, trace)
