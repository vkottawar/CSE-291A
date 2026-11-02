"""The fixed workflow: 3 researchers -> planner -> up to 2 executors -> reporter.

The code decides the order and what each agent sees; the planner decides the per-task content.
"""

from .agent import run_agent
from .trace import Trace

RESEARCHERS = [("R1", "files_researcher"), ("R2", "history_researcher"), ("R3", "account_researcher")]


def run_workflow(prompt, env, runtime, llm, trace: Trace):
    ctx = dict(env=env, runtime=runtime, llm=llm, trace=trace)

    facts = {}
    for sid, role in RESEARCHERS:
        out = run_agent(sid, role, {"request": prompt}, ["user.request"], **ctx)
        for f in (out or {}).get("facts", []):
            facts[f"{sid}.{f['name']}"] = f["value"]

    plan = run_agent("P", "planner", {"request": prompt, "facts": facts}, sorted(facts), **ctx) or {}

    results = {}
    # The sub-plans share no facts, so E1 and E2 are independent; they run one after the other here.
    for i, sub in enumerate(plan.get("subplans", [])[:2]):
        eid = f"E{i + 1}"
        uses = sub.get("uses", [])
        given = {k: facts[k] for k in uses if k in facts}
        missing = [k for k in uses if k not in facts]
        if missing:
            trace.warnings.append({"step": eid, "missing_facts": missing})
        subplan = {"goal": sub.get("goal"), "steps": sub.get("steps")}
        results[eid] = run_agent(eid, "executor", {"subplan": subplan, "facts": given}, [f"P.subplans[{i}]", *given], **ctx)

    reads = ["user.request", *sorted(facts), *[f"{e}.done" for e in results]]
    report = run_agent("REP", "reporter", {"request": prompt, "facts": facts, "executor_results": results}, reads, **ctx)
    return report
