"""The fixed workflow, in one of two layouts:

  planner-first (default): research planner (P0) -> the researchers it asks -> planner -> executors -> reporter
  researchers-first:       3 researchers -> planner -> up to 2 executors -> reporter

The code decides the order and what each agent sees; the planners decide the per-task content.
"""

from .agent import run_agent
from .trace import Trace

RESEARCHERS = [("R1", "files_researcher"), ("R2", "history_researcher"), ("R3", "account_researcher")]
LAYOUTS = ["planner-first", "researchers-first"]  # the first is the default


def run_workflow(prompt, env, runtime, llm, trace: Trace, layout: str = "planner-first", fault=None):
    """fault: an optional FaultInjector (faults.py); without it the run is exactly the clean workflow."""
    ctx = dict(env=env, runtime=runtime, llm=llm, trace=trace)

    if layout == "planner-first":
        questions = (run_agent("P0", "research_planner", {"request": prompt}, ["user.request"], **ctx) or {}).get("questions", {})
        # A researcher P0 asks nothing is skipped; each one asked sees only its own question.
        research = [
            (sid, role, {"request": prompt, "question": questions[sid]}, ["user.request", f"P0.questions.{sid}"])
            for sid, role in RESEARCHERS
            if questions.get(sid)
        ]
    else:
        research = [(sid, role, {"request": prompt}, ["user.request"]) for sid, role in RESEARCHERS]

    facts = {}
    for sid, role, inputs, reads in research:
        out = run_agent(sid, role, inputs, reads, **ctx,
                        before_chat=fault.before_model_call if fault else None)
        if fault:
            fault.on_researcher(sid, out)
        for f in (out or {}).get("facts", []):
            key = f"{sid}.{f['name']}"
            if key in facts:  # never let a second fact with the same name overwrite the first
                n = 2
                while f"{key}_{n}" in facts:
                    n += 1
                trace.warnings.append({"step": sid, "duplicate_fact": f["name"], "stored_as": f"{key}_{n}"})
                key = f"{key}_{n}"
            facts[key] = f["value"]

    if fault:
        fault.validate_handoff(facts, prompt)
    plan = run_agent("P", "planner", {"request": prompt, "facts": facts}, sorted(facts), **ctx) or {}

    results = {}
    subplans = plan.get("subplans", [])
    if len(subplans) > 2:  # only two executors exist; never drop work silently
        trace.warnings.append({"step": "P", "dropped_subplans": [s.get("goal") for s in subplans[2:]]})
    # The sub-plans share no facts, so E1 and E2 are independent; they run one after the other here.
    for i, sub in enumerate(subplans[:2]):
        eid = f"E{i + 1}"
        uses = sub.get("uses", [])
        given = {k: facts[k] for k in uses if k in facts}
        missing = [k for k in uses if k not in facts]
        if missing:
            trace.warnings.append({"step": eid, "missing_facts": missing})
        subplan = {"goal": sub.get("goal"), "steps": sub.get("steps")}
        if fault:
            subplan, given = fault.on_subplan(eid, subplan, given)
        hook = fault.on_tool_call if fault else None
        results[eid] = run_agent(
            eid, "executor", {"subplan": subplan, "facts": given}, [f"P.subplans[{i}]", *given], **ctx, arg_hook=hook,
            result_hook=fault.on_tool_result if fault else None,
        )
        # An executor's own report can't be trusted: flag claimed actions with no successful tool call.
        done = (results[eid] or {}).get("done") or []
        if done and not any(c.error is None for c in trace.steps[-1].tool_calls):
            trace.warnings.append({"step": eid, "claimed_without_action": done})

    reads = ["user.request", *sorted(facts), *[f"{e}.done" for e in results]]
    report = run_agent("REP", "reporter", {"request": prompt, "facts": facts, "executor_results": results}, reads, **ctx)
    if fault:
        fault.finish()
    return report
