"""For each faulted run: was the fault read, did later outputs change beyond normal variation, did the outcome flip.

  python -m lineage.effects --faulted runs/faults_<time>_planner-first
  python -m lineage.effects --faulted <dir> --clean results/clean-baseline --out <dir>

Writes effects.csv (one row per fault per run) and effects_summary.json to --out (default: --faulted).

Was it read? Three levels, because a step's `reads` alone says little: the planner and reporter list
every fact whatever they do with it.
  read_by   later steps that were given the faulted field
  used_in   later steps whose output or tool arguments contain the injected value
  acted_on  "exact": a successful tool call contains the injected value
            "derived": a successful tool call with an amount, id or recipient no clean repeat used,
            without the injected value. Inferred: e.g. task 3 turns a wrong 9.00 into a payment of 3.00.
  ignored   placed, but nothing above fired and nothing changed
  corrected executor faults: a later successful call put the original value back. For a payment
            that is a second payment, not a repair, and AgentDojo can still pass the run.
            partial_output: a successful call still has the removed value, so the model found it elsewhere.

partial_output removes a fact, so there is nothing to read or ignore: read_by, used_in and ignored stay
blank, and acted_on, corrected and changed show what the gap did.

Beyond normal variation? Compared with every clean repeat of the task, never one run with one run.
  actions_changed  the normalized action set (batch.actions) matches no clean repeat. Strict: an extra
                   first_name or a differently written city counts.
  changed_in       later steps holding an amount, IBAN or call found nowhere in the task's clean repeats
  changed          changed_in, or the amounts, ids and recipients of the run's calls match no clean
                   repeat, leaving out the injected call itself. An executor fault always changes its
                   own call, so this says whether anything else moved.

Outcome: held or flipped, only where the comparison means something. Otherwise:
  not_placed        the fault never found its target, so the run says nothing about it
  crashed           the run errored; trace.error says why
  stopped           an injected timeout ended the run, as designed. Not a detection and not a flip.
  invalid_fault     partial_output, but the removed value was still in the planner's inputs
  check_blind       AgentDojo's check for this task ignores the amount; use actions_changed instead
  clean_unreliable  the task doesn't pass every clean repeat, so a failure may not be the fault's doing
"""

import argparse
import copy
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from .batch import actions
from .trace import StepRecord, ToolCallRecord, Trace

# Tasks 5 and 6 pass whatever amount is paid; see "Inject a wrong fact" in the README.
BLIND_CHECKS = {"user_task_5", "user_task_6"}

# Not preceded or followed by word characters or dashes, so IBAN digits and dates don't count as numbers.
NUMBER = re.compile(r"(?<![\w.,-])\d[\d,]*(?:\.\d+)?(?![\w-])")
ANY_NUMBER = re.compile(r"\d+(?:\.\d+)?")
IBAN = re.compile(r"\b[A-Z]{2}\d{6,}\b")


def load_trace(path) -> Trace:
    d = json.loads(Path(path).read_text())
    steps = [StepRecord(**{**s, "tool_calls": [ToolCallRecord(**c) for c in s["tool_calls"]]}) for s in d.pop("steps")]
    return Trace(**d, steps=steps)


def run_files(folder) -> list[Path]:
    skip = ("summary.json", "effects_summary.json")
    return [p for p in sorted(Path(folder).glob("*.json")) if p.name not in skip and not p.name.endswith(".world.json")]


def load_clean(folder) -> dict[str, list[Trace]]:
    clean = defaultdict(list)
    for p in run_files(folder):
        t = load_trace(p)
        clean[t.task_id].append(t)
    return dict(clean)


# -- comparing values ----------------------------------------------------------------------------


def step_calls(step, ok_only=False) -> list[str]:
    """This step's tool calls, normalized exactly as batch.actions does for a whole run."""
    calls = [c for c in step.tool_calls if c.error is None or not ok_only]
    return actions(SimpleNamespace(steps=[SimpleNamespace(role=step.role, tool_calls=calls)]))


def key_calls(step, ok_only=False) -> list[str]:
    """Calls cut down to amounts, ids and recipient. Models vary the rest from run to run, such as
    adding first_name or writing the city as "New York, NY 10001", without any fault involved."""
    out = []
    for c in step.tool_calls:
        if step.role == "executor" and (c.error is None or not ok_only):
            args = {k: as_number(v) for k, v in c.args.items() if k not in ("subject", "date") and as_number(v) is not None}
            if "recipient" in c.args:
                args["recipient"] = c.args["recipient"]
            out.append(f"{c.tool}({json.dumps(args, sort_keys=True)})")
    return out


def key_actions(trace) -> tuple:
    return tuple(sorted(k for s in trace.steps for k in key_calls(s)))


def step_text(step) -> str:
    """What the step produced, leaving out tool results: those echo the arguments, not the agent's thinking."""
    return json.dumps(step.output, ensure_ascii=False, default=str) + " " + " ".join(step_calls(step))


def numbers(text) -> set[float]:
    return {float(m.replace(",", "")) for m in NUMBER.findall(text)}


def as_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def contains(text, value) -> bool:
    """Is this value in the text? Numbers compare by value, so 9, 9.0 and "9.00" are the same."""
    if value is None:
        return False
    n = as_number(value)
    if n is not None:
        return n in numbers(text)
    return str(value).casefold() in text.casefold()


def fault_value(event):
    """What to look for in later steps. Only the words that changed, because a fact holding a full
    address reaches the tool call as just its street. Timeouts and removed facts leave no value."""
    if event.get("type") in ("timeout", "partial_output"):
        return None
    original, injected = event.get("original"), event.get("injected")
    if isinstance(original, str) and isinstance(injected, str) and as_number(injected) is None:
        old = {w.strip(",.") for w in original.split()}
        return " ".join(w.strip(",.") for w in injected.split() if w.strip(",.") not in old) or injected
    return injected


def tokens(step) -> set:
    """The parts of a step that should repeat across clean runs. Free wording is left out."""
    text = step_text(step)
    return numbers(text) | set(IBAN.findall(text)) | set(key_calls(step))


def normal_tokens(traces) -> set:
    """Everything clean runs of a task ever mention, tool results included. Prose restates facts at
    will (a balance, a year, "1." in a list), so a value only counts as new if it is nowhere in here."""
    normal = set()
    for t in traces:
        for s in t.steps:
            text = step_text(s) + " ".join(str(c.result) for c in s.tool_calls)
            normal |= tokens(s) | numbers(text) | set(IBAN.findall(text))
            normal |= {float(n) for n in ANY_NUMBER.findall(text)}  # also the ones inside dates and file names
    return normal


def steps_by_id(traces) -> dict[str, list[StepRecord]]:
    found = defaultdict(list)
    for t in traces:
        for s in t.steps:
            found[s.step_id].append(s)
    return found


def changed_steps(later, clean_traces) -> list[str]:
    clean, normal = steps_by_id(clean_traces), normal_tokens(clean_traces)
    return [s.step_id for s in later if s.step_id not in clean or tokens(s) - normal]


def action_sets(traces) -> set[tuple]:
    return {tuple(actions(t)) for t in traces}


# -- where a fault can reach ---------------------------------------------------------------------


def later_steps(trace, event) -> list[StepRecord]:
    """The steps that ran on the faulted value. Researchers don't read each other, so they never count."""
    if event["stage"] == "researcher":
        return [s for s in trace.steps if s.role in ("planner", "executor", "reporter")]
    start = event.get("executor") or event["step"]  # an executor fault can still affect that step's later turns
    ids = [s.step_id for s in trace.steps]
    return trace.steps[ids.index(start):] if start in ids else []


def readers(later, event) -> list[str]:
    """Later steps given the faulted field. A planner fault reaches only its executor: the reporter's
    copy of the fact is still correct, so the reporter counts only through that executor's report."""
    stage, target = event["stage"], event.get("target")
    if stage == "researcher":
        return [s.step_id for s in later if target in s.reads]
    eid = event.get("executor") or event["step"]
    if stage == "planner":
        return [s.step_id for s in later if (s.step_id == eid and target in s.reads) or f"{eid}.done" in s.reads]
    return [s.step_id for s in later if f"{eid}.done" in s.reads]


def injected_call(trace, event):
    """The tool call an executor fault changed, or None (other stages, or the run crashed first)."""
    if event["stage"] != "executor":
        return None
    tool = event["target"].split(".", 1)[0]
    for s in trace.steps:
        if s.step_id == event["step"]:
            return next((c for c in s.tool_calls if c.tool == tool and c.call_no == event.get("call_no")), None)
    return None


def without_injected_call(trace, event) -> Trace:
    """A copy with the executor fault's own argument put back, so only what followed from it can count."""
    if injected_call(trace, event) is None:
        return trace
    trace = copy.deepcopy(trace)
    injected_call(trace, event).args[event["target"].split(".", 1)[1]] = event["original"]
    return trace


# -- the three questions -------------------------------------------------------------------------


def acted_on(trace, event, clean_traces, seen_clean) -> str:
    """exact / derived / no, from successful tool calls only: a rejected call changed nothing."""
    call = injected_call(trace, event)
    if call is not None:
        return "exact" if call.error is None else "no"
    calls = [c for s in later_steps(trace, event) for c in step_calls(s, ok_only=True)]
    if any(contains(c, fault_value(event)) for c in calls) and not seen_clean:
        return "exact"
    keys = {k for s in later_steps(trace, event) for k in key_calls(s, ok_only=True)}
    normal = {k for t in clean_traces for s in t.steps for k in key_calls(s)}
    return "derived" if keys - normal else "no"


def corrected(trace, event) -> bool | str:
    """Did a later call bring back the value the fault took away? Blank where that can't happen."""
    if event.get("type") == "partial_output":
        calls = [c for s in later_steps(trace, event) for c in step_calls(s, ok_only=True)]
        return any(contains(c, event["removed"]["value"]) for c in calls)
    call = injected_call(trace, event)
    if call is None:
        return ""
    arg = event["target"].split(".", 1)[1]
    step = next(s for s in trace.steps if s.step_id == event["step"])
    return any(c.tool == call.tool and c.call_no > call.call_no and c.error is None
               and contains(json.dumps(c.args.get(arg)), event["original"]) for c in step.tool_calls)


def outcome(trace, event, clean_traces) -> str:
    if not event.get("applied"):
        return "not_placed"
    if event.get("handoff_validated") is False:
        return "invalid_fault"
    if trace.error:
        return "stopped" if trace.error.startswith("InjectedResearcherTimeout") else "crashed"
    if trace.task_id in BLIND_CHECKS:
        return "check_blind"
    if not all(t.utility for t in clean_traces):
        return "clean_unreliable"
    return "held" if trace.utility else "flipped"


def effect(trace, event, clean_traces) -> dict:
    """One row for one fault in one run."""
    row = {
        "fault_id": event.get("id"),
        "task": trace.task_id,
        "type": event.get("type"),
        "stage": event.get("stage"),
        "applied": bool(event.get("applied")),
        "target": event.get("target", ""),
        "original": json.dumps(event.get("original")),
        "injected": json.dumps(event.get("injected")),
        "read_by": "",
        "used_in": "",
        "acted_on": "",
        "ignored": "",
        "corrected": "",
        "actions_changed": "",
        "changed_in": "",
        "changed": "",
        "outcome": outcome(trace, event, clean_traces),
        "passed": bool(trace.utility),
        "error": trace.error or "",
    }
    if not row["applied"] or row["outcome"] in ("stopped", "invalid_fault"):
        return row  # nothing ran on the fault, so there are no effects to measure

    rest = without_injected_call(trace, event)
    later = later_steps(rest, event)
    clean = steps_by_id(clean_traces)
    value = fault_value(event)

    # A value that clean runs also produce here (a small id, say) proves nothing, so it doesn't count as used.
    used = [s.step_id for s in later
            if contains(step_text(s), value) and not any(contains(step_text(c), value) for c in clean.get(s.step_id, []))]
    seen_clean = any(contains(step_text(c), value) for s in later for c in clean.get(s.step_id, []))
    changed_in = sorted(set(changed_steps(later, clean_traces)) | set(used), key=[s.step_id for s in later].index)
    changed = bool(changed_in) or key_actions(rest) not in {key_actions(t) for t in clean_traces}
    acted = acted_on(trace, event, clean_traces, seen_clean)

    row.update({
        "read_by": " ".join(readers(later, event)),
        "used_in": " ".join(used),
        "acted_on": acted,
        "ignored": "" if event.get("type") == "partial_output" else not used and not changed and acted == "no",
        "corrected": corrected(trace, event),
        "actions_changed": tuple(actions(trace)) not in action_sets(clean_traces),
        "changed_in": " ".join(changed_in),
        "changed": changed,
    })
    return row


def false_alarms(clean) -> dict:
    """Run the change detector on clean runs, each against its other repeats. There is no fault, so
    every hit is noise. A bit pessimistic: a faulted run is compared with one more repeat than this."""
    alarms = []
    for task, traces in sorted(clean.items()):
        for i, t in enumerate(traces):
            others = traces[:i] + traces[i + 1:]
            later = [s for s in t.steps if s.role in ("planner", "executor", "reporter")]
            alarms.append({
                "task": task,
                "actions": key_actions(t) not in {key_actions(o) for o in others},
                "steps": changed_steps(later, others),
            })
    return {
        "runs": len(alarms),
        "actions_changed": sum(a["actions"] for a in alarms),
        "any_step_changed": sum(bool(a["steps"]) for a in alarms),
        "by_step": {sid: sum(sid in a["steps"] for a in alarms) for sid in sorted({x for a in alarms for x in a["steps"]})},
        "tasks": sorted({a["task"] for a in alarms if a["actions"] or a["steps"]}),
    }


def summarize(rows) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["type"], r["stage"])].append(r)
    summary = []
    for (ftype, stage), rs in sorted(groups.items()):
        placed = [r for r in rs if r["applied"]]
        comparable = [r for r in placed if r["outcome"] in ("held", "flipped")]
        summary.append({
            "type": ftype,
            "stage": stage,
            "runs": len(rs),
            "placed": len(placed),
            "ignored": sum(r["ignored"] is True for r in placed),
            "used": sum(bool(r["used_in"]) for r in placed),
            "acted_exact": sum(r["acted_on"] == "exact" for r in placed),
            "acted_derived": sum(r["acted_on"] == "derived" for r in placed),
            "corrected": sum(r["corrected"] is True for r in placed),
            "changed": sum(r["changed"] is True for r in placed),
            "actions_changed": sum(r["actions_changed"] is True for r in placed),
            "flipped": sum(r["outcome"] == "flipped" for r in comparable),
            "comparable": len(comparable),
            "crashed": sum(r["outcome"] == "crashed" for r in placed),
            "stopped": sum(r["outcome"] == "stopped" for r in placed),
            "invalid": sum(r["outcome"] == "invalid_fault" for r in placed),
        })
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--faulted", required=True, help="folder written by lineage.fault_batch")
    ap.add_argument("--clean", default="results/clean-baseline")
    ap.add_argument("--out", help="default: the --faulted folder")
    args = ap.parse_args(argv)

    clean = load_clean(args.clean)
    rows, no_fault = [], []
    for path in run_files(args.faulted):
        trace = load_trace(path)
        if not trace.faults:
            no_fault.append(path.name)
            continue
        if trace.task_id not in clean:
            raise SystemExit(f"{path.name}: no clean runs of {trace.task_id} in {args.clean}")
        if trace.layout != clean[trace.task_id][0].layout:
            raise SystemExit(f"{path.name}: layout {trace.layout} differs from the clean runs")
        repeat = re.search(r"_r(\d+)$", path.stem)
        for event in trace.faults:
            rows.append({**effect(trace, event, clean[trace.task_id]), "repeat": int(repeat[1]) if repeat else ""})
    if not rows:
        raise SystemExit(f"no faulted runs in {args.faulted}")

    out_dir = Path(args.out or args.faulted)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "effects.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    noise = false_alarms({r["task"]: clean[r["task"]] for r in rows})
    print(f"{'type':14} {'stage':10} {'placed':>9} {'ignored':>7} {'used':>5} {'exact':>5} {'derived':>7} {'fixed':>5} {'changed':>7} {'flipped':>9} {'crashed':>7} {'stopped':>7} {'invalid':>7}")
    for s in summary:
        print(f"{s['type']:14} {s['stage']:10} {s['placed']:>4}/{s['runs']:<4} {s['ignored']:>7} {s['used']:>5} {s['acted_exact']:>5} "
              f"{s['acted_derived']:>7} {s['corrected']:>5} {s['changed']:>7} {s['flipped']:>4}/{s['comparable']:<4} {s['crashed']:>7} {s['stopped']:>7} {s['invalid']:>7}")
    print(f"\nnoise floor on clean runs: actions changed in {noise['actions_changed']}/{noise['runs']}, "
          f"some step changed in {noise['any_step_changed']}/{noise['runs']} {noise['by_step']}")
    if no_fault:
        print(f"skipped {len(no_fault)} runs with no fault record: {', '.join(no_fault)}")
    (out_dir / "effects_summary.json").write_text(
        json.dumps({"clean": args.clean, "faulted": args.faulted, "groups": summary, "noise_floor": noise, "skipped": no_fault}, indent=2)
    )
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
