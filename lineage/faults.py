"""Wrong-fact fault injection at three stages: researcher, planner, executor.

A fault spec is a small JSON object. The same spec is reused by every recovery method, so all of
them face the identical fault. Example:

  {"id": "t0-researcher", "task": "user_task_0", "type": "wrong_fact", "stage": "researcher",
   "step": "R1", "fact": ["bill_amount", "amount"], "change": {"kind": "scale", "factor": 0.9}}

Stages and what they target:

  researcher  "step" (R1, R2 or R3) and "fact": the value of one fact in that researcher's output
              is changed before the planner sees it. The researcher's recorded output holds the
              wrong value (it is what everyone downstream read); the original is kept in trace.faults.
  planner     "fact" (optional "executor": "E1"/"E2"): the planner hands one executor a wrong value
              for a fact its sub-plan uses. The researcher's output stays correct, so only that
              executor (and what it does) is affected.
  executor    "tool" and "arg" (optional "call_no"): one argument of one world-changing tool call is
              changed before the call runs, as if the executor itself got it wrong. The run record
              shows the wrong arguments.

"fact" is a list of name candidates, because the model names facts differently from run to run:
an exact name match wins, then the first candidate that is a substring of a fact name.

"change" kinds (default: scale by 0.9 for numbers, bump the last digit for strings):
  scale  {"factor": 0.9}     numbers only, rounded to 2 decimals
  offset {"delta": 1}        numbers only (good for ids)
  digit  {}                  bump the last digit in a string ("UK...890" -> "UK...891", "Street 123" -> "Street 124")
  set    {"value": ...}      replace outright

A spec fires at most once per run. If its target never shows up, trace.faults records applied=False
with the reason: the fault was not placed, which is different from a fault that was placed but ignored.
"""

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ("researcher", "planner", "executor")


@dataclass
class FaultSpec:
    id: str
    task: str
    stage: str
    type: str = "wrong_fact"
    step: str | None = None  # researcher stage: R1 / R2 / R3
    fact: list[str] = field(default_factory=list)  # researcher and planner stages
    executor: str | None = None  # planner stage: limit to E1 or E2
    tool: list[str] = field(default_factory=list)  # executor stage
    arg: str | None = None  # executor stage
    call_no: int | None = None  # executor stage: 1-based call number within the step, as in ToolCallRecord
    change: dict = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if self.type != "wrong_fact":
            raise ValueError(f"{self.id}: only wrong_fact is implemented here, got {self.type}")
        if self.stage not in STAGES:
            raise ValueError(f"{self.id}: stage must be one of {STAGES}")
        if isinstance(self.fact, str):
            self.fact = [self.fact]
        if isinstance(self.tool, str):
            self.tool = [self.tool]
        if self.stage == "researcher" and not (self.step and self.fact):
            raise ValueError(f"{self.id}: researcher faults need 'step' and 'fact'")
        if self.stage == "planner" and not self.fact:
            raise ValueError(f"{self.id}: planner faults need 'fact'")
        if self.stage == "executor" and not (self.tool and self.arg):
            raise ValueError(f"{self.id}: executor faults need 'tool' and 'arg'")


def load_specs(path) -> list[FaultSpec]:
    data = json.loads(Path(path).read_text())
    return [FaultSpec(**d) for d in (data if isinstance(data, list) else [data])]


def load_spec(path, fault_id=None) -> FaultSpec:
    specs = load_specs(path)
    if fault_id:
        found = [s for s in specs if s.id == fault_id]
        if not found:
            raise SystemExit(f"no fault with id {fault_id} in {path}")
        return found[0]
    if len(specs) != 1:
        raise SystemExit(f"{path} has {len(specs)} faults; pick one with --fault-id")
    return specs[0]


class FaultInjector:
    """Holds one spec for one run and applies it at the right hook. The workflow calls the hooks."""

    def __init__(self, spec: FaultSpec, trace):
        self.spec = spec
        self.trace = trace
        self.applied = False

    # -- hooks ---------------------------------------------------------------------------------

    def on_researcher(self, step_id, output):
        """Called right after a researcher finishes, before its facts are collected."""
        s = self.spec
        if self.applied or s.stage != "researcher" or step_id != s.step:
            return
        facts = (output or {}).get("facts", []) if isinstance(output, dict) else []
        names = [f.get("name", "") for f in facts]
        i = _match(names, s.fact)
        if i is None:
            return  # reported as not placed at the end of the run
        f = facts[i]
        new = mutate(f["value"], s.change)
        self._log(step=step_id, target=f"{step_id}.{f['name']}", original=f["value"], injected=new)
        f["value"] = new  # in place: the recorded output now shows what downstream steps actually read

    def on_subplan(self, eid, subplan, given):
        """Called before an executor runs, with the sub-plan text and the fact values it will get."""
        s = self.spec
        if self.applied or s.stage != "planner" or (s.executor and s.executor != eid):
            return subplan, given
        keys = list(given)
        i = _match([k.split(".", 1)[-1] for k in keys], s.fact)
        if i is None:
            return subplan, given
        key = keys[i]
        old = given[key]
        new = mutate(old, s.change)
        given = {**given, key: new}
        subplan = copy.deepcopy(subplan)
        if isinstance(subplan.get("steps"), str):  # keep the sub-plan text consistent with the wrong value
            subplan["steps"] = _replace_value(subplan["steps"], old, new)
        self._log(step="P", executor=eid, target=key, original=old, injected=new)
        return subplan, given

    def on_tool_call(self, step_id, tool, args, call_no):
        """Called before a tool call runs. Returns the (possibly changed) arguments."""
        s = self.spec
        if self.applied or s.stage != "executor" or tool not in s.tool or s.arg not in args:
            return args
        if s.call_no is not None and call_no != s.call_no:
            return args
        old = args[s.arg]
        new = mutate(old, s.change)
        self._log(step=step_id, target=f"{tool}.{s.arg}", call_no=call_no, original=old, injected=new)
        return {**args, s.arg: new}

    def finish(self):
        """Called once at the end of the run. Records a fault that never found its target."""
        if not self.applied:
            where = {"researcher": f"fact {self.spec.fact} in {self.spec.step}",
                     "planner": f"fact {self.spec.fact} in any sub-plan",
                     "executor": f"{self.spec.arg} in a {self.spec.tool} call"}[self.spec.stage]
            self.trace.faults.append({**self._base(), "applied": False, "reason": f"no {where}"})

    # -- helpers -------------------------------------------------------------------------------

    def _base(self):
        return {"id": self.spec.id, "type": self.spec.type, "stage": self.spec.stage}

    def _log(self, **event):
        self.applied = True
        self.trace.faults.append({**self._base(), "applied": True, **event})


def _match(names, candidates):
    """Index of the first name that equals a candidate, else the first that contains one."""
    for c in candidates:
        if c in names:
            return names.index(c)
    for c in candidates:
        for i, n in enumerate(names):
            if c in n:
                return i
    return None


def mutate(value, change=None):
    """Deterministic wrong value. Raises if the change can't apply, so a bad spec fails loudly."""
    change = change or {}
    kind = change.get("kind") or ("scale" if _is_number(value) or _numeric_str(value) else "digit")

    if kind == "set":
        new = change["value"]
    elif kind in ("scale", "offset"):
        if _numeric_str(value):  # "98.70" stays a string, with the same number of decimals
            decimals = len(value.split(".")[1]) if "." in value else 0
            return f"{mutate(float(value), change):.{decimals}f}"
        if not _is_number(value):
            raise ValueError(f"{kind} needs a number, got {value!r}")
        if kind == "scale":
            new = round(value * change.get("factor", 0.9), 2)
        else:
            new = value + change.get("delta", 1)
        if isinstance(value, int) and float(new).is_integer():
            new = int(new)
    elif kind == "digit":
        if not isinstance(value, str) or not re.search(r"\d", value):
            raise ValueError(f"digit needs a string with a digit, got {value!r}")
        m = list(re.finditer(r"\d", value))[-1]
        new = value[: m.start()] + str((int(m.group()) + 1) % 10) + value[m.end():]
    else:
        raise ValueError(f"unknown change kind {kind}")

    if new == value:  # a fault that changes nothing would silently count as "ignored"
        raise ValueError(f"change {change} leaves {value!r} unchanged")
    return new


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_str(v):
    return isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v.strip()) is not None


def _replace_value(text, old, new):
    """Replace a value written in free text, e.g. 98.7 / 98.70 -> 88.83. Leaves text alone if absent."""
    if _is_number(old):
        pattern = rf"(?<![\d.]){re.escape(str(old))}0*(?![\d])"
        return re.sub(pattern, str(new), text)
    return text.replace(str(old), str(new)) if isinstance(old, str) and old else text
