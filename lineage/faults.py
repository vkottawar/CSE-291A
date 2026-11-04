"""Wrong-fact fault injection at three stages: researcher, planner, executor.

Researcher-only extensions: type="timeout" raises InjectedResearcherTimeout before
the selected researcher's first model request (synthetic, not a network outage).
type="partial_output" removes one exact-name fact after valid JSON output and
before collection; original/injected outputs are recorded, JSON remains valid.
These extensions accept only stage="researcher", step=R1/R2/R3. Partial output
requires fact candidates; they do not use the wrong_fact change options below.

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
TYPES = ("wrong_fact", "timeout", "partial_output")


class InjectedResearcherTimeout(TimeoutError):
    """Synthetic pre-request failure, not evidence of a real provider timeout."""


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
        if self.type not in TYPES:
            raise ValueError(f"{self.id}: type must be one of {TYPES}")
        if self.stage not in STAGES:
            raise ValueError(f"{self.id}: stage must be one of {STAGES}")
        if isinstance(self.fact, str):
            self.fact = [self.fact]
        if isinstance(self.tool, str):
            self.tool = [self.tool]
        if self.type in ("timeout", "partial_output"):
            if self.stage != "researcher" or self.step not in ("R1", "R2", "R3"):
                raise ValueError(f"{self.id}: {self.type} requires researcher stage and R1/R2/R3")
            if self.type == "partial_output" and not self.fact:
                raise ValueError(f"{self.id}: partial_output requires exact fact-name candidates")
            if self.type == "partial_output" and any(not isinstance(name, str) or not name.strip() for name in self.fact):
                raise ValueError(f"{self.id}: partial_output fact names must be nonempty strings")
            if self.type == "timeout" and self.fact:
                raise ValueError(f"{self.id}: timeout targets a model request, not a fact")
            if self.change or self.tool or self.arg or self.executor or self.call_no is not None:
                raise ValueError(f"{self.id}: {self.type} does not accept wrong-fact change/call options")
            return
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

    def before_model_call(self, step_id, turn):
        """Timeout the target researcher's first attempt, before any provider request."""
        s = self.spec
        if self.applied or s.type != "timeout" or step_id != s.step or turn != 1:
            return
        self._log(step=step_id, target=f"{step_id}.model_call", turn=turn,
                  point="before_model_call", synthetic=True, provider_request_sent=False,
                  original="model request would be attempted", injected="timeout exception")
        raise InjectedResearcherTimeout(f"{s.id}: synthetic timeout before {step_id} model request")

    def on_researcher(self, step_id, output):
        """Called right after a researcher finishes, before its facts are collected."""
        s = self.spec
        if self.applied or s.stage != "researcher" or step_id != s.step:
            return
        facts = (output or {}).get("facts", []) if isinstance(output, dict) else []
        names = [f.get("name", "") for f in facts]
        if s.type == "timeout":
            return
        if s.type == "partial_output":
            # Exact names only: never accidentally remove another field by substring.
            i = next((names.index(name) for name in s.fact if name in names), None)
            if i is None:
                return
            original = copy.deepcopy(output)
            removed = facts.pop(i)
            self._log(step=step_id, target=f"{step_id}.{removed['name']}",
                      point="researcher_output_before_collection", operation="drop_fact",
                      original=original, injected=copy.deepcopy(output), removed=copy.deepcopy(removed))
            return
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

    def validate_handoff(self, facts, request=None):
        """Check the business payload, including facts and the original user request."""
        if not self.applied or self.spec.type != "partial_output":
            return
        event = next(e for e in self.trace.faults if e["id"] == self.spec.id and e["applied"])
        old = event["removed"]["value"]
        residual = _contains_value({"request": request, "facts": facts}, old)
        event.update(handoff_validated=not residual, residual_value_present=residual)
        if residual:
            raise ValueError(f"{self.spec.id}: removed value remains in planner inputs; planner not called")

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
            if self.spec.type == "timeout":
                self.trace.faults.append({**self._base(), "applied": False,
                    "reason": f"no first model-call attempt in {self.spec.step}"})
                return
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


def _contains_value(data, value):
    if isinstance(data, dict):
        return any(_contains_value(k, value) or _contains_value(v, value) for k, v in data.items())
    if isinstance(data, (list, tuple)):
        return any(_contains_value(v, value) for v in data)
    if isinstance(value, str):
        return bool(value) and isinstance(data, str) and value in data
    return type(data) is type(value) and data == value


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
