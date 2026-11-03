"""The run record: for every step, which fields it read, what it output, and which tools it called.

This is the dependency record that recovery (Phase 2) will walk.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ToolCallRecord:
    tool: str
    args: dict
    call_no: int  # 1-based count of calls this step has made; the ledger matches on (step, tool, call_no)
    result: str
    error: str | None


@dataclass
class StepRecord:
    step_id: str
    role: str
    reads: list[str]  # field-level references, e.g. "user.request", "R1.bill_amount", "P.subplans[0]"
    output: object = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    status: str = "ok"  # ok | bad_output | max_turns
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


@dataclass
class Trace:
    task_id: str
    layout: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    utility: bool | None = None
    error: str | None = None  # set when the run crashed (API error, bad tool args, ...); kept, not dropped
    seconds: float = 0.0
    started: float = field(default_factory=time.time)

    def tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    def print_summary(self) -> None:
        for s in self.steps:
            calls = ", ".join(f"{c.tool}({_short(c.args)})" for c in s.tool_calls) or "-"
            print(f"{s.step_id:4} {s.role:19} {s.status:10} reads={s.reads}")
            print(f"{'':4} tools: {calls}")
            print(f"{'':4} output: {_short(s.output, 150)}")
        for w in self.warnings:
            print(f"WARNING {w}")
        if self.error:
            print(f"ERROR {self.error}")
        print(f"utility: {self.utility}")


def _short(obj, n=80) -> str:
    text = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return text if len(text) <= n else text[: n - 3] + "..."
