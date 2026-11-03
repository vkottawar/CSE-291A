"""Checks for wrong-fact injection. No API key needed.

  .venv/bin/python -m unittest tests.test_faults -v

A fake model replays a clean run from results/clean-baseline step by step, in the real AgentDojo
banking world. The replay doesn't react to what it is shown, so these tests check where each fault
lands (inputs, run record, world), not how a real model would respond to it.
"""

import json
import unittest
from pathlib import Path

from lineage.faults import FaultSpec, load_specs, mutate
from lineage.llm import Reply, ToolCallRequest
from lineage.runner import run_task

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "results" / "clean-baseline"
SPECS = ROOT / "faults" / "wrong_fact.json"


class ReplayLLM:
    """Answers each step with the tool calls, then the output, recorded in a clean run."""

    def __init__(self, trace_path):
        steps = json.loads(Path(trace_path).read_text())["steps"]
        self.steps = {s["step_id"]: s for s in steps}
        self.turn = {}
        self.inputs = {}  # step id -> what that agent was shown

    def chat(self, step_id, messages, tools):
        s = self.steps[step_id]
        self.inputs.setdefault(step_id, json.loads(messages[1]["content"]))
        n = self.turn.get(step_id, 0)
        self.turn[step_id] = n + 1
        if n == 0 and s["tool_calls"]:
            calls = [ToolCallRequest(f"{step_id}-{c['call_no']}", c["tool"], dict(c["args"])) for c in s["tool_calls"]]
            return Reply(content=None, tool_calls=calls)
        out = s["output"]
        return Reply(content=out if isinstance(out, str) else json.dumps(out))


def run(task_n, fault=None, repeat=1):
    llm = ReplayLLM(CLEAN / f"user_task_{task_n}_r{repeat}.json")
    task, trace, env = run_task(f"user_task_{task_n}", llm, fault=fault, return_env=True)
    return llm, trace, env


def spec(fault_id):
    return next(s for s in load_specs(SPECS) if s.id == fault_id)


def payments(env):
    """Payments the run made: AgentDojo stores the user's IBAN as sender and the workflow pays on TODAY."""
    return [t for t in env.bank_account.transactions if t.recipient != "me" and t.date == "2022-04-01"]


# AgentDojo's checks for these tasks pass whatever amount the agent pays, so they can't show a flip:
#   5: passes if ANY 50.00 payment to Spotify exists, and the March payment already in history is one.
#   6: looks at the FIRST scheduled payment mentioning Spotify, which is the existing standing order.
VACUOUS_CHECK = {"user_task_5", "user_task_6"}


def world_values(env):
    acct = env.bank_account
    return [t.amount for t in acct.transactions + acct.scheduled_transactions] + [env.user_account.street]


class TestMutate(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(mutate(98.7, {"kind": "scale", "factor": 0.9}), 88.83)
        self.assertEqual(mutate(10, {"kind": "scale", "factor": 0.9}), 9)
        self.assertEqual(mutate(7, {"kind": "offset", "delta": 1}), 8)
        self.assertEqual(mutate("UK12345678901234567890", {"kind": "digit"}), "UK12345678901234567891")
        self.assertEqual(mutate("Dalton Street 123, New York", {"kind": "digit"}), "Dalton Street 124, New York")
        self.assertEqual(mutate("98.70"), "88.83")
        self.assertEqual(mutate("x", {"kind": "set", "value": "y"}), "y")

    def test_no_op_change_is_an_error(self):
        with self.assertRaises(ValueError):
            mutate(0, {"kind": "scale", "factor": 0.9})
        with self.assertRaises(ValueError):
            mutate("no digits here", {"kind": "digit"})


class TestSpecs(unittest.TestCase):
    def test_spec_file_loads(self):
        specs = load_specs(SPECS)
        self.assertEqual(len(specs), 24)
        self.assertEqual(len({s.id for s in specs}), 24)
        self.assertEqual({s.stage for s in specs}, {"researcher", "planner", "executor"})

    def test_bad_spec_rejected(self):
        with self.assertRaises(ValueError):
            FaultSpec(id="x", task="user_task_0", stage="researcher", fact=["amount"])  # no step
        with self.assertRaises(ValueError):
            FaultSpec(id="x", task="user_task_0", stage="nowhere")


class TestTask0(unittest.TestCase):
    """Task 0: pay the bill in the file (98.70 to UK12...890)."""

    def test_clean_replay_passes(self):
        _, trace, env = run(0)
        self.assertTrue(trace.utility, trace.error)
        self.assertEqual(trace.faults, [])
        self.assertEqual([p.amount for p in payments(env)], [98.7])

    def test_researcher_stage(self):
        llm, trace, _ = run(0, spec("t0-researcher"))
        [ev] = trace.faults
        self.assertTrue(ev["applied"])
        self.assertEqual((ev["step"], ev["target"], ev["original"], ev["injected"]), ("R1", "R1.bill_amount", 98.7, 88.83))
        # the researcher's recorded output holds what downstream read
        r1 = next(s for s in trace.steps if s.step_id == "R1")
        self.assertIn(88.83, [f["value"] for f in r1.output["facts"]])
        # planner and executor were shown the wrong value
        self.assertEqual(llm.inputs["P"]["facts"]["R1.bill_amount"], 88.83)
        self.assertEqual(llm.inputs["E1"]["facts"]["R1.bill_amount"], 88.83)

    def test_planner_stage(self):
        llm, trace, _ = run(0, spec("t0-planner"))
        [ev] = trace.faults
        self.assertEqual((ev["step"], ev["executor"], ev["target"], ev["injected"]), ("P", "E1", "R1.bill_amount", 88.83))
        # the planner saw the right value, the researcher's output is untouched, only E1 got it wrong
        self.assertEqual(llm.inputs["P"]["facts"]["R1.bill_amount"], 98.7)
        r1 = next(s for s in trace.steps if s.step_id == "R1")
        self.assertIn(98.7, [f["value"] for f in r1.output["facts"]])
        self.assertEqual(llm.inputs["E1"]["facts"]["R1.bill_amount"], 88.83)

    def test_executor_stage_changes_the_world(self):
        _, trace, env = run(0, spec("t0-executor"))
        [ev] = trace.faults
        self.assertEqual((ev["step"], ev["target"], ev["call_no"], ev["original"], ev["injected"]),
                         ("E1", "send_money.amount", 1, 98.7, 88.83))
        e1 = next(s for s in trace.steps if s.step_id == "E1")
        self.assertEqual(e1.tool_calls[0].args["amount"], 88.83)  # the record shows the wrong call
        self.assertEqual([p.amount for p in payments(env)], [88.83])  # and the world has the wrong payment
        self.assertFalse(trace.utility)  # AgentDojo's check flips

    def test_fault_not_placed_is_recorded(self):
        s = FaultSpec(id="miss", task="user_task_0", stage="researcher", step="R2", fact=["bill_amount"])
        _, trace, _ = run(0, s)
        [ev] = trace.faults
        self.assertFalse(ev["applied"])
        self.assertIn("R2", ev["reason"])
        self.assertTrue(trace.utility)

    def test_fires_once(self):
        s = FaultSpec(id="once", task="user_task_0", stage="researcher", step="R1", fact=["bill", "amount"])
        _, trace, _ = run(0, s)
        self.assertEqual(len(trace.faults), 1)

    def test_wrong_task_rejected(self):
        with self.assertRaises(ValueError):
            run(3, spec("t0-researcher"))


class TestAllSpecsPlace(unittest.TestCase):
    """Every spec finds its target in at least one clean replay of its task, so none is dead on arrival.
    Real runs can still name facts differently; fault_batch reports those as NOT PLACED."""

    def test_each_spec_places(self):
        for s in load_specs(SPECS):
            n = int(s.task.rsplit("_", 1)[1])
            placed = []
            for r in (1, 2, 3):
                _, trace, _ = run(n, s, repeat=r)
                placed.append(trace.faults[0]["applied"])
            with self.subTest(fault=s.id):
                self.assertTrue(any(placed), f"{s.id} never placed in 3 clean replays")
            print(f"  {s.id:16} placed in {sum(placed)}/3 replays: {trace.faults[0].get('target', trace.faults[0].get('reason'))}")

    def test_executor_faults_change_the_world(self):
        for s in [s for s in load_specs(SPECS) if s.stage == "executor"]:
            n = int(s.task.rsplit("_", 1)[1])
            _, clean_trace, _ = run(n)
            _, trace, _ = run(n, s)
            _, _, env = run(n, s)
            with self.subTest(fault=s.id):
                self.assertTrue(clean_trace.utility)
                self.assertTrue(trace.faults[0]["applied"])
                injected = trace.faults[0]["injected"]
                self.assertTrue(any(injected == v or (isinstance(v, str) and str(injected) in v) for v in world_values(env)),
                                f"{s.id}: {injected!r} not found in the world")
                if s.task in VACUOUS_CHECK:
                    self.assertTrue(trace.utility)  # documents the blind spot; if AgentDojo fixes it, this fails
                else:
                    self.assertFalse(trace.utility, f"{s.id}: AgentDojo still passed with {trace.faults[0]}")


if __name__ == "__main__":
    unittest.main()
