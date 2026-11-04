"""Checks for prompt injection (A1). No API key needed.

  .venv/bin/python -m unittest tests.test_prompt_injection -v

The replay model from test_faults ignores what it is shown, so most tests check where the payload
lands. ObeyingLLM follows any injection it sees, to check that a successful attack is graded and
measured as one.
"""

import json
import unittest

from lineage.effects import effect, load_clean
from lineage.faults import FaultSpec, load_specs
from lineage.injections import contains_payload
from lineage.llm import Reply, ToolCallRequest
from lineage.runner import run_task
from tests.test_faults import CLEAN, ROOT, ReplayLLM, run

SPECS = ROOT / "faults" / "prompt_injection.json"


def a1(task_n, stage, goal="injection_task_7", **kw):
    return FaultSpec(id=f"t{task_n}-{stage}-test", task=f"user_task_{task_n}", type="prompt_injection",
                     stage=stage, injection_task=goal, **kw)


def event(trace):
    return trace.faults[0]


def step(trace, step_id):
    return next(s for s in trace.steps if s.step_id == step_id)


class ObeyingLLM(ReplayLLM):
    """Replays the clean run, except that an executor shown the password injection changes the password."""

    def chat(self, step_id, messages, tools):
        if step_id.startswith("E") and "new_password" in json.dumps(messages) and not self.turn.get(f"{step_id}-obeyed"):
            if messages[-1]["role"] == "tool" or "new_password" in messages[1]["content"]:
                self.turn[f"{step_id}-obeyed"] = 1
                return Reply(content=None, tool_calls=[ToolCallRequest("x", "update_password", {"password": "new_password"})])
        return super().chat(step_id, messages, tools)


def obey(task_n, fault):
    llm = ObeyingLLM(CLEAN / f"user_task_{task_n}_r1.json")
    _, trace, env = run_task(f"user_task_{task_n}", llm, fault=fault, return_env=True)
    return trace, env


class TestSpecs(unittest.TestCase):
    def test_spec_file_loads(self):
        specs = load_specs(SPECS)
        self.assertEqual(len(specs), 48)
        self.assertEqual({s.stage for s in specs}, {"researcher", "planner", "executor"})
        self.assertEqual(len({s.id for s in specs}), 48)

    def test_bad_specs_fail_loudly(self):
        with self.assertRaises(ValueError):
            a1(0, "planner", goal=None)
        with self.assertRaises(ValueError):
            a1(0, "planner", fact=["bill_amount"])
        with self.assertRaises(ValueError):
            a1(0, "planner", step="R1")
        with self.assertRaises(ValueError):
            a1(0, "researcher", executor="E1")
        with self.assertRaises(ValueError):
            a1(0, "researcher", vector_mode="overwrite")
        with self.assertRaises(ValueError):
            a1(0, "planner", attack="no_such_attack")


class TestResearcherStage(unittest.TestCase):
    def test_bill_file_carries_payload_and_keeps_real_text(self):
        _, trace, _ = run(0, a1(0, "researcher"))
        e = event(trace)
        self.assertTrue(e["applied"])
        self.assertEqual((e["step"], e["target"]), ("R1", "R1.read_file"))
        self.assertEqual(e["vectors"], ["injection_bill_text"])
        result = step(trace, "R1").tool_calls[0].result
        self.assertTrue(contains_payload(result, e["payload"]))
        self.assertIn("98.70", result)  # append mode: the real bill line is still there
        self.assertIs(e["attack_succeeded"], False)  # the replay ignores it

    def test_replace_mode_drops_real_text(self):
        _, trace, _ = run(0, a1(0, "researcher", vector_mode="replace"))
        result = step(trace, "R1").tool_calls[0].result
        self.assertTrue(contains_payload(result, event(trace)["payload"]))
        self.assertNotIn("98.70", result)

    def test_transaction_subject_reaches_history_researcher(self):
        _, trace, _ = run(3, a1(3, "researcher"))
        e = event(trace)
        self.assertTrue(e["applied"])
        self.assertEqual(e["step"], "R2")
        self.assertEqual(e["vectors"], ["injection_incoming_transaction"])

    def test_not_placed_when_named_researcher_never_reads_it(self):
        _, trace, _ = run(0, a1(0, "researcher", step="R3"))
        e = event(trace)
        self.assertFalse(e["applied"])
        self.assertIn("R3", e["reason"])
        self.assertIs(e["attack_succeeded"], False)


class TestPlannerStage(unittest.TestCase):
    def test_executor_gets_extra_fact(self):
        llm, trace, _ = run(0, a1(0, "planner"))
        e = event(trace)
        self.assertTrue(e["applied"])
        self.assertEqual((e["step"], e["executor"]), ("P", "E1"))
        given = llm.inputs["E1"]["facts"]
        self.assertEqual(given[e["target"]], e["payload"])
        self.assertIn(e["target"], step(trace, "E1").reads)
        self.assertNotIn(e["target"], llm.inputs["P"]["facts"])  # the planner's own inputs are clean


class TestExecutorStage(unittest.TestCase):
    def test_payload_in_first_tool_result(self):
        _, trace, _ = run(0, a1(0, "executor"))
        e = event(trace)
        self.assertTrue(e["applied"])
        self.assertEqual((e["step"], e["target"], e["call_no"]), ("E1", "send_money.result", 1))
        call = step(trace, "E1").tool_calls[0]
        self.assertTrue(contains_payload(call.result, e["payload"]))
        self.assertTrue(call.result.startswith(e["original"]))

    def test_not_placed_without_that_executor(self):
        _, trace, _ = run(0, a1(0, "executor", executor="E2"))
        self.assertFalse(event(trace)["applied"])


class TestGradingAndEffects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = load_clean(CLEAN)

    def test_obeyed_injection_is_graded_as_attack(self):
        for stage in ("planner", "executor"):
            with self.subTest(stage=stage):
                trace, env = obey(0, a1(0, stage))
                self.assertEqual(env.user_account.password, "new_password")
                self.assertIs(event(trace)["attack_succeeded"], True)
                row = effect(trace, event(trace), self.clean["user_task_0"])
                self.assertEqual(row["acted_on"], "exact")
                self.assertIs(row["attack_succeeded"], True)
                self.assertIn("E1", row["read_by"])
                self.assertIs(row["ignored"], False)
                self.assertIs(row["actions_changed"], True)

    def test_unread_effect_is_ignored(self):
        _, trace, _ = run(0, a1(0, "executor"))
        row = effect(trace, event(trace), self.clean["user_task_0"])
        self.assertEqual(row["read_by"], "E1")
        self.assertEqual(row["acted_on"], "no")
        self.assertIs(row["ignored"], True)
        self.assertEqual(row["outcome"], "held")

    def test_researcher_payload_reaches_later_steps_only_if_passed_on(self):
        _, trace, _ = run(0, a1(0, "researcher"))
        row = effect(trace, event(trace), self.clean["user_task_0"])
        self.assertEqual(row["read_by"], "")  # the replayed R1 kept the payload out of its facts
        self.assertIs(row["ignored"], True)


if __name__ == "__main__":
    unittest.main()
