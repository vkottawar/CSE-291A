"""Checks for lineage.effects. No API key needed.

  .venv/bin/python -m unittest tests.test_effects -v

Faulted runs come from the real injector with the replay model from test_faults. The replay doesn't
react to the fault, so a researcher or planner fault comes out ignored; a test that needs a reaction
edits the run record to what a real model would have done.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from lineage.effects import contains, effect, false_alarms, load_clean, load_trace, main, numbers
from lineage.trace import ToolCallRecord
from tests.test_faults import CLEAN, run, spec

clean = load_clean(CLEAN)


def row_for(trace):
    return effect(trace, trace.faults[0], clean[trace.task_id])


def faulted(task_n, fault_id):
    return run(task_n, spec(fault_id))[1]


def executor_step(trace):
    return next(s for s in trace.steps if s.role == "executor")


class TestValueMatching(unittest.TestCase):
    def test_numbers_compare_by_value(self):
        self.assertTrue(contains("paid 9.00 today", 9))
        self.assertTrue(contains("paid 9.00 today", "9.0"))
        self.assertTrue(contains("a total of 1,100.00.", 1100))

    def test_part_of_a_longer_number_is_no_match(self):
        self.assertFalse(contains("paid 19.00", 9))
        self.assertFalse(contains("paid 9.50", 9))

    def test_iban_digits_and_dates_are_not_numbers(self):
        self.assertEqual(numbers("to UK12345678901234567890 on 2022-04-01"), set())

    def test_strings_match_inside_text(self):
        self.assertTrue(contains("moved to dalton street 124", "Dalton Street 124"))
        self.assertFalse(contains("moved to Dalton Street 123", "Dalton Street 124"))

    def test_missing_value_matches_nothing(self):
        self.assertFalse(contains("anything", None))


class TestRead(unittest.TestCase):
    def test_fault_nobody_acts_on_is_ignored_not_robustness(self):
        row = row_for(faulted(0, "t0-researcher"))
        self.assertEqual(row["read_by"], "P E1 REP")  # listed as read, and still nothing happened
        self.assertEqual(row["used_in"], "")
        self.assertEqual(row["acted_on"], "no")
        self.assertIs(row["ignored"], True)
        self.assertEqual(row["outcome"], "held")

    def test_value_reaching_a_tool_call_is_acted_on(self):
        trace = faulted(0, "t0-researcher")
        executor_step(trace).tool_calls[0].args["amount"] = 88.83
        trace.utility = False
        row = row_for(trace)
        self.assertEqual(row["used_in"], "E1")
        self.assertEqual(row["acted_on"], "exact")
        self.assertIs(row["ignored"], False)
        self.assertIs(row["actions_changed"], True)
        self.assertEqual(row["outcome"], "flipped")

    def test_rejected_call_is_not_acted_on(self):
        trace = faulted(0, "t0-researcher")
        call = executor_step(trace).tool_calls[0]
        call.args["amount"], call.error = 88.83, "rejected"
        self.assertEqual(row_for(trace)["acted_on"], "no")

    def test_value_computed_from_the_fault_is_derived(self):
        # Task 3 refunds the friend's payment minus 6, so a wrong 9.00 becomes a payment of 3.00.
        trace = faulted(3, "t3-researcher")
        executor_step(trace).tool_calls[0].args["amount"] = 3.0
        row = row_for(trace)
        self.assertNotIn("E1", row["used_in"].split())
        self.assertEqual(row["acted_on"], "derived")
        self.assertIs(row["ignored"], False)

    def test_extra_name_on_a_call_is_not_derived(self):
        # A real t13 run added first_name to the call. Nothing to do with the fault.
        trace = faulted(13, "t13-researcher")
        executor_step(trace).tool_calls[0].args["first_name"] = "Emma"
        row = row_for(trace)
        self.assertEqual(row["acted_on"], "no")
        self.assertIs(row["actions_changed"], True)  # the strict comparison still sees it
        self.assertIs(row["changed"], False)

    def test_retry_with_the_original_value_is_corrected(self):
        trace = faulted(0, "t0-executor")
        step = executor_step(trace)
        step.tool_calls.append(ToolCallRecord("send_money", {"amount": 98.7, "recipient": "x"}, 2, "sent", None))
        self.assertIs(row_for(trace)["corrected"], True)

    def test_no_retry_is_not_corrected(self):
        self.assertIs(row_for(faulted(0, "t0-executor"))["corrected"], False)
        self.assertEqual(row_for(faulted(0, "t0-researcher"))["corrected"], "")

    def test_planner_fault_reaches_the_reporter_only_through_its_executor(self):
        trace = faulted(0, "t0-planner")
        reporter = next(s for s in trace.steps if s.role == "reporter")
        self.assertEqual(row_for(trace)["read_by"], "E1 REP")
        reporter.reads.remove("E1.done")  # it still lists R1.bill_amount, but its copy was never wrong
        self.assertEqual(row_for(trace)["read_by"], "E1")


class TestChanged(unittest.TestCase):
    def test_executor_fault_changing_only_its_own_call_changes_nothing_else(self):
        row = row_for(faulted(0, "t0-executor"))
        self.assertEqual(row["acted_on"], "exact")
        self.assertIs(row["actions_changed"], True)
        self.assertIs(row["changed"], False)
        self.assertIs(row["ignored"], False)

    def test_new_amount_in_a_later_step_is_a_change(self):
        trace = faulted(0, "t0-researcher")
        trace.steps[-1].output = "Paid 77.11 in total."
        self.assertEqual(row_for(trace)["changed_in"], "REP")

    def test_prose_restating_a_known_value_is_not_a_change(self):
        trace = faulted(0, "t0-researcher")
        trace.steps[-1].output = "1. Paid your December 2023 bill. Your balance is 1810.00."
        self.assertEqual(row_for(trace)["changed_in"], "")

    def test_action_seen_in_any_clean_repeat_is_normal(self):
        # Task 15 has two clean action sets, so a run matching either one is not a change.
        for r in (1, 2, 3):
            self.assertIs(row_for(run(15, spec("t15-researcher"), repeat=r)[1])["actions_changed"], False)

    def test_value_clean_runs_also_produce_is_not_used(self):
        trace = faulted(0, "t0-researcher")
        trace.faults[0]["injected"] = 1810.0  # the balance, already in the clean reports
        self.assertEqual(row_for(trace)["used_in"], "")


class TestOutcome(unittest.TestCase):
    def test_unplaced_fault_has_no_effects(self):
        trace = faulted(0, "t0-researcher")
        trace.faults[0] = {"id": "x", "type": "wrong_fact", "stage": "researcher", "applied": False, "reason": "no fact"}
        row = row_for(trace)
        self.assertEqual(row["outcome"], "not_placed")
        self.assertEqual(row["ignored"], "")  # not placed is different from placed but ignored

    def test_crash_is_kept_and_not_called_a_flip(self):
        trace = faulted(0, "t0-researcher")
        trace.error, trace.utility = "APIError: boom", False
        row = row_for(trace)
        self.assertEqual(row["outcome"], "crashed")
        self.assertEqual(row["error"], "APIError: boom")

    def test_blind_check_is_not_called_held(self):
        trace = faulted(5, "t5-executor")
        self.assertEqual(row_for(trace)["outcome"], "check_blind")

    def test_task_failing_clean_is_not_called_a_flip(self):
        trace = faulted(0, "t0-researcher")
        trace.task_id, trace.utility = "user_task_12", False  # 0/3 clean
        self.assertEqual(row_for(trace)["outcome"], "clean_unreliable")


class TestNoiseFloor(unittest.TestCase):
    def test_identical_repeats_raise_no_alarm(self):
        traces = [load_trace(CLEAN / "user_task_0_r1.json")] * 3
        noise = false_alarms({"user_task_0": traces})
        self.assertEqual((noise["runs"], noise["any_step_changed"], noise["actions_changed"]), (3, 0, 0))


    def test_noise_on_the_real_clean_runs_stays_low(self):
        # Comparing prose step by step fired on 19 of these 36 runs, which would make `changed` useless.
        self.assertLessEqual(false_alarms(clean)["any_step_changed"], 7)


class TestCommandLine(unittest.TestCase):
    def test_writes_a_row_per_faulted_run_and_skips_unfaulted_files(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            faulted(0, "t0-researcher").save(folder / "t0-researcher_r1.json")
            faulted(0, "t0-executor").save(folder / "t0-executor_r2.json")
            run(0)[1].save(folder / "no_fault_r1.json")
            (folder / "t0-executor_r2.world.json").write_text("{}")
            (folder / "summary.json").write_text("{}")
            main(["--faulted", d, "--clean", str(CLEAN)])
            with open(folder / "effects.csv") as f:
                rows = {r["fault_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(set(rows), {"t0-researcher", "t0-executor"})
        self.assertEqual(rows["t0-executor"]["repeat"], "2")
        self.assertEqual(rows["t0-researcher"]["ignored"], "True")


if __name__ == "__main__":
    unittest.main()
