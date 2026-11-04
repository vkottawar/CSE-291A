"""Checks for lineage.restart. No API key needed.

  .venv/bin/python -m unittest tests.test_restart -v

The fake model from test_faults replays a clean run, so a restart always does the clean actions.
Executor faults are used because they change the world even though the replay ignores what it is shown.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from lineage.effects import load_clean
from lineage.faults import load_specs
from lineage.restart import clean_world, closest, load_world, restart_folder, same_field, state
from lineage.runner import run_task
from tests.test_faults import CLEAN, ROOT, ReplayLLM, spec

CLEAN_RUNS = load_clean(CLEAN)
INJECTIONS = {s.id: s for s in load_specs(ROOT / "faults" / "prompt_injection.json")}


def faulted_folder(folder, task_n, fault_id):
    """A folder like fault_batch writes, holding one faulted run."""
    llm = ReplayLLM(CLEAN / f"user_task_{task_n}_r1.json")
    fault = INJECTIONS.get(fault_id) or spec(fault_id)
    _, trace, env = run_task(f"user_task_{task_n}", llm, fault=fault, return_env=True)
    trace.save(Path(folder) / f"{fault_id}_r1.json")
    (Path(folder) / f"{fault_id}_r1.world.json").write_text(env.model_dump_json(indent=2))
    return trace


def restart(folder, task_n, **kwargs):
    llm = ReplayLLM(CLEAN / f"user_task_{task_n}_r1.json")
    summary = restart_folder(folder, llm, clean_dir=CLEAN, **kwargs)
    rows = (Path(folder) / "restart" / "restart.csv").read_text().splitlines()
    return summary, rows


def row(folder):
    with open(Path(folder) / "restart" / "restart.csv", newline="") as f:
        return next(csv.DictReader(f))


class TestCleanWorld(unittest.TestCase):
    def test_replaying_a_clean_run_rebuilds_its_payment(self):
        env = clean_world(CLEAN_RUNS["user_task_0"][0])
        self.assertIn(("UK12345678901234567890", 98.7, False), state(env)["transactions"])

    def test_a_clean_run_matches_itself(self):
        s = [state(clean_world(t)) for t in CLEAN_RUNS["user_task_0"]]
        self.assertEqual(sum(closest(s[0], s).values()), 0)


class TestAccountFields(unittest.TestCase):
    def test_city_wording_matches(self):
        self.assertTrue(same_field("city", "new york", "new york, ny 10001"))

    def test_wrong_street_number_differs(self):
        self.assertFalse(same_field("street", "dalton street 124", "dalton street 123"))

    def test_password_must_be_exact(self):
        self.assertFalse(same_field("password", "password", "password123"))

    def test_an_empty_address_differs(self):
        self.assertFalse(same_field("city", "", "new york"))


class TestRestart(unittest.TestCase):
    def test_wrong_payment_stays_after_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            faulted_folder(d, 0, "t0-executor")
            summary, _ = restart(d, 0)
            world = json.loads((Path(d) / "restart" / "t0-executor_r1.world.json").read_text())
        g = summary["groups"][0]
        self.assertEqual((g["clean_end_before"], g["clean_end_after"]), (0, 0))
        self.assertEqual((g["extra_payments"], g["missing_payments"]), (1, 0))
        self.assertEqual(g["passed_after"], 1)  # the check can't see the wrong payment next to the right one
        self.assertEqual(g["passed_but_not_clean"], 1)
        paid = [t["amount"] for t in world["bank_account"]["transactions"] if t["date"] == "2022-04-01"]
        self.assertEqual(sorted(paid), [88.83, 98.7])

    def test_wrong_update_is_overwritten_by_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            faulted_folder(d, 13, "t13-executor")
            summary, _ = restart(d, 13)
        g = summary["groups"][0]
        self.assertEqual((g["clean_end_before"], g["clean_end_after"]), (0, 1))

    def test_fault_not_placed_gets_no_restart(self):
        with tempfile.TemporaryDirectory() as d:
            t0 = faulted_folder(d, 0, "t0-executor")
            faulted_folder(d, 13, "t13-executor")
            t0.faults[0]["applied"] = False
            t0.save(Path(d) / "t0-executor_r1.json")
            summary, rows = restart(d, 13)
        self.assertEqual(summary["skipped_not_placed"], ["t0-executor_r1.json"])
        self.assertEqual(len(rows), 2)  # header and the one restart

    def test_ids_limit_what_restarts(self):
        with tempfile.TemporaryDirectory() as d:
            faulted_folder(d, 0, "t0-executor")
            faulted_folder(d, 13, "t13-executor")
            _, rows = restart(d, 13, ids={"t13-executor"})
        self.assertEqual(len(rows), 2)

    def test_restart_files_do_not_confuse_the_faulted_folder(self):
        with tempfile.TemporaryDirectory() as d:
            faulted_folder(d, 0, "t0-executor")
            restart(d, 0)
            top = sorted(p.name for p in Path(d).glob("*.json"))
        self.assertEqual(top, ["t0-executor_r1.json", "t0-executor_r1.world.json"])


class TestRescore(unittest.TestCase):
    def test_rescoring_needs_no_model_and_gives_the_same_row(self):
        with tempfile.TemporaryDirectory() as d:
            faulted_folder(d, 0, "t0-executor")
            restart(d, 0)
            first = row(d)
            restart_folder(d, None, clean_dir=CLEAN, rescore=True)
            self.assertEqual(row(d), first)

    def test_rescoring_without_a_saved_restart_stops(self):
        with tempfile.TemporaryDirectory() as d:
            faulted_folder(d, 0, "t0-executor")
            with self.assertRaises(SystemExit):
                restart_folder(d, None, clean_dir=CLEAN, rescore=True)


class TestPromptInjection(unittest.TestCase):
    def test_an_attacker_password_still_counts_after_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            faulted_folder(d, 0, "t0-executor-pw")
            world_path = Path(d) / "t0-executor-pw_r1.world.json"
            world = load_world(world_path)
            world.user_account.password = "new_password"
            world_path.write_text(world.model_dump_json())
            restart(d, 0)
            r = row(d)
        self.assertEqual(r["attack_after"], "True")
        self.assertEqual(r["clean_end"], "False")

    def test_attack_text_in_the_world_stays_but_is_not_a_change(self):
        with tempfile.TemporaryDirectory() as d:
            trace = faulted_folder(d, 3, "t3-researcher-pw")
            self.assertTrue(trace.faults[0]["applied"])
            restart(d, 3)
            saved = json.loads((Path(d) / "restart" / "t3-researcher-pw_r1.json").read_text())
            world = (Path(d) / "restart" / "t3-researcher-pw_r1.world.json").read_text()
        self.assertIn("new_password", world)
        self.assertEqual(saved["transaction_changes"]["modified"], [])

    def test_other_faults_leave_the_attack_columns_blank(self):
        with tempfile.TemporaryDirectory() as d:
            faulted_folder(d, 0, "t0-executor")
            restart(d, 0)
            r = row(d)
        self.assertEqual((r["attack_before"], r["attack_after"]), ("", ""))


if __name__ == "__main__":
    unittest.main()
