# A1 prompt injection (not merged yet)

Work in progress, saved here before the teammate's F1 fault code (`lineage/faults.py`) landed on main.
It still needs reworking to fit that `FaultInjector` / JSON-spec design before it can be merged.

| File | What it is |
| --- | --- |
| `faults_a1.py` | The A1 module: AgentDojo attack payloads at researcher / planner / executor, plus read/repeated detection |
| `a1_changes.patch` | The edits A1 made to `workflow.py`, `runner.py`, `trace.py`, `__main__.py`, `batch.py` and `README.md` (against commit 8e4f8a8) |
| `fake_run.py` | End-to-end check with a scripted stand-in model (no API key needed); written for the pre-merge code |

`fake_run.py` and the patch target the code as it was at 8e4f8a8, so they won't run against the current main as is.
