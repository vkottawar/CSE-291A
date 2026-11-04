# Lineage

CSE 291A (Fall 2026), Option 5: Failure Propagation and Recovery in Multi-Agent Systems.

When one agent's result turns out wrong or malicious, Lineage aims to fix only what was built on it, instead of restarting the whole workflow, without repeating or hiding irreversible actions such as payments.

## Setup

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then put your TritonAI key in .env
```

## Run one task

```bash
.venv/bin/python -m lineage --task user_task_0
```

This runs one [AgentDojo](https://github.com/ethz-spylab/agentdojo) banking task (benchmark version `v1.2.2`) through the 7-agent workflow, prints each step, reports whether AgentDojo's check passed (`utility`), and saves the full run record to `runs/`.

Task ids are `user_task_0` to `user_task_15`. Options: `--model` and `--base-url` override `.env`; `--seed` sets the seed (default 0).

## How it works

A research planner decides which of three researchers (files, transaction history, account) are needed and asks each one a question; they answer with read-only tools. A planner splits the work into up to two sub-plans, each listing the facts it uses. Two executors carry out the sub-plans; they are the only agents that can change the world, and each sees only its own facts. A reporter answers the user. `--layout researchers-first` runs the older layout, where all three researchers run without questions.

The run record lists, for every step, the facts it read, its output, and its tool calls numbered per step. Recovery will follow these dependencies.

## Layout

| Path | Contents |
| --- | --- |
| `lineage/roles.py` | Each role's instructions, allowed tools and output format |
| `lineage/agent.py` | One agent's tool loop, recorded into the run record |
| `lineage/workflow.py` | The order: research planner, researchers, planner, executors, reporter |
| `lineage/trace.py` | The run record format |
| `lineage/effects.py` | For each fault: was it read, did later outputs change, did the outcome flip |
| `lineage/llm.py` | Client for any OpenAI-compatible server; reads `.env` |
| `lineage/__main__.py` | Command line |

## Inject a wrong fact

Fault specs live in `faults/wrong_fact.json`: 8 reliable tasks (0, 3, 4, 5, 6, 11, 13, 15) × 3 stages. For each task the same fact gets the same wrong value at every stage, so only the place it enters differs:

- `researcher`: one fact in a researcher's output is changed before the planner sees it.
- `planner`: the planner hands one executor a wrong value for a fact its sub-plan uses.
- `executor`: one argument of one world-changing tool call is changed before it runs.

```bash
.venv/bin/python -m lineage --fault faults/wrong_fact.json --fault-id t0-researcher   # one run
.venv/bin/python -m lineage.fault_batch --tasks 0 --repeats 1                         # one task, all 3 stages
.venv/bin/python -m lineage.fault_batch                                               # all 24 specs × 3 repeats
```

`fault_batch` writes each run record, the banking world after the run (`*.world.json`, the starting point for restart baselines) and `runs.csv` to `runs/faults_<time>/`. Every run record has a `faults` list saying what was changed, where, and the original and injected values. `applied: false` means the fault never found its target (not placed), which is different from a fault that was placed but ignored. The spec format is documented at the top of `lineage/faults.py`.

AgentDojo's checks for tasks 5 and 6 pass whatever amount is paid (5 accepts the March payment already in history; 6 looks at the existing Spotify standing order), so use the `actions` column, not `passed`, to see whether those outcomes changed.

Tests (no API key needed; a fake model replays the committed clean runs):

```bash
.venv/bin/python -m unittest tests.test_faults -v
```

## Measure what a fault did

```bash
.venv/bin/python -m lineage.effects --faulted runs/faults_<time>_planner-first
.venv/bin/python -m unittest tests.test_effects -v   # no API key needed
```

This compares each faulted run with the clean repeats in `results/clean-baseline` and writes `effects.csv` (one row per fault per run) and `effects_summary.json` to the faulted folder. The columns are explained at the top of `lineage/effects.py`.

Read `changed` against the noise floor it prints: the same check run on clean runs, where every hit is a false alarm. `flipped` only counts runs in the `held` or `flipped` outcome, so crashes, faults that were not placed and tasks 5 and 6 stay out of it; use `actions_changed` for tasks 5 and 6.
