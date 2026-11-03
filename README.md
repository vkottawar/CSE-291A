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
| `lineage/llm.py` | Client for any OpenAI-compatible server; reads `.env` |
| `lineage/__main__.py` | Command line |
