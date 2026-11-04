"""run_agent: one agent = one role's prompt + tool loop, recorded into the trace."""

import json
import time

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str

from .roles import ROLES
from .trace import StepRecord, ToolCallRecord, Trace


def tool_schemas(runtime, names):
    return [
        {
            "type": "function",
            "function": {
                "name": runtime.functions[n].name,
                "description": runtime.functions[n].description,
                "parameters": runtime.functions[n].parameters.model_json_schema(),
            },
        }
        for n in names
    ]


def run_agent(step_id, role_name, inputs, reads, env, runtime, llm, trace: Trace, max_turns=10, arg_hook=None, before_chat=None):
    """arg_hook mutates tool arguments; before_chat(step_id, turn) runs before a model request."""
    role = ROLES[role_name]
    record = StepRecord(step_id=step_id, role=role_name, reads=reads)
    trace.steps.append(record)
    start = time.time()

    messages = [
        {"role": "system", "content": role.system},
        {"role": "user", "content": json.dumps(inputs, default=str)},
    ]
    record.inputs = json.loads(messages[1]["content"])
    tools = tool_schemas(runtime, role.tools)

    for _ in range(max_turns):
        try:
            if before_chat:
                before_chat(step_id, record.turns + 1)
            record.turns += 1
            reply = llm.chat(step_id, messages, tools)
        except Exception as e:
            record.status = "timeout" if isinstance(e, TimeoutError) else "error"
            record.seconds = time.time() - start
            raise
        record.prompt_tokens += reply.prompt_tokens
        record.completion_tokens += reply.completion_tokens

        if reply.tool_calls and arg_hook:
            # Before the call is added to the history, so the agent itself looks like it asked for it.
            for n, c in enumerate(reply.tool_calls):
                c.args = arg_hook(step_id, c.name, c.args, len(record.tool_calls) + n + 1)
        if reply.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                if call.name in role.tools:
                    result, error = runtime.run_function(env, call.name, call.args)
                    result = tool_result_to_str(result)
                else:
                    result, error = "", f"Tool {call.name} is not available to the {role_name}."
                record.tool_calls.append(
                    ToolCallRecord(call.name, call.args, len(record.tool_calls) + 1, result, error)
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": error or result})
            continue

        if not role.returns_json:
            record.output = reply.content or ""
            break
        try:
            record.output = _parse_json(reply.content)
            break
        except ValueError:
            messages.append({"role": "assistant", "content": reply.content})
            messages.append({"role": "user", "content": "That was not valid JSON. Reply with ONLY the JSON object."})
            record.status = "bad_output"
    else:
        record.status = "max_turns"

    if record.output is not None and record.status == "bad_output":
        record.status = "ok"  # recovered on a retry; the extra turn is still counted
    record.seconds = time.time() - start
    return record.output


def _parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(str(e)) from e
