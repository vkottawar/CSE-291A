"""LLM client. Works with any server that speaks the OpenAI chat API (TritonAI, llama.cpp server, ...).

Configure it in the .env file at the project root (or with environment variables, which win):
  LLM_API_KEY   your API key
  LLM_BASE_URL  the server URL, e.g. "https://<tritonai-host>/v1" or "http://localhost:8080/v1"
  LLM_MODEL     the model name the server expects
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if value:
            os.environ.setdefault(key.strip(), value)


_load_env_file(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "")
MODEL = os.environ.get("LLM_MODEL", "")


@dataclass
class ToolCallRequest:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM:
    def __init__(self, model=None, base_url=None, api_key=None, seed: int = 0, temperature: float = 0.0):
        self.model = model or MODEL
        base_url = base_url or BASE_URL
        api_key = api_key or API_KEY
        missing = [name for name, v in [("LLM_API_KEY", api_key), ("LLM_BASE_URL", base_url), ("LLM_MODEL", self.model)] if not v]
        if missing:
            raise SystemExit(f"LLM not configured: set {', '.join(missing)} (see lineage/llm.py)")

        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self.seed = seed
        self.temperature = temperature

    def chat(self, step_id, messages, tools) -> Reply:
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature, seed=self.seed)
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            ToolCallRequest(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}"))
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage
        return Reply(
            content=msg.content,
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
