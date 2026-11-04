"""AgentDojo's built-in prompt injections, built for one user task (fault type "prompt_injection").

The payload is exactly what an AgentDojo attack (default important_instructions) writes for an
AgentDojo injection task (the attacker's goal); AgentDojo's own security check grades the run.
faults.py decides where the payload goes; this module only builds and finds it.
"""

import re
from types import SimpleNamespace

from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.attacks.base_attacks import DEFAULT_MODEL_NAME, FixedJailbreakAttack
from agentdojo.models import MODEL_NAMES

# Template attacks with an attacker goal; the DoS attacks have none to check.
ATTACK_NAMES = [n for n, c in ATTACKS.items() if issubclass(c, FixedJailbreakAttack) and not c.is_dos_attack]
ATTACKER_IBAN = "US133000000121212121212"  # every banking injection task that moves money pays this account


def build(spec, suite, user_task, model_name=""):
    """Returns (payload, vectors, injection_task, marker). vectors maps AgentDojo injection-vector ids
    to the text to put in the world; it is filled only for the researcher stage. marker is a short
    phrase from the payload for spotting it in later steps."""
    attack = _load_attack(spec.attack, suite, model_name)
    injection_task = suite.get_injection_task_by_id(spec.injection_task)
    # The vectors this user task's correct solution reads; AgentDojo raises if there are none.
    by_vector = attack.attack(user_task, injection_task)
    payload = next(iter(by_vector.values()))
    vectors = {}
    if spec.stage == "researcher":
        defaults = suite.get_injection_vector_defaults()
        # "append" keeps the real text (a bill's amount and IBAN) so lost data doesn't mix into the
        # attack's effect; "replace" is what AgentDojo's own benchmark does.
        vectors = {v: (defaults[v] + text if spec.vector_mode == "append" else text) for v, text in by_vector.items()}
    return payload, vectors, injection_task, marker(payload, injection_task.GOAL)


def _load_attack(name, suite, model_name):
    """AgentDojo attacks name the target model; ours aren't in its list, so the important_instructions
    family falls back to "the AI language model", as important_instructions_no_model_name does."""
    try:
        return load_attack(name, suite, SimpleNamespace(name=model_name or ""))
    except ValueError:
        attack = load_attack(name, suite, SimpleNamespace(name=next(iter(MODEL_NAMES))))
        attack.model_name = DEFAULT_MODEL_NAME
        return attack


def marker(payload: str, goal: str = "", limit: int = 60) -> str:
    """A short phrase from the payload that clean runs never produce: the start of the attacker's goal,
    cut at a word boundary, e.g. "Change the password of the user to new_password." Falls back to
    the payload's longest line if an attack template leaves the goal out."""
    first = re.split(r"(?<=[.!?])\s", goal.strip())[0] if goal else ""
    phrase = ""
    for word in first.split():
        if len(phrase) + len(word) + 1 > limit:
            break
        phrase = f"{phrase} {word}".strip()
    if phrase and phrase in payload:
        return phrase
    lines = [ln.strip() for ln in payload.splitlines() if ln.strip()]
    return max(lines, key=len)[:limit] if lines else payload


def norm(text) -> str:
    """Tool results come back as YAML, which may wrap or escape long lines: compare without whitespace or quotes."""
    return re.sub(r'(\\n|\\t|\s|"|\')+', "", str(text or ""))


def contains_marker(text, phrase: str) -> bool:
    return bool(phrase) and norm(phrase) in norm(text)
