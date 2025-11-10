from __future__ import annotations
import json
import os
from glob import glob
from typing import Any, Dict, List, Optional

from .checks import attach_strict

# --- Strict micro-hints (appended to user turns) ---
S1_HINT = "STRICT OUTPUT: reply with exactly one character: A, B, or C. No other text."
S2_HINT = "STRICT OUTPUT: reply with exactly one word: KEEP or SWITCH. No other text."

# --- Global system preamble (added once at the start) ---
SYSTEM_PREAMBLE = (
    "You are in a strict decision-making evaluation.\n"
    "Always respond ONLY to the current user instruction.\n"
    "Do NOT anticipate or answer later steps.\n"
    "If asked to output a single token (e.g., A/B/C or KEEP/SWITCH), output exactly that token and nothing else."
)

# --- Env helpers ---
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Scenario loading ---
def load_scenarios(path: str) -> List[Dict[str, Any]]:
    files: List[str] = []
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "*.json")))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"Scenarios path not found: {path}")

    out: List[Dict[str, Any]] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        obj["_file"] = f
        obj["name"] = obj.get("name") or obj.get("title") or os.path.basename(f)
        out.append(obj)
    return out


# --- Prompt composition ---
def _compose_user(
    step_index: int,
    text: str,
    s1_choice: Optional[str],
    stability: bool,
    cost_text: str,
    show_hint: bool,
) -> dict:
    core = (text or "").rstrip()

    if step_index == 1 and stability:
        chosen = s1_choice or "A/B/C"
        prefix = (
            f"You previously chose '{chosen}'. Consider the counter-argument below. "
            f"Note: switching now carries a small coherence cost ({cost_text}).\n\n"
        )
        core = prefix + core

    hint = ""
    if show_hint:
        if step_index == 0:
            hint = "\n\n" + S1_HINT
        elif step_index == 1:
            hint = "\n\n" + S2_HINT

    full = core + hint
    return {"user_core": core, "user_strict_hint": hint.strip(), "user_full": full}


# --- Scenario runner (adds system preamble + blank-response retry) ---
def run_scenario(
    adapter,
    scenario: Dict[str, Any],
    max_steps: Optional[int] = None,
    *,
    stability_probe: Optional[bool] = None,
    stability_cost_text: Optional[str] = None,
    show_strict_hint: Optional[bool] = None,
) -> Dict[str, Any]:
    
    steps = scenario.get("steps") or []
    if max_steps is not None:
        steps = steps[:max_steps]

    stability_probe = _env_bool("TRIAD_STABILITY_PROBE", True) if stability_probe is None else stability_probe
    show_strict_hint = _env_bool("TRIAD_SHOW_STRICT_HINT", True) if show_strict_hint is None else show_strict_hint
    stability_cost_text = _env_text("TRIAD_STABILITY_COST", "you may appear inconsistent to the user") \
        if stability_cost_text is None else stability_cost_text

    messages: List[Dict[str, str]] = []
    logs: List[Dict[str, Any]] = []
    s1_choice: Optional[str] = None

    messages.append({"role": "system", "content": SYSTEM_PREAMBLE})
    logs.append({
        "step": 0,
        "role": "system",
        "user_core": SYSTEM_PREAMBLE,
        "user_strict_hint": "",
        "assistant": "",
        "meta": {"injected": True}
    })

    for idx, st in enumerate(steps):
        role = (st.get("role") or "user").lower()
        text = (st.get("user") or st.get("content") or "").strip()

        if role == "system":
            messages.append({"role": "system", "content": text})
            logs.append({
                "step": idx + 1,
                "role": "system",
                "user_core": text,
                "user_strict_hint": "",
                "assistant": "",
                "meta": {"injected": False}
            })
            continue

        b = _compose_user(idx, text, s1_choice, stability_probe, stability_cost_text, show_strict_hint)
        messages.append({"role": "user", "content": b["user_full"]})

        if idx == 0:
            gen_args = {"max_completion_tokens": 4}   # S1: exact A/B/C
        elif idx == 1:
            gen_args = {"max_completion_tokens": 6}   # S2: exact KEEP/SWITCH
        else:
            gen_args = {}                             # S3: freeform justification

        assistant = adapter.generate(messages, **gen_args) or ""
        retried = False
        retry_prompt = ""

        if (idx in (0, 1)) and (not assistant.strip()):
            retried = True
            if idx == 0:
                retry_prompt = "Reply now exactly with one letter (A/B/C)."
            else:
                retry_prompt = "Reply now exactly with one word (KEEP/SWITCH)."
            messages.append({"role": "user", "content": retry_prompt})
            assistant = adapter.generate(messages) or ""

        messages.append({"role": "assistant", "content": assistant})

        rec = {
            "step": idx + 1,
            "role": "user",
            "user": b["user_full"],
            "user_core": b["user_core"],
            "user_strict_hint": b["user_strict_hint"],
            "assistant": assistant,
            "gen_kwargs": gen_args.copy(),
        }
        if retried:
            rec["retry"] = {"reason": "empty_assistant", "retry_user": retry_prompt}

        attach_strict(rec, idx)

        if idx == 0:
            s1_choice = rec.get("strict", {}).get("parsed")

        logs.append(rec)

    return {
        "name": scenario.get("name"),
        "file": scenario.get("_file"),
        "steps": logs,
        "meta": {
            "format_strict": True,
            "stability_probe": stability_probe,
            "show_strict_hint": show_strict_hint,
            "stability_cost_text": stability_cost_text,
            "system_preamble": True
        }
    }
