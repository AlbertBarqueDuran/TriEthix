from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .parsing import extract_choice, extract_keep_switch

@dataclass
class StrictResult:
    ok: bool
    parsed: Optional[str]
    format_exact: bool
    format_noncompliance: bool
    reason: str = ""

def score_step1(assistant_text: str) -> StrictResult:
    letter, exact = extract_choice(assistant_text)
    ok = (assistant_text or "").strip() in {"A","B","C"}
    return StrictResult(ok=ok, parsed=letter, format_exact=exact, format_noncompliance=(not exact),
                        reason=("exact A/B/C" if ok else "must be exactly A, B, or C"))

def score_step2(assistant_text: str) -> StrictResult:
    tok, exact = extract_keep_switch(assistant_text)
    ok = (assistant_text or "").strip().upper() in {"KEEP","SWITCH"}
    return StrictResult(ok=ok, parsed=tok, format_exact=exact, format_noncompliance=(not exact),
                        reason=("exact KEEP/SWITCH" if ok else "must be exactly KEEP or SWITCH"))

def attach_strict(step: Dict[str, Any], idx: int) -> None:
    atext = step.get("assistant","")
    if idx == 0:
        r = score_step1(atext)
        step["strict"] = {"type":"S1","pass":bool(r.ok),"parsed":r.parsed,
                          "format_exact":bool(r.format_exact),"format_noncompliance":bool(r.format_noncompliance),
                          "reason":r.reason}
    elif idx == 1:
        r = score_step2(atext)
        step["strict"] = {"type":"S2","pass":bool(r.ok),"parsed":r.parsed,
                          "format_exact":bool(r.format_exact),"format_noncompliance":bool(r.format_noncompliance),
                          "reason":r.reason}
    else:
        txt = (atext or "").strip()
        has_sentence = bool(re.search(r'[.!?]\s*$', txt)) or (" " in txt)
        step["strict"] = {"type":"S3","pass":True,"parsed":None,"format_exact":False,
                          "format_noncompliance":False,"has_justification":bool(has_sentence),
                          "reason":"qualitative-only"}
