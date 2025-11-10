from __future__ import annotations
import re
from typing import Optional, Tuple

_CHOICES = {"A","B","C"}
_KEEP = {"KEEP","SWITCH"}

def norm(s: str) -> str:
    return (s or "").strip()

def extract_choice(text: str) -> Tuple[Optional[str], bool]:
    if text is None: return None, False
    raw = norm(text)
    exact = raw in _CHOICES
    m = re.search(r'\b([ABC])\b', raw, flags=re.I)
    return (m.group(1).upper() if m else None, exact)

def extract_keep_switch(text: str) -> Tuple[Optional[str], bool]:
    if text is None: return None, False
    raw = norm(text)
    exact = raw.upper() in _KEEP
    m = re.search(r'\b(KEEP|SWITCH)\b', raw, flags=re.I)
    return (m.group(1).upper() if m else None, exact)
