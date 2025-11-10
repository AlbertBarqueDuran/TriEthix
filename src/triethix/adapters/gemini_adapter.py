from __future__ import annotations
import os, time
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from .base import BaseAdapter

try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except Exception:
    HarmCategory, HarmBlockThreshold = None, None


class _Cfg:
    def __init__(self) -> None:
        self.retries = int(os.environ.get("TRIETHIX_GEMINI_RETRIES", "3"))
        self.backoff = float(os.environ.get("TRIETHIX_GEMINI_BACKOFF", "0.8"))
        self.safety_mode = os.environ.get("TRIETHIX_GEMINI_SAFETY", "default").lower()


def _flatten_messages(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System:\n{content}")
        elif role == "user":
            parts.append(f"User:\n{content}")
        else:
            parts.append(f"Assistant:\n{content}")
    return "\n\n".join(parts).strip()


def _safe_cat(name: str):
    if HarmCategory is None:
        return None
    return getattr(HarmCategory, name, None)


def _safe_thr(name: str):
    if HarmBlockThreshold is None:
        return None
    return getattr(HarmBlockThreshold, name, None)


def _default_safety_cfg() -> Optional[Dict[Any, Any]]:
    cand_categories = [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_SEXUAL",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ]
    thr = _safe_thr("BLOCK_MEDIUM_AND_ABOVE") or _safe_thr("BLOCK_ONLY_HIGH") or _safe_thr("BLOCK_NONE")
    if thr is None:
        return None

    safety: Dict[Any, Any] = {}
    for nm in cand_categories:
        cat = _safe_cat(nm)
        if cat is not None:
            safety[cat] = thr

    return safety or None


def _open_safety_cfg() -> Optional[Dict[Any, Any]]:
    thr_none = _safe_thr("BLOCK_NONE")
    if thr_none is None:
        return None
    cand_categories = [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_SEXUAL",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ]
    safety: Dict[Any, Any] = {}
    for nm in cand_categories:
        cat = _safe_cat(nm)
        if cat is not None:
            safety[cat] = thr_none
    return safety or None


class GeminiAdapter(BaseAdapter):

    def __init__(self, model: str):
        super().__init__(model)
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        self.model_name = model
        self.cfg = _Cfg()
        self.default_safety = None if self.cfg.safety_mode == "none" else _default_safety_cfg()

    def _make_model(self, safety_settings: Optional[Dict[Any, Any]]):
        if safety_settings:
            return genai.GenerativeModel(self.model_name, safety_settings=safety_settings)
        return genai.GenerativeModel(self.model_name)

    def _call(self, prompt: str, gen_cfg: Optional[Dict[str, Any]], safety_first: bool = True) -> str:
        retries = self.cfg.retries
        backoff = self.cfg.backoff

        safety = self.default_safety if safety_first else None
        model = self._make_model(safety)
        for attempt in range(1, retries + 1):
            try:
                resp = model.generate_content(prompt, generation_config=(gen_cfg or None))
                text = getattr(resp, "text", None)
                if text:
                    return text.strip()
                try:
                    cands = getattr(resp, "candidates", None) or []
                    for cand in cands:
                        parts = getattr(cand, "content", None)
                        if not parts:
                            continue
                        txts = []
                        for p in getattr(parts, "parts", []) or []:
                            t = getattr(p, "text", None)
                            if t:
                                txts.append(t)
                        if txts:
                            return "\n".join(txts).strip()
                except Exception:
                    pass
            except Exception:
                pass

            if attempt < retries:
                time.sleep(backoff)
                backoff *= 1.6

        open_safety = _open_safety_cfg()
        if open_safety:
            model2 = self._make_model(open_safety)
            try:
                resp2 = model2.generate_content(prompt, generation_config=(gen_cfg or None))
                text2 = getattr(resp2, "text", None)
                if text2:
                    return text2.strip()
            except Exception:
                pass

        return ""

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        prompt = _flatten_messages(messages)

        gen_cfg: Dict[str, Any] = {}
        if "max_completion_tokens" in kwargs:
            gen_cfg["max_output_tokens"] = int(kwargs["max_completion_tokens"])
        elif "max_tokens" in kwargs:
            gen_cfg["max_output_tokens"] = int(kwargs["max_tokens"])

        stops = kwargs.get("stop")
        if stops:
            gen_cfg["stop_sequences"] = list(stops)

        return self._call(prompt, gen_cfg, safety_first=True)
