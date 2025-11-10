from __future__ import annotations
import os, time
from typing import List, Dict, Any
import anthropic
from .base import BaseAdapter

class _Cfg:
    def __init__(self) -> None:
        self.retries = int(os.environ.get("TRIETHIX_ANTHROPIC_RETRIES", "3"))
        self.backoff = float(os.environ.get("TRIETHIX_ANTHROPIC_BACKOFF", "0.8"))

class AnthropicAdapter(BaseAdapter):
    def __init__(self, model: str):
        super().__init__(model)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.cfg = _Cfg()

    def generate(self, messages: List[Dict[str,str]], **kwargs) -> str:
        system = None
        convo = []
        for m in messages:
            role = m.get("role")
            content = m.get("content","")
            if role == "system":
                system = (system + "\n" if system else "") + content
            elif role == "user":
                convo.append({"role": "user", "content": content})
            else:
                convo.append({"role": "assistant", "content": content})

        max_tokens = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 128
        stop = kwargs.get("stop") or None
        stop_sequences = list(stop) if stop else None

        delay = self.cfg.backoff
        for attempt in range(1, self.cfg.retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    system=system,
                    max_tokens=int(max_tokens),
                    stop_sequences=stop_sequences,
                    messages=convo,
                )
                texts = [c.text for c in resp.content if getattr(c, "type", None) == "text"]
                return ("\n".join(texts)).strip()
            except anthropic.RateLimitError as e:
                if attempt >= self.cfg.retries:
                    raise
                time.sleep(delay); delay *= 1.6
            except anthropic.InternalServerError as e:
                if attempt >= self.cfg.retries:
                    raise
                time.sleep(delay); delay *= 1.6

        return ""
