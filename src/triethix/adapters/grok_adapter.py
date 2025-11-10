from __future__ import annotations
import os, time, json
from typing import List, Dict, Any, Optional
import requests
from .base import BaseAdapter

class _Cfg:
    def __init__(self) -> None:
        self.base_url = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
        self.timeout = float(os.environ.get("TRIETHIX_XAI_TIMEOUT", "90"))
        self.retries = int(os.environ.get("TRIETHIX_XAI_RETRIES", "3"))
        self.backoff = float(os.environ.get("TRIETHIX_XAI_BACKOFF", "0.8"))

class GrokAdapter(BaseAdapter):
    def __init__(self, model: str):
        super().__init__(model)
        key = os.environ.get("XAI_API_KEY")
        if not key:
            raise RuntimeError("XAI_API_KEY not set")
        self.key = key
        self.cfg = _Cfg()
        self.url = f"{self.cfg.base_url}/chat/completions"
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "User-Agent": "TriEthix/0.1"
        }

    def _post_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(self.url, headers=self.headers, data=json.dumps(payload), timeout=self.cfg.timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}
            raise RuntimeError(f"Grok HTTP {resp.status_code}: {body}") from e
        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError("Grok returned non-JSON response") from e

    def _post_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        delay = self.cfg.backoff
        for attempt in range(1, self.cfg.retries + 1):
            try:
                return self._post_once(payload)
            except (requests.ReadTimeout, requests.ConnectionError) as e:
                if attempt >= self.cfg.retries:
                    raise RuntimeError(f"Grok timeout after {attempt} attempts") from e
                time.sleep(delay); delay *= 1.6
            except RuntimeError as e:
                msg = str(e)
                should_retry = any(code in msg for code in ["HTTP 429","HTTP 500","HTTP 502","HTTP 503","HTTP 504"])
                if not should_retry or attempt >= self.cfg.retries:
                    raise
                time.sleep(delay); delay *= 1.6
        raise RuntimeError("Grok: exhausted retries")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}

        if "max_completion_tokens" in kwargs:
            payload["max_tokens"] = int(kwargs["max_completion_tokens"])
        elif "max_tokens" in kwargs:
            payload["max_tokens"] = int(kwargs["max_tokens"])

        if "stop" in kwargs and kwargs["stop"]:
            payload["stop"] = kwargs["stop"]

        data = self._post_with_retry(payload)

        try:
            content: Optional[str] = data["choices"][0]["message"]["content"]
        except Exception:
            content = None
        return (content or "").strip()
