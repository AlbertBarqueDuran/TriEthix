from __future__ import annotations
import os
import time
import json
import typing as T
import requests


class DeepSeekAdapter:
    def __init__(
        self,
        model: str,
        *,
        timeout: T.Optional[float] = None,
        max_retries: int = 5,
        backoff: float = 1.6,
        session: T.Optional[requests.Session] = None,
    ) -> None:
        self.model = model
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        self.base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.timeout = float(os.environ.get("DEEPSEEK_TIMEOUT_S", timeout or 60))
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = session or requests.Session()

        self._chat_url = self._join(self.base_url, "/chat/completions")
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._known = {"deepseek-chat", "deepseek-reasoner"}
        if self.model not in self._known:
            pass

    @staticmethod
    def _join(base: str, path: str) -> str:
        return f"{base.rstrip('/')}{path}"

    @staticmethod
    def _sanitize_messages(msgs: T.List[dict]) -> T.List[dict]:
        clean = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            m2 = dict(m)
            if "reasoning_content" in m2:
                m2.pop("reasoning_content", None)
            if "role" in m2 and "content" in m2:
                clean.append(m2)
        return clean

    def _post_once(self, payload: dict) -> dict:
        resp = self.session.post(
            self._chat_url,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )
        if not resp.ok:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {body}")
        try:
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"DeepSeek: invalid JSON response: {e}")

    def _post_with_retry(self, payload: dict) -> dict:
        attempt = 0
        last_err = None
        while attempt < self.max_retries:
            try:
                return self._post_once(payload)
            except Exception as e:
                last_err = e
                time.sleep(self.backoff ** attempt * 0.5)
                attempt += 1
        raise RuntimeError(f"DeepSeek request failed after {self.max_retries} attempts: {last_err}")

    def generate(
        self,
        messages: T.List[dict],
        **gen_args,
    ) -> str:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of {role, content} dicts")

        clean_msgs = self._sanitize_messages(messages)

        payload: dict = {
            "model": self.model,
            "messages": clean_msgs,
            "stream": False,
        }

        if "max_tokens" in gen_args and gen_args["max_tokens"]:
            payload["max_tokens"] = int(gen_args["max_tokens"])
        if "stop" in gen_args and gen_args["stop"]:
            payload["stop"] = gen_args["stop"]

        data = self._post_with_retry(payload)

        try:
            choice = data["choices"][0]
            msg = choice.get("message", {}) or {}
            content = msg.get("content")
            if content is None:
                content = msg.get("reasoning_content") or ""
            return content or ""
        except Exception as e:
            raise RuntimeError(f"DeepSeek parse error: {e}\nRaw: {json.dumps(data)[:1000]}")
