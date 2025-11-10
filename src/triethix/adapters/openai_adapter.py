from __future__ import annotations
import os, time
from typing import List, Dict, Any
from openai import OpenAI, BadRequestError
from .base import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model: str):
        super().__init__(model)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        args: Dict[str, Any] = {"model": self.model, "messages": messages}
        if "max_completion_tokens" in kwargs:
            args["max_completion_tokens"] = kwargs["max_completion_tokens"]
        elif "max_tokens" in kwargs:
            args["max_completion_tokens"] = kwargs["max_tokens"]

        for _ in range(3):
            try:
                comp = self.client.chat.completions.create(**args)
                return (comp.choices[0].message.content or "").strip()
            except BadRequestError as e:
                msg = str(e)
                if ("max_tokens" in msg and "unsupported" in msg) or \
                   ("max_completion_tokens" in msg and "unsupported" in msg):
                    args.pop("max_completion_tokens", None)
                    time.sleep(0.2)
                    continue
                raise

        comp = self.client.chat.completions.create(model=self.model, messages=messages)
        return (comp.choices[0].message.content or "").strip()
