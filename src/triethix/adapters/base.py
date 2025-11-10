from __future__ import annotations
from typing import List, Dict, Any

class BaseAdapter:
    def __init__(self, model: str):
        self.model = model
    def generate(self, messages: List[Dict[str,str]], **kwargs) -> str:
        raise NotImplementedError
