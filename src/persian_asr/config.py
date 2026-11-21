import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw=raw)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def ensure_dirs(self) -> None:
        keys = [
            "data.raw_dir",
            "data.processed_dir",
            "data.splits_dir",
            "output.checkpoint_dir",
            "output.log_dir",
            "output.metrics_dir",
        ]
        for key in keys:
            path = self.get(key)
            if path:
                os.makedirs(path, exist_ok=True)