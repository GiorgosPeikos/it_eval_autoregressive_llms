from __future__ import annotations

from pathlib import Path

from it_eval_framework.utils.io import read_json, write_json


class RunState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.state = read_json(self.path) if self.path.exists() else {"steps": {}}

    def is_complete(self, step: str) -> bool:
        return self.state.get("steps", {}).get(step, {}).get("status") == "completed"

    def mark(self, step: str, status: str, **payload) -> None:
        self.state.setdefault("steps", {})[step] = {"status": status, **payload}
        write_json(self.path, self.state)
