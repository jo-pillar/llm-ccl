from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


class ManifestStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported manifest schema: {data.get('schema_version')}")
        return data

    def create(self, payload: dict[str, Any]) -> None:
        if self.path.exists():
            raise FileExistsError(self.path)
        self._write(payload)

    def update(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        payload = self.load()
        mutator(payload)
        self._write(payload)
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        self._write(payload)

    def _write(self, payload: dict[str, Any]) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
