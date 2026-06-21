from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class FlowSimRunner:
    def __init__(self, binary: str | Path, timeout_s: float | None = None) -> None:
        self.binary = str(binary)
        self.timeout_s = timeout_s

    def simulate_sketch(
        self,
        *,
        config_path: str | Path,
        sketch_path: str | Path,
        output_path: str | Path,
    ) -> dict[str, Any]:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.binary,
            "simulate-sketch",
            "--config",
            str(config_path),
            "--sketch",
            str(sketch_path),
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"flow-sim-rs simulate-sketch failed with {completed.returncode}: {completed.stderr or completed.stdout}"
            )
        return json.loads(output.read_text(encoding="utf-8"))

