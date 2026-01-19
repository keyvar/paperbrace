from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tomllib


@dataclass(frozen=True)
class AppConfig:
    llm_model: str = ""
    llm_max_new_tokens: int = 350
    llm_temperature: float = 0.2


def load_config(path: Optional[Path] = None) -> AppConfig:
    """
    Load Paperbrace config from a TOML file.

    Precedence is handled by the caller; this just loads one file.
    If the file is missing, returns defaults.
    """
    path = path or Path("paperbrace.toml")
    if not path.exists():
        return AppConfig()

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    llm = (data.get("llm") or {})

    return AppConfig(
        llm_model=str(llm.get("model", "")),
        llm_max_new_tokens=int(llm.get("max_new_tokens", 350)),
        llm_temperature=float(llm.get("temperature", 0.2)),
    )
