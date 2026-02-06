from __future__ import annotations
import os, sys

def main() -> None:
    if "--offline" in sys.argv and "--no-offline" not in sys.argv:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from paperbrace.cli import app  # noqa: PLC0415
    app()
