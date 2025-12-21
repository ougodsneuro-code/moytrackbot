import os
import logging
from pathlib import Path

log = logging.getLogger("songbot")

def load_env_robust(env_path: str = ".env") -> None:
    """
    Robust .env loader:
    - reads KEY=VALUE lines
    - ignores comments/blanks
    - does not overwrite already set env vars
    """
    p = Path(env_path)

    if not p.exists():
        log.info("No .env file found at %s (ok)", p.resolve())
        return

    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if not k:
                continue
            os.environ.setdefault(k, v)

        log.info(".env loaded from %s", p.resolve())
    except Exception as e:
        log.exception("Failed to load .env: %s", e)
