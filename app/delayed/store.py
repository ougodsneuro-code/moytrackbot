import os
import json
import logging
from threading import Lock
from typing import Dict, Any

log = logging.getLogger("songbot")

DELAYED_TRACKS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "delayed_tracks.json",
)
DELAYED_TRACKS_PATH = os.path.normpath(DELAYED_TRACKS_PATH)

DELAYED_TRACKS: Dict[str, Dict[str, Any]] = {}
DELAYED_TRACKS_LOCK = Lock()


def load_delayed_tracks() -> None:
    """При старте: читаем delayed_tracks.json в память."""
    global DELAYED_TRACKS
    try:
        with open(DELAYED_TRACKS_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                DELAYED_TRACKS.clear()
                DELAYED_TRACKS.update(loaded)
            else:
                log.warning("delayed_tracks.json has non-dict root, resetting to empty")
                DELAYED_TRACKS = {}
    except FileNotFoundError:
        log.info("delayed_tracks.json not found, starting with empty DELAYED_TRACKS")
        DELAYED_TRACKS = {}
    except Exception as e:
        log.error(f"Failed to load delayed_tracks.json: {e}")
        DELAYED_TRACKS = {}


def save_delayed_tracks() -> None:
    """Сохраняем DELAYED_TRACKS на диск (человеческий JSON)."""
    try:
        tmp_path = DELAYED_TRACKS_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(DELAYED_TRACKS, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, DELAYED_TRACKS_PATH)
    except Exception as e:
        log.error(f"Failed to save delayed_tracks.json: {e}")
