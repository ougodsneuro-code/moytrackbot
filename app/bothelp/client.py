import logging
import requests
from typing import Optional

from app.bothelp.auth import fetch_bothelp_token

log = logging.getLogger("songbot")


def _bothelp_auth_headers(
    api_base: str,
    client_id: str,
    client_secret: str,
) -> Optional[dict]:
    token, _ = fetch_bothelp_token(
        api_base=api_base,
        client_id=client_id,
        client_secret=client_secret,
    )
    if not token:
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def send_message_to_bothelp_via_cuid(
    *,
    api_base: str,
    client_id: str,
    client_secret: str,
    cuid: str,
    text: str,
) -> bool:
    headers = _bothelp_auth_headers(api_base, client_id, client_secret)
    if not headers:
        log.error("BotHelp send failed: no auth headers")
        return False

    url = f"{api_base.rstrip('/')}/v1/messages"
    payload = {
        "recipient": {"cuid": cuid},
        "message": {"type": "text", "text": text},
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        if r.status_code >= 400:
            log.error(f"BotHelp send error {r.status_code}: {r.text[:300]}")
            return False
        return True
    except Exception as e:
        log.error(f"BotHelp send exception: {e}")
        return False


def upload_audio_to_bothelp(
    *,
    api_base: str,
    client_id: str,
    client_secret: str,
    cuid: str,
    audio_url: str,
    filename: str = "track.mp3",
) -> bool:
    headers = _bothelp_auth_headers(api_base, client_id, client_secret)
    if not headers:
        log.error("BotHelp audio upload failed: no auth headers")
        return False

    url = f"{api_base.rstrip('/')}/v1/messages"
    payload = {
        "recipient": {"cuid": cuid},
        "message": {
            "type": "audio",
            "audio": {
                "url": audio_url,
                "filename": filename,
            },
        },
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        if r.status_code >= 400:
            log.error(f"BotHelp audio error {r.status_code}: {r.text[:300]}")
            return False
        return True
    except Exception as e:
        log.error(f"BotHelp audio exception: {e}")
        return False
