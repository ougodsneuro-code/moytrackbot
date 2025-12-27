import time
import logging
import requests
from typing import Optional, Tuple

log = logging.getLogger("songbot")

_bothelp_token: Optional[str] = None
_bothelp_token_expire_at: float = 0.0


def fetch_bothelp_token(
    api_base: str,
    client_id: str,
    client_secret: str,
    force: bool = False,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Возвращает (token, expire_at_unix).
    Кэширует токен в памяти, чтобы не дергать OAuth каждый раз.
    """
    global _bothelp_token, _bothelp_token_expire_at

    now = time.time()
    if (
        (not force)
        and _bothelp_token
        and now < (_bothelp_token_expire_at - 30)
    ):
        return _bothelp_token, int(_bothelp_token_expire_at)

    if not client_id or not client_secret:
        log.error("BotHelp OAuth: client id/secret missing")
        return None, None

    try:
        url = f"{api_base.rstrip('/')}/oauth/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        r = requests.post(url, data=data, timeout=20)
        if r.status_code >= 400:
            log.error(f"BotHelp OAuth failed: {r.status_code} {r.text[:300]}")
            return None, None

        j = r.json()
        token = j.get("access_token")
        expires_in = int(j.get("expires_in") or 0)
        if not token or not expires_in:
            log.error(f"BotHelp OAuth bad response: {str(j)[:300]}")
            return None, None

        _bothelp_token = token
        _bothelp_token_expire_at = now + expires_in
        return _bothelp_token, int(_bothelp_token_expire_at)

    except Exception as e:
        log.error(f"BotHelp OAuth exception: {e}")
        return None, None
