import os
import time
import json
import logging
import requests
from typing import Optional, Tuple, List, Dict, Any

log = logging.getLogger("songbot")




# --- lazy init from env (so direct imports work in REPL/tests) ---
def _lazy_init_from_env():
    """
    Lazy-init BotHelp config from environment if init_bothelp() wasn't called.
    Uses module-level globals only (no phantom _api_base/_client_id vars).
    """
    global _BOTHELP_API_BASE, _BOTHELP_OAUTH_URL, _BOTHELP_CLIENT_ID, _BOTHELP_CLIENT_SECRET

    # if already configured — do nothing
    if _BOTHELP_CLIENT_ID and _BOTHELP_CLIENT_SECRET:
        return

    _BOTHELP_API_BASE = (_BOTHELP_API_BASE or os.getenv("BOTHELP_API_BASE", "https://api.bothelp.io")).rstrip("/")
    _BOTHELP_OAUTH_URL = (os.getenv("BOTHELP_OAUTH_URL", f"{_BOTHELP_API_BASE}/oauth/token")).rstrip("/")
    _BOTHELP_CLIENT_ID = os.getenv("BOTHELP_CLIENT_ID", "").strip()
    _BOTHELP_CLIENT_SECRET = os.getenv("BOTHELP_CLIENT_SECRET", "").strip()




def init_bothelp(
    api_base: str,
    oauth_url: str,
    client_id: str,
    client_secret: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Инициализация клиента BotHelp (один раз на старте main).
    """
    global log, _BOTHELP_API_BASE, _BOTHELP_OAUTH_URL, _BOTHELP_CLIENT_ID, _BOTHELP_CLIENT_SECRET
    if logger is not None:
        log = logger

    _BOTHELP_API_BASE = (api_base or "https://api.bothelp.io").rstrip("/")
    _BOTHELP_OAUTH_URL = (oauth_url or f"{_BOTHELP_API_BASE}/oauth/token").rstrip("/")
    _BOTHELP_CLIENT_ID = (client_id or "").strip()
    _BOTHELP_CLIENT_SECRET = (client_secret or "").strip()


def _mask_key(key: str, show: int = 4) -> str:
    if not key:
        return "***empty***"
    if len(key) <= show:
        return "*" * len(key)
    return key[:show] + "*" * (len(key) - show)


    """
    Возвращает (token, expire_at_unix). Кэширует токен в памяти.
    """
    global _bothelp_token, _bothelp_token_expire_at

    now = time.time()
    if (
        (not force)
        and _bothelp_token
        and now < (_bothelp_token_expire_at - 30)
    ):
        return _bothelp_token, int(_bothelp_token_expire_at)

    if not _BOTHELP_CLIENT_ID or not _BOTHELP_CLIENT_SECRET:
        log.error("BotHelp OAuth: client id/secret missing")
        return None, None

    try:
        data = {
            "grant_type": "client_credentials",
            "client_id": _BOTHELP_CLIENT_ID,
            "client_secret": _BOTHELP_CLIENT_SECRET,
        }
        resp = requests.post(_BOTHELP_OAUTH_URL, data=data, timeout=20)
        if resp.status_code >= 400:
            log.error(f"BotHelp OAuth failed: {resp.status_code} {resp.text[:300]}")
            return None, None

        j = resp.json()
        token = j.get("access_token")
        expires_in = int(j.get("expires_in") or 0)
        if not token or not expires_in:
            log.error(f"BotHelp OAuth bad response: {str(j)[:300]}")
            return None, None

        _bothelp_token = token
        _bothelp_token_expire_at = now + expires_in
        log.info(f"BotHelp OAuth: got token {_mask_key(token, 6)} expire_in={expires_in}s")
        return _bothelp_token, int(_bothelp_token_expire_at)

    except Exception as e:
        log.error(f"BotHelp OAuth exception: {e}")
        return None, None



def fetch_bothelp_token(force: bool = False) -> Tuple[Optional[str], Optional[int]]:
    """
    Возвращает (token, expire_at_unix).
    Кэширует токен в памяти, чтобы не дергать OAuth каждый раз.
    """
    _lazy_init_from_env()
    global _bothelp_token, _bothelp_token_expire_at

    now = time.time()
    if (
        (not force)
        and _bothelp_token
        and now < (_bothelp_token_expire_at - 30)
    ):
        return _bothelp_token, int(_bothelp_token_expire_at)

    if not _BOTHELP_CLIENT_ID or not _BOTHELP_CLIENT_SECRET:
        log.error("BotHelp OAuth: client id/secret missing")
        return None, None

    try:
        url = (_BOTHELP_OAUTH_URL or f"{_BOTHELP_API_BASE.rstrip('/')}/oauth/token")
        data = {
            "grant_type": "client_credentials",
            "client_id": _BOTHELP_CLIENT_ID,
            "client_secret": _BOTHELP_CLIENT_SECRET,
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
        log.info(f"BotHelp OAuth: got token {_mask_key(token, 6)} expire_in={expires_in}s")
        return _bothelp_token, int(_bothelp_token_expire_at)

    except Exception as e:
        log.error(f"BotHelp OAuth exception: {e}")
        return None, None

def _bothelp_authorization_header(force: bool = False) -> Optional[str]:
    tok, _ = fetch_bothelp_token(force=force)
    if not tok:
        return None
    return f"Bearer {tok}"


def send_message_to_bothelp_via_cuid(
    subscriber_cuid: str,
    msgs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not subscriber_cuid:
        return {"ok": False, "error": "missing_cuid"}

    cleaned_payload = []
    for m in msgs:
        if isinstance(m, dict):
            cleaned_payload.append(m)

    url = f"{_BOTHELP_API_BASE}/v1/subscribers/cuid/{subscriber_cuid}/messages"

    def _do_post(auth_header: str):
        headers = {
            "Content-Type": "application/vnd.api+json",
            "Accept": "application/json",
            "Authorization": auth_header,
        }
        return requests.post(
            url,
            headers=headers,
            data=json.dumps(cleaned_payload, ensure_ascii=False).encode("utf-8"),
            timeout=60,
        )

    last_err = None
    for attempt in range(1, 4):
        auth_header = _bothelp_authorization_header(force=False)
        if not auth_header:
            return {"ok": False, "error": "no_valid_bothelp_token"}

        try:
            resp = _do_post(auth_header)
        except requests.exceptions.RequestException as e:
            last_err = f"exception:{e}"
            log.exception(f"BotHelp send: exception on attempt {attempt}")
            time.sleep(min(2 ** attempt, 8))
            continue

        status = resp.status_code
        txt = resp.text
        log.info(f"BotHelp POST {url} => {status}")

        if status in (401, 403):
            # refresh token and retry
            _bothelp_authorization_header(force=True)
            last_err = f"http_{status}"
            time.sleep(min(2 ** attempt, 8))
            continue

        try:
            body_json = resp.json()
        except Exception:
            body_json = {"raw": txt}

        if 200 <= status < 300:
            return {"ok": True, "status": status, "response": body_json}

        last_err = body_json.get("error", txt)
        time.sleep(min(2 ** attempt, 8))

    return {"ok": False, "error": last_err or "send_failed"}


def upload_audio_to_bothelp(audio_bytes: bytes, filename: str = "track.mp3") -> Optional[str]:
    auth_header = _bothelp_authorization_header(force=False)
    if not auth_header:
        log.error("upload_audio_to_bothelp: no bothelp token")
        return None

    url = f"{_BOTHELP_API_BASE}/v1/attachments"
    headers = {"Authorization": auth_header}
    files = {"file": (filename, audio_bytes, "audio/mpeg")}

    try:
        resp = requests.post(url, headers=headers, files=files, timeout=120)
    except Exception as e:
        log.exception("BotHelp upload exception: %s", e)
        return None

    if resp.status_code not in (200, 201):
        log.error(f"BotHelp upload error {resp.status_code}: {resp.text[:500]}")
        return None

    try:
        j = resp.json()
    except Exception:
        log.error("BotHelp upload not JSON")
        return None

    att_id = (j.get("data", {}).get("id") or j.get("id"))
    if not att_id:
        log.error(f"BotHelp upload no attachment id in {j}")
        return None
    return att_id
