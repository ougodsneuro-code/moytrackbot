def _mask_key(key: str, show: int = 4) -> str:
    if not key:
        return "***empty***"
    if len(key) <= show:
        return "*" * len(key)
    return key[:show] + "*" * (len(key) - show)

def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False
