import time
import threading
import logging
from typing import Dict, Any, List

from .store import (
    DELAYED_TRACKS,
    DELAYED_TRACKS_LOCK,
    load_delayed_tracks,
    save_delayed_tracks,
)

log = logging.getLogger("songbot")

# важно: подгружаем отложенные задачи при импорте модуля
load_delayed_tracks()

_DELAYED_BOOT_RESTORE_DONE = False


def restore_delayed_sends_on_boot(_send_tracks_to_user):
    """
    После рестарта:
    - если время отправки уже прошло — сразу досылаем треки и чистим запись;
    - если время в будущем — заново вешаем таймер.
    """
    with DELAYED_TRACKS_LOCK:
        items = list(DELAYED_TRACKS.items())

    if not items:
        log.info("[DELAYED] no delayed tasks to restore on boot")
        return

    now = time.time()
    restored = 0
    sent_immediately = 0

    for task_id, entry in items:
        cuid = entry.get("cuid")
        provider = entry.get("provider")
        tracks = entry.get("tracks") or []
        send_at_ts = float(entry.get("send_at_ts") or 0)

        if not cuid or not provider or not tracks:
            log.warning(f"[DELAYED] invalid entry in delayed_tracks for task {task_id}, dropping")
            with DELAYED_TRACKS_LOCK:
                DELAYED_TRACKS.pop(task_id, None)
                save_delayed_tracks()
            continue

        if not send_at_ts or send_at_ts <= now:
            log.info(
                f"[DELAYED] boot: send_at_ts in past, sending now "
                f"(task_id={task_id}, cuid={cuid})"
            )
            try:
                _send_tracks_to_user(cuid, provider, task_id, tracks)
                sent_immediately += 1
            except Exception:
                log.exception(f"[DELAYED] boot send failed for task {task_id}")
            with DELAYED_TRACKS_LOCK:
                DELAYED_TRACKS.pop(task_id, None)
                save_delayed_tracks()
            continue

        delta = max(0, send_at_ts - now)
        log.info(
            f"[DELAYED] boot: rescheduling task {task_id} in ~{int(delta)}s "
            f"(cuid={cuid}, provider={provider})"
        )

        def _do_send(task_id=task_id, cuid=cuid, provider=provider):
            with DELAYED_TRACKS_LOCK:
                entry2 = DELAYED_TRACKS.pop(task_id, None)
                save_delayed_tracks()

            if not entry2:
                log.warning(f"[DELAYED] entry for task {task_id} disappeared before send")
                return

            tracks2 = entry2.get("tracks") or []
            if not tracks2:
                log.warning(f"[DELAYED] entry for task {task_id} has no tracks on send, skip")
                return

            _send_tracks_to_user(cuid, provider, task_id, tracks2)
            log.info(f"[DELAYED] sent delayed tracks for task {task_id} (cuid={cuid})")

        t = threading.Timer(delta, _do_send)
        t.daemon = True
        t.start()
        restored += 1

    log.info(
        f"[DELAYED] boot restore: rescheduled={restored}, "
        f"sent_immediately={sent_immediately}"
    )


def restore_delayed_sends_once(_send_tracks_to_user):
    global _DELAYED_BOOT_RESTORE_DONE
    if _DELAYED_BOOT_RESTORE_DONE:
        return
    _DELAYED_BOOT_RESTORE_DONE = True
    try:
        restore_delayed_sends_on_boot(_send_tracks_to_user)
    except Exception:
        log.exception("[DELAYED] restore on boot failed")


def schedule_delayed_send(
    _send_tracks_to_user,
    cuid: str,
    provider: str,
    task_id: str,
    tracks: List[Dict[str, Any]],
    send_at_ts: float,
):
    """
    Кладём готовые треки в delayed-очередь, чтобы:
    - отправить их в нужный момент времени (send_at_ts — unix timestamp)
    - пережить рестарт сервиса (всё лежит в delayed_tracks.json)
    """
    if not tracks:
        log.warning(f"[DELAYED] no tracks for task {task_id}, skip")
        return

    now_ts = time.time()
    if not send_at_ts or send_at_ts <= now_ts:
        log.info(
            f"[DELAYED] send_at_ts already in past or empty, sending now "
            f"(task_id={task_id}, cuid={cuid})"
        )
        _send_tracks_to_user(cuid, provider, task_id, tracks)
        return

    with DELAYED_TRACKS_LOCK:
        DELAYED_TRACKS[task_id] = {
            "cuid": cuid,
            "provider": provider,
            "tracks": tracks,
            "send_at_ts": float(send_at_ts),
        }
        save_delayed_tracks()

    def _do_send():
        with DELAYED_TRACKS_LOCK:
            entry = DELAYED_TRACKS.pop(task_id, None)
            save_delayed_tracks()
        if not entry:
            log.warning(f"[DELAYED] entry for task {task_id} disappeared, skip")
            return

        _send_tracks_to_user(
            entry.get("cuid") or cuid,
            entry.get("provider") or provider,
            task_id,
            entry.get("tracks") or tracks,
        )
        log.info(f"[DELAYED] sent delayed tracks for task {task_id} (cuid={cuid})")

    delta = max(0, send_at_ts - time.time())
    t = threading.Timer(delta, _do_send)
    t.daemon = True
    t.start()

    log.info(
        f"[DELAYED] scheduled send for task {task_id} "
        f"(cuid={cuid}, in ~{int(delta)}s)"
    )
