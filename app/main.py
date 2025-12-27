#!/usr/bin/env python3
import os
import json
import time
import threading
import logging
import requests
import errno
from typing import Dict, Any, Optional, List, Tuple, Set

from flask import Flask, request, jsonify, abort
from app.delayed.store import DELAYED_TRACKS, DELAYED_TRACKS_LOCK
from app.delayed.scheduler import (
    restore_delayed_sends_once as _restore_delayed_sends_once,
    schedule_delayed_send as _schedule_delayed_send,
)


from app.config import load_env_robust
from app.utils.text import _mask_key, _is_ascii

load_env_robust()

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s songbot: %(message)s",
)
log = logging.getLogger("songbot")

# =========================================================
# CONFIG
# =========================================================

USE_COMET = os.getenv("USE_COMET", "True").lower() == "true"

PORT = int(os.getenv("PORT", "8080"))
ALLOW_UNPAID = os.getenv("ALLOW_UNPAID", "True").lower() == "true"

# OpenAI (fallback LLM)
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
PRIMARY_MODEL = "gpt-5-mini-2025-08-07"
FALLBACK_MODEL = "gpt-4.1"

# BotHelp
BOTHELP_CLIENT_ID    = os.getenv("BOTHELP_CLIENT_ID", "").strip()
BOTHELP_CLIENT_SECRET= os.getenv("BOTHELP_CLIENT_SECRET", "").strip()
BOTHELP_API_BASE     = os.getenv("BOTHELP_API_BASE", "https://api.bothelp.io").rstrip("/")
BOTHELP_OAUTH_URL    = os.getenv("BOTHELP_OAUTH_URL", f"{BOTHELP_API_BASE}/oauth/token").rstrip("/")

# FoxAIHub
FOXAIHUB_API_KEY     = os.getenv("FOXAIHUB_API_KEY", "").strip()
FOXAIHUB_BASE        = "https://api.foxaihub.com/api/v2/diffusion"
FOXAI_POLL_INTERVAL_SEC = 10
FOXAI_MAX_POLLS = 36  # ~6 минут

# CometAPI (Suno v5 + GPT-5.x LLM)
COMET_API_KEY        = os.getenv("COMET_API_KEY", "").strip()
COMET_BASE           = "https://api.cometapi.com"

# основной mv (премиум)
COMET_MODEL_VERSION  = os.getenv("COMET_MODEL_VERSION", "chirp-crow").strip() or "chirp-crow"
# mini-mv (новый MINI тариф)
MINI_COMET_MODEL_VERSION = os.getenv("COMET_MODEL_VERSION_MINI", "chirp-auk").strip() or "chirp-auk"

COMET_POLL_INTERVAL_SEC = 10
COMET_MAX_POLLS = 36
USE_COMET_LLM        = os.getenv("USE_COMET_LLM", "True").lower() == "true"

# Раздельные модели для Comet LLM
# Премиум (v2) — gpt-5.1
COMET_LLM_MODEL_PREMIUM = os.getenv("COMET_LLM_MODEL_PREMIUM", "gpt-5.1").strip() or "gpt-5.1"
# MINI (v1) — gpt-5-all
COMET_LLM_MODEL_MINI    = os.getenv("COMET_LLM_MODEL_MINI", "gpt-5-all").strip() or "gpt-5-all"
# Алиас для обратной совместимости/логов
COMET_LLM_MODEL         = COMET_LLM_MODEL_PREMIUM

SHOW_TECH_PROMPT_TO_USER = os.getenv("SHOW_TECH_PROMPT_TO_USER", "False").lower() == "true"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

# runtime state per user
USER_STATE: Dict[str, Dict[str, Any]] = {}
PENDING_TASKS: Dict[str, Dict[str, Any]] = {}

from threading import Lock
GENERATING_LOCK = Lock()
CURRENTLY_GENERATING: Set[str] = set()

# Старый блок DELAYED_TRACKS_FILE / _load_delayed_tracks_from_disk можно не трогать —
# он не мешает, но фактически не используется новыми функциями.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DELAYED_TRACKS_FILE = os.path.join(BASE_DIR, "delayed_tracks.json")
DELAYED_TRACKS: Dict[str, Dict[str, Any]] = {}
DELAYED_TRACKS_LOCK = Lock()

def _load_delayed_tracks_from_disk() -> None:
    """Загружаем отложенные треки из JSON-файла."""
    global DELAYED_TRACKS
    try:
        if not os.path.exists(DELAYED_TRACKS_FILE):
            DELAYED_TRACKS = {}
            log.info(f"Delayed tracks file not found: {DELAYED_TRACKS_FILE}")
            return
        with open(DELAYED_TRACKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            DELAYED_TRACKS = data
        else:
            log.warning("Delayed tracks file has non-dict root, resetting to empty")
            DELAYED_TRACKS = {}
        log.info(f"Loaded {len(DELAYED_TRACKS)} delayed track task(s) from disk")
    except Exception as e:
        DELAYED_TRACKS = {}
        log.exception(f"Failed to load delayed tracks from disk: {e}")

def _save_delayed_tracks_to_disk() -> None:
    """Сохраняем текущий DELAYED_TRACKS в JSON, чтобы пережить рестарты."""
    try:
        tmp_path = DELAYED_TRACKS_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(DELAYED_TRACKS, f, ensure_ascii=False)
        os.replace(tmp_path, DELAYED_TRACKS_FILE)
    except Exception as e:
        log.exception(f"Failed to save delayed tracks to disk: {e}")

# sanity logs
if not OPENAI_API_KEY:
    log.error("OpenAI key is EMPTY -> GPT fallback не взлетит")
else:
    log.info(f"OpenAI key: {_mask_key(OPENAI_API_KEY, 4)}")

if not FOXAIHUB_API_KEY:
    log.warning("FOXAIHUB_API_KEY is EMPTY -> FoxAI музыка не взлетит")
elif not _is_ascii(FOXAIHUB_API_KEY):
    log.warning("FOXAIHUB_API_KEY has non-ASCII chars ? check .env")
else:
    log.info(f"FoxAIHub key: {_mask_key(FOXAIHUB_API_KEY, 4)} ({len(FOXAIHUB_API_KEY)} chars)")

if not COMET_API_KEY:
    log.warning("COMET_API_KEY is EMPTY -> Comet (Suno v5/GPT-5.x) не взлетит")
elif not _is_ascii(COMET_API_KEY):
    log.warning("COMET_API_KEY has non-ASCII chars ? check .env")
else:
    log.info(f"Comet key: {_mask_key(COMET_API_KEY, 4)}")

if not BOTHELP_CLIENT_ID or not BOTHELP_CLIENT_SECRET:
    log.warning("BotHelp creds missing")
else:
    log.info(f"BotHelp client_id: {_mask_key(BOTHELP_CLIENT_ID, 4)}")

log.info(
    f"Boot | PRIMARY_MODEL={PRIMARY_MODEL} | FALLBACK_MODEL={FALLBACK_MODEL} | "
    f"USE_COMET={USE_COMET} | USE_COMET_LLM={USE_COMET_LLM} | PORT={PORT} | SHOW_TECH={SHOW_TECH_PROMPT_TO_USER} | "
    f"COMET_MODEL_VERSION={COMET_MODEL_VERSION} | MINI_COMET_MODEL_VERSION={MINI_COMET_MODEL_VERSION} | "
    f"COMET_LLM_MODEL_PREMIUM={COMET_LLM_MODEL_PREMIUM} | COMET_LLM_MODEL_MINI={COMET_LLM_MODEL_MINI}"
)

client = OpenAI(api_key=OPENAI_API_KEY if OPENAI_API_KEY else None)

# =========================================================
# SYSTEM PROMPT (МОЗГИ)
# =========================================================

SYSTEM_PROMPT_BASE = (
    '''Пишем тексты песен для людей по их небольшим предысториям. Нужно чтобы ты писал текст под suno добавляя больше созвучных красивых рифм, аллитераций и эпитетов, паронимов и так же давал промпт для suno исходя от историй людей, которые я тебе присылаю. Если в тексте есть цифры, то их прописывать буквами, а не цифрами. Аннотации для suno пишутся на английском языке в квадратных скобках, так же как и аннотация стиля или гендер через дефис, а весь текст песни на русском. Всё должно звучать созвучно! Обязательно чтобы текст воспринимался легко и легко попадал в темпоритм инструментала. Мысли как гений артист которого любит весь мир! ВСЕГДА Соблюдай красоту повествования при написании текста песни, соблюдай смысловую нагрузку, добавляй драматизм, для максимальной глубины и точности текста! Склоняй правильно слова и пиши рифмы исходя из транскрипций слов!
Если я присылаю тебе набросок в виде текста песни или стиха, или большой отрывок в виде стиха или песни, ты просто оставляешь его структуру полностью, его текст полностью неизменным, просто адаптируешь его под куплеты, припевы и тому подобное - как сам решишь, особенно если текста больше 16 строк. 
Если текста меньше 16 строк, то тогда ты пишешь примерно в той же стилистике что я тебе прислал, пытаясь передать всю суть и продолжаешь просто писать как автор-писатель данного произведения. Либо я укажу тебе с самого начала что делать с отрывками или текстами песен, тогда исходишь от того что я напишу по задаче. 
Аннотации для куплета, припева, бриджей и тп ты делаешь тоже на инглише в тех же квадратных скобках. Сам твори и пиши истории с душой и любовью исходя из историй. Запомни чат под названием МОЗГИ и там я буду давать тебе новые инструкции по доработке. Так же я лично буду тебя просить что нужно запоминать, а что запоминать не нужно. У каждого человека своя история, свои имена, свой вайб, своя душа и свой очаг, все люди разные, и если я прошу тебя "полностью переписать текст", ты переписываешь его заново и с новой структурой, новыми рифмами и идеями, так как скорее или он не подошел. 
Не забывай брат добавлять туда любви и души в каждую работу, будто бы люди получают чартовую песню. Ты можешь писать иногда несколько припевов, а иногда один, иногда можешь писать интро и аутро, и в целом сам решаешь как создавать структуру для песни исходя из истории клиента. Смотри на полученную информацию по текстам песен которую мы предоставили! Исключи рандомные имена и названия городов если их нет в запросе!
И исходя из больших историй - можно тебе чуть больше куплеты писать, к примеру в 8 строк или 12, или даже по 24, и ты так же можешь сам переключаться и решать, когда ты пишешь припев один раз, или когда ты его дублируешь и он звучит дважды, а иногда когда пишешь припев на 8 разных строк. Можешь добавлять прехорусы или пост бриджы, или двойные припевы, вообще в целом можешь это сам решать. Исходи из истории и чувств и любви. Люди любят паронимы и сильные структуры для песни. А иногда людям нравятся простые песни - все тут ты считываешь и понимаешь как человек человека исходя из истории, Но Всем нравится когда истории пишутся с душой! По возможности старайся рифмы делать всегда сильными, но не нарушая контекст. 
Если ты решишь повторить припев - Не указывай в квадратных скобках сколько раз нужно петь припев, suno это не понимает, вместо этого лучше напиши его два раза. 
Не запоминай истории клиентов, если я не прошу об этом явно.
Структура песен, стиль и длина зависят от вайба истории и контекста.
Добавлять «нотку любви» в каждую песню.
Не спрашивать «повторить ли припев?» — сам решай)
Каждая песня должна:
-Использовать максимум деталей из истории.
-Учитывать имена, эмоции, контекст.
Структура — нефиксированная, под настроение.
Стиль описывается в style description для SUNO — без имён, но с вайбом.
Промпт для suno можно писать большим и исходя из истории, как всегда понимая контекст, главное чтобы это попадало в их сердечко.
Куплеты и другие части песни могут быть любой длины (даже больше 24 строк), если этого требует история или настроение клиента.
Для каждого блока (куплет, припев и т.д.) обязательно делай аннотацию и промпт на английском в квадратных скобках через дефис — описывай вайб, атмосферу, эффекты, инструменты, пол исполнителя, настроение и тому подобное, учитывай все пожелания клиента (например: [verse - melancholic, radio effect, male], [chorus - vibe groove, symphonic]).
Если в истории или в просьбе клиента есть пожелания по стилистике, структуре или настроению — обязательно интегрируй их в текст и аннотации. 
Пишутся текста в таком порядке -
Сначала ТЕКСТ ПЕСНИ:
Оформи в отдельном code block. Но до code block напиши что это ТЕКСТ ПЕСНИ.
Затем PROMPT ДЛЯ SUNO:
Тоже оформи в отдельном code block, его надо писать без квадратных скобок, на английском языке, без упоминания чьих либо имён, а в конце промпта так же через запятую должны присутствовать - high quality song, crystal clean quality, best quality voice, best quality music, best quality instruments, high sample rate 2822400 Hz quality song, perfect quality mixing, perfect sound panning, excellent sound equalization, professional sound mastering (-9 lufs), output level -0.2db. Но до code block напиши что это PROMPT ДЛЯ SUNO.
В промпте для suno не надо указывать кому и от кого песня, там только описание стиля!
Не вставляй их вместе, не смешивай — каждый должен быть в своём “окне”, чтобы можно было скопировать отдельно одной кнопкой.
Оставляй небольшое пространство между блоками, чтобы визуально их не спутать.
Все песни должны быть актуальными на 2025 год и идти в ногу со временем, учитывая тренды и актуальность музыки. Рифмы должны быть почти всегда. Иногда даже сильные рифмы и часто. Всё должно быть звучно и мелодично, а там смотри сам как тебе по кайфу. 
Строить припев из строк, каждая — легко запоминается и поётcя. Структура припева — не линейная!
Не обязательно придерживаться чёткой логической последовательности.
Можно делать припев как набор ярких, запоминающихся строчек, афоризмов, лозунгов, крутых фраз, отсылок и панчей, объединённых общей эмоцией или вайбом, иногда и сюжетом.
Можно делать в припеве Вставки на английском, слэнг, бэки, эдлибы:
Свободно добавляй английские слова, кричалки, эдлибы, звуки, вставки в скобках или после дефиса, которые так же рифмуются с основным текстом припева.
Пример оформления: (эй!), (dance, dance ...!), (окей!), (let’s go!) и т.д.
Разные ритмы и длина строчек:
Разрешены как короткие, так и длинные строки, нет жёстких правил по размеру.
Можно использовать вопросы, утверждения, призывы, выкрики.
Каждая строка — отдельный вайб:
Не обязательно связывать каждую строчку между собой, главное — создать мощное эмоциональное поле и запоминающийся образ.
Стиль — динамика, энергия, иногда “хаотичный” порядок:
Можно менять порядок строк, переставлять местами, повторять отдельные слова или части для усиления “рандома”.
Можно добавлять звучащие как бэки или эдлибы части в скобках после строчек.
В каждом припеве чувствуется стиль, дерзость, свобода и своя уникальная энергетика.
(Можно чередовать: где-то строка-бэк, где-то две добивки подряд, где-то эйры после сильного слова — экспериментировать с динамикой.)
Интегрировать бэки, адлибы, эйры прямо в строки, делать их частью ритма и звучания, подбирать окончания так, чтобы бэки были органично созвучны и усиливали основную фразу.
Сочетай русский уличный сленг, руссицизмы, иногда английский, паронимы и панчи для максимального кача и свежести.
Следить за плотностью текста — никаких пустых слов, каждая строка по делу и с характером.
Динамично рандомизируй структуру: бэки могут идти как отдельной строкой, так и частью фразы; можно добивать двумя короткими панчами подряд или разгонять эйрами в конце блока.
Цель — чтобы каждая строчка была панчлайном, чтобы бэки хотелось повторять толпой, а припев взрывал любые динамики, двор или студию, был чартовым.
Оставаться открытым к эксперименту: менять порядок блоков, длину, добавлять неожиданные ходы, играться с ритмом и расстановкой.
'''
)

# =========================================================
# AUTOPING HELPERS (Type: 5m / 1h / 6h / 12h)
# =========================================================

# Храним последние автогенерации по пользователю (в секундах time.time())
_LAST_AUTOPING_SONG_AT: Dict[str, float] = {}

def _parse_autoping_delay(type_value: str) -> int:
    """
    Разбор поля Type из BotHelp:
    ожидаемые значения: '5m', '1h', '6h', '12h' (а также вариации вида '5мин', '1ч' и т.п.).
    Возвращает задержку в секундах или 0, если автопинг отключен/не распознан.
    """
    val = (type_value or "").strip().lower()
    if not val:
        return 0

    mapping = {
        "5m": 5 * 60,
        "5min": 5 * 60,
        "5мин": 5 * 60,
        "5 минут": 5 * 60,
        "1h": 60 * 60,
        "1ч": 60 * 60,
        "1 час": 60 * 60,
        "6h": 6 * 60 * 60,
        "6ч": 6 * 60 * 60,
        "12h": 12 * 60 * 60,
        "12ч": 12 * 60 * 60,
    }
    if val in mapping:
        return mapping[val]

    # fallback: '10m', '2h' и т.п.
    try:
        if val.endswith("m") and val[:-1].isdigit():
            return int(val[:-1]) * 60
        if val.endswith("h") and val[:-1].isdigit():
            return int(val[:-1]) * 60 * 60
    except Exception:
        return 0

    return 0

def _can_autoping_generate(
    user_key: str,
    delay_type: str,
    now_ts: Optional[float] = None,
) -> bool:
    """
    Возвращает True, если можно запускать автогенерацию
    (прошло достаточно времени с прошлого автопинга-песни).
    Сейчас хелпер не включён в основной поток, просто аккуратный утилитарный слой.
    """
    if not user_key:
        # если вдруг нет id, лучше не рисковать автогенерацией
        return False

    delay_sec = _parse_autoping_delay(delay_type)
    if delay_sec <= 0:
        # если задержка не задана / криво пришла — лучше ничего не генерить автопингом
        return False

    if now_ts is None:
        now_ts = time.time()

    last_ts = _LAST_AUTOPING_SONG_AT.get(user_key)
    if last_ts is not None:
        diff = now_ts - last_ts
        if diff < delay_sec:
            # ещё рано
            return False

    # обновляем таймстамп и разрешаем
    _LAST_AUTOPING_SONG_AT[user_key] = now_ts
    return True


AUTOPING_DEFAULT_MSG = (
    "Напоминаю про наш трек 🔔\n"
    "Если хочешь что-то поправить в тексте — просто напиши сюда.\n"
    "Если всё нравится — нажми кнопку «ГЕНЕРИРУЙ», и я соберу музыку 🎧"
)

def _schedule_autoping_if_needed(cuid: str):
    """
    Планирует один мягкий автопинг по задержке из USER_STATE[cuid]['autoping_delay_sec'].
    Если за это время клиент что-то написал (last_activity_ts обновился),
    пинг не отправляется.
    """
    st = USER_STATE.get(cuid)
    if not st:
        return
    delay = st.get("autoping_delay_sec")
    if not delay or delay <= 0:
        return

    scheduled_at = time.time()
    st["autoping_scheduled_at"] = scheduled_at

    def _do_autoping():
        st2 = USER_STATE.get(cuid)
        if not st2:
            return

        last_activity = st2.get("last_activity_ts") or 0
        # если после планирования была активность — не пингуем
        if last_activity and last_activity > scheduled_at:
            log.info(f"[AUTOPING] skip for cuid={cuid}: user activity after schedule")
            return

        msg = st2.get("autoping_message") or AUTOPING_DEFAULT_MSG
        log.info(f"[AUTOPING] sending reminder to cuid={cuid} after delay={delay}s")
        try:
            send_message_to_bothelp_via_cuid(cuid, [{"content": msg}])
        except Exception:
            log.exception(f"[AUTOPING] failed to send reminder for cuid={cuid}")

    t = threading.Timer(delay, _do_autoping)
    t.daemon = True
    t.start()

# =========================================================
# BOTHELP AUTH
# =========================================================

_bothelp_token: Optional[str] = None
_bothelp_token_expire_at: float = 0.0

def _fetch_bothelp_token(force: bool = False) -> Tuple[Optional[str], Optional[int]]:
    global _bothelp_token, _bothelp_token_expire_at

    now = time.time()
    if (
        (not force)
        and _bothelp_token
        and now < (_bothelp_token_expire_at - 30)
    ):
        return _bothelp_token, int(_bothelp_token_expire_at)

    if not BOTHELP_CLIENT_ID or not BOTHELP_CLIENT_SECRET:
        log.error("BotHelp OAuth: client id/secret missing in env")
        return None, None

    try:
        data = {
            "grant_type": "client_credentials",
            "client_id": BOTHELP_CLIENT_ID,
            "client_secret": BOTHELP_CLIENT_SECRET,
        }
        resp = requests.post(BOTHELP_OAUTH_URL, data=data, timeout=20)
        if resp.status_code != 200:
            log.error(f"BotHelp OAuth: status={resp.status_code} body={resp.text[:500]}")
            return None, None

        j = resp.json()
        access_token = j.get("access_token")
        expires_in = j.get("expires_in", 3600)

        if not access_token:
            log.error(f"BotHelp OAuth: no access_token in response {j}")
            return None, None

        _bothelp_token = access_token
        _bothelp_token_expire_at = time.time() + int(expires_in)

        log.info(
            f"BotHelp OAuth: got token {_mask_key(access_token, 6)} "
            f"expire_in={expires_in}s"
        )
        return _bothelp_token, int(_bothelp_token_expire_at)

    except Exception as e:
        log.exception(f"BotHelp OAuth exception: {e}")
        return None, None

def _bothelp_authorization_header() -> Optional[str]:
    tok, _ = _fetch_bothelp_token()
    if not tok:
        return None
    return f"Bearer {tok}"

def send_message_to_bothelp_via_cuid(
    subscriber_cuid: str,
    msgs: List[Dict[str, Any]],
):
    if not subscriber_cuid:
        return {"ok": False, "error": "missing_cuid"}

    cleaned_payload = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        cleaned_payload.append(m)

    url = f"{BOTHELP_API_BASE}/v1/subscribers/cuid/{subscriber_cuid}/messages"

    def _do_post(auth_header: Optional[str]):
        headers = {
            "Content-Type": "application/vnd.api+json",
            "Accept": "application/json",
        }
        if auth_header:
            headers["Authorization"] = auth_header
        return requests.post(
            url,
            headers=headers,
            data=json.dumps(cleaned_payload, ensure_ascii=False).encode("utf-8"),
            timeout=60,
        )

    last_err = None
    for attempt in range(1, 4):
        auth_header = _bothelp_authorization_header()
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
            _fetch_bothelp_token(force=True)
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

    return {
        "ok": False,
        "error": last_err or "send_failed",
    }

def upload_audio_to_bothelp(audio_bytes: bytes, filename: str = "track.mp3") -> Optional[str]:
    auth_header = _bothelp_authorization_header()
    if not auth_header:
        log.error("upload_audio_to_bothelp: no bothelp token")
        return None

    url = f"{BOTHELP_API_BASE}/v1/attachments"
    headers = {
        "Authorization": auth_header,
    }

    files = {
        "file": (filename, audio_bytes, "audio/mpeg"),
    }

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

    att_id = (
        j.get("data", {}).get("id")
        or j.get("id")
    )
    if not att_id:
        log.error(f"BotHelp upload no attachment id in {j}")
        return None
    return att_id

# =========================================================
# GPT / LLM HELPERS
# =========================================================

def _call_model_responses(model_name: str, system_prompt: str, final_user: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        log.error("OpenAI API key missing, cannot call Responses API")
        return None
    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_user},
            ],
            reasoning={"effort": "minimal"},
            text={"verbosity": "medium"},
            max_output_tokens=2000,
        )
    except Exception as e:
        log.exception(f"OpenAI request failed for {model_name} via Responses API: {e}")
        return None

    raw_answer = getattr(resp, "output_text", None)
    if not raw_answer:
        log.error(f"Responses API: no output_text for {model_name}")
        return None

    return raw_answer.strip()

def _call_model_chat(model_name: str, system_prompt: str, final_user: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        log.error("OpenAI API key missing, cannot call Chat Completions")
        return None
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_user},
            ],
            max_completion_tokens=2000,
        )
    except Exception as e:
        log.exception(f"OpenAI request failed for {model_name} via Chat Completions: {e}")
        return None

    if not resp or not resp.choices:
        log.error(f"Chat Completions empty response for {model_name}: {resp}")
        return None

    raw_answer = resp.choices[0].message.content
    if not raw_answer:
        log.error(f"Chat Completions no content for {model_name}")
        return None

    return raw_answer.strip()

def _call_comet_chat(model_name: str, system_prompt: str, final_user: str) -> Optional[str]:
    if not COMET_API_KEY:
        log.error("COMET_API_KEY missing, cannot call Comet LLM")
        return None

    if not _is_ascii(COMET_API_KEY):
        log.error("COMET_API_KEY has non-ASCII chars, abort LLM call")
        return None

    model_name = (model_name or COMET_LLM_MODEL).strip() or COMET_LLM_MODEL

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {COMET_API_KEY}",
    }

    try:
        resp = requests.post(
            f"{COMET_BASE}/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=80,
        )
    except Exception as e:
        log.exception("Comet LLM request exception: %s", e)
        return None

    if resp.status_code != 200:
        log.error(f"Comet LLM HTTP {resp.status_code}: {resp.text[:500]}")
        return None

    try:
        j = resp.json()
    except Exception:
        log.exception("Comet LLM non-JSON response: %s", resp.text[:500])
        return None

    try:
        choices = j.get("choices") or []
        if not choices:
            log.error(f"Comet LLM: empty choices in response {j}")
            return None
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            log.error(f"Comet LLM: no content in first choice {j}")
            return None
        return content.strip()
    except Exception as e:
        log.exception(f"Comet LLM parse error: {e}")
        return None

def _extract_blocks_from_model_answer(raw_answer: str) -> Dict[str, str]:
    import re
    text_blocks = re.findall(r"```(.*?)```", raw_answer, flags=re.S)

    lyrics_text = ""
    style_prompt = ""

    if len(text_blocks) >= 1:
        lyrics_text = text_blocks[0].strip()
    if len(text_blocks) >= 2:
        style_prompt = text_blocks[1].strip()

    if not lyrics_text:
        lyrics_text = raw_answer

    return {
        "lyrics": lyrics_text,
        "suno_prompt": style_prompt,
    }

def _negative_prompt_text() -> str:
    return (
        "bad low quality, mutated robotic voice, dirty poor mixing and mastering, "
        "bad low quality, noisy, slurred speech, lifeless, unnatural tone, low sampling rate, "
        "artificial grainy crackling cheap sound."
    )

import re as _re
def _collapse_ann_for_user(lyrics: str) -> str:
    def _repl(m):
        inside = m.group(1)
        head = inside.split("-", 1)[0].strip()
        head_low = head.lower()
        if head_low in ("verse","chorus","bridge","intro","outro","pre-chorus","post-chorus","pre chorus","post chorus"):
            canon = head_low.replace(" ", "-")
            return f"[{canon}]"
        first = head.split()[0].lower()
        return f"[{first}]"
    return _re.sub(r"\[(.*?)\]", _repl, lyrics)

def generate_song_pack(
    user_name: str,
    story: str,
    prev_lyrics: Optional[str],
    client_edit: Optional[str],
    use_comet_llm: bool,
    comet_llm_model: Optional[str],
) -> Dict[str, Any]:
    if prev_lyrics and client_edit:
        final_user_prompt = (
            "У клиента уже есть черновик песни. "
            "Твоя задача — ОБНОВИТЬ текст и Suno-промпт по правкам клиента. "
            "НЕ начинай заново. Сохрани историю, имена, вайб.\n\n"
            f"ИСТОРИЯ:\n{story}\n\n"
            f"ПРЕДЫДУЩИЙ ТЕКСТ:\n{prev_lyrics}\n\n"
            f"ПРАВКИ КЛИЕНТА:\n{client_edit}\n\n"
            "Выведи строго два блока: сначала 'ТЕКСТ ПЕСНИ:' + ```...```, затем пустая строка, затем "
            "'PROMPT ДЛЯ SUNO:' + ```...``` (англ, без имён), как указано в правилах."
        )
    else:
        final_user_prompt = (
            "Нужно написать ПЕРВУЮ версию песни по истории клиента.\n\n"
            f"ИСТОРИЯ:\n{story}\n\n"
            "Соблюдай формат: 'ТЕКСТ ПЕСНИ:' + ```...```, пустая строка, 'PROMPT ДЛЯ SUNO:' + ```...``` "
            "(англ, без имён), и в конце промпта добавь обязательный хвост качества."
        )

    raw_answer = None
    used_model = None

    if use_comet_llm and COMET_API_KEY and _is_ascii(COMET_API_KEY):
        model_for_comet = (comet_llm_model or COMET_LLM_MODEL).strip() or COMET_LLM_MODEL
        log.info(f"LLM: using Comet {model_for_comet} for lyrics generation")
        raw_answer = _call_comet_chat(model_for_comet, SYSTEM_PROMPT_BASE, final_user_prompt)
        used_model = f"{model_for_comet}@comet"
    else:
        log.info("LLM: using OpenAI Responses API directly (Comet disabled for this flow)")

    if raw_answer is None and OPENAI_API_KEY:
        if use_comet_llm:
            log.warning("LLM: Comet returned None or failed, fallback to OpenAI Responses API")
        raw_answer = _call_model_responses(PRIMARY_MODEL, SYSTEM_PROMPT_BASE, final_user_prompt)
        used_model = PRIMARY_MODEL

    if raw_answer is None and OPENAI_API_KEY:
        log.warning("LLM: Responses empty, fallback to OpenAI Chat Completions")
        raw_answer = _call_model_chat(FALLBACK_MODEL, SYSTEM_PROMPT_BASE, final_user_prompt)
        used_model = FALLBACK_MODEL

    if raw_answer is None:
        return {"ok": False, "error": "all_llm_failed"}

    parts = _extract_blocks_from_model_answer(raw_answer)
    lyrics_text  = parts.get("lyrics", "")
    style_prompt = parts.get("suno_prompt", "")

    neg_text = _negative_prompt_text()

    log.info(f"Song text generated with model {used_model}")

    return {
        "ok": True,
        "lyrics": lyrics_text,
        "suno_prompt": style_prompt,
        "suno_negative": neg_text,
        "raw": raw_answer,
        "used_model": used_model,
    }

# =========================================================
# FOXAI FLOW
# =========================================================

def _first_line_title(lyrics_text: str) -> str:
    if not lyrics_text:
        return "Custom Track"
    first_line = lyrics_text.strip().split("\n")[0].strip().replace("\r", " ")
    return first_line[:60] or "Custom Track"

def foxaihub_submit_compose(
    lyrics_text: str,
    style_prompt: str,
    negative_prompt: str,
    cuid: str,
) -> Dict[str, Any]:
    if not FOXAIHUB_API_KEY:
        log.error("FoxAIHub: missing FOXAIHUB_API_KEY")
        return {"ok": False, "error": "no_key"}

    headers = {
        "api-key": FOXAIHUB_API_KEY,
        "Content-Type": "application/json",
    }

    title_guess = _first_line_title(lyrics_text)

    body = {
        "title": title_guess,
        "conditions": [
            {
                "lyrics": lyrics_text if lyrics_text else "[Instrumental]",
                "strength": 0.5,
                "condition_start": 0,
                "condition_end": 1
            },
            {
                "prompt": style_prompt if style_prompt else "emotional modern pop, cinematic vibe",
                "strength": 0.5,
                "condition_start": 0,
                "condition_end": 1
            }
        ]
    }

    if negative_prompt:
        body["conditions"][1]["prompt"] = (
            body["conditions"][1]["prompt"].strip()
            + " | avoid: "
            + negative_prompt.strip()
        )

    try:
        resp = requests.post(
            f"{FOXAIHUB_BASE}/task",
            headers=headers,
            json=body,
            timeout=60,
        )
    except Exception as e:
        log.exception("FoxAIHub submit exception: %s", e)
        return {"ok": False, "error": "request_exception"}

    if resp.status_code != 200:
        log.error("FoxAIHub HTTP %s: %s", resp.status_code, resp.text[:500])
        return {"ok": False, "error": f"http_{resp.status_code}", "raw": resp.text[:1000]}

    try:
        j = resp.json()
    except Exception:
        log.exception("FoxAIHub non-JSON resp: %s", resp.text[:500])
        return {"ok": False, "error": "non_json_response"}

    if not j.get("success"):
        log.error("FoxAIHub generation failed: %s", j)
        return {"ok": False, "error": "generation_failed", "resp": j}

    task_id = j.get("task_id")
    if not task_id:
        log.error("FoxAIHub: no task_id in response %s", j)
        return {"ok": False, "error": "no_task_id", "resp": j}

    log.info(f"FoxAIHub task created: {task_id} for cuid={cuid}")
    return {"ok": True, "task_id": task_id}

def _collect_audio_urls_from_obj(obj) -> List[Dict[str, Any]]:
    results = []

    def norm_title(x):
        if isinstance(x, str) and x.strip():
            return x.strip()
        return "Track"

    def maybe_add(node, possible_audio_keys, possible_image_keys, title_keys=("title", "name")):
        if not isinstance(node, dict):
            return
        audio_url = None
        for k in possible_audio_keys:
            if k in node and isinstance(node[k], str) and node[k].startswith("http"):
                audio_url = node[k]
                break
        if not audio_url:
            return

        image_url = None
        for k in possible_image_keys:
            if k in node and isinstance(node[k], str) and node[k].startswith("http"):
                image_url = node[k]
                break

        title = None
        for tk in title_keys:
            if tk in node:
                title = node[tk]
                break

        results.append({
            "title": norm_title(title),
            "audio_url": audio_url,
            "image_url": image_url
        })

    AUDIO_KEYS = (
        "audio_url", "audio", "audioMp3", "audio_mp3", "mp3_url", "url", "download_url", "file", "file_url"
    )
    IMAGE_KEYS = ("image_url", "cover_url", "image", "cover")

    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            maybe_add(cur, AUDIO_KEYS, IMAGE_KEYS)
            for key, val in cur.items():
                if key in ("data", "result", "results", "clips", "items", "outputs"):
                    stack.append(val)
                elif isinstance(val, (dict, list)):
                    stack.append(val)
        elif isinstance(cur, list):
            for x in cur:
                stack.append(x)

    seen = set()
    uniq = []
    for r in results:
        au = r.get("audio_url")
        if au and au not in seen:
            seen.add(au)
            uniq.append(r)
    return uniq

def _extract_tracks_from_fox_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracks = []
    tracks.extend(_collect_audio_urls_from_obj(item))

    if "data" in item:
        tracks.extend(_collect_audio_urls_from_obj(item["data"]))

    for k in ("result", "results"):
        if k in item:
            tracks.extend(_collect_audio_urls_from_obj(item[k]))

    seen = set()
    uniq = []
    for t in tracks:
        au = t.get("audio_url")
        if au and au not in seen:
            seen.add(au)
            uniq.append(t)
    return uniq

def foxaihub_check_task(task_id: str) -> Dict[str, Any]:
    headers = {
        "api-key": FOXAIHUB_API_KEY,
    }

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    def _get_list_style():
        return requests.get(
            f"{FOXAIHUB_BASE}/task?ids={task_id}",
            headers=headers,
            timeout=80,
            verify=False,
        )

    def _get_single_style():
        return requests.get(
            f"{FOXAIHUB_BASE}/task/{task_id}",
            headers=headers,
            timeout=80,
            verify=False,
        )

    try:
        resp = _get_list_style()
    except requests.exceptions.ReadTimeout:
        log.warning(f"FoxAIHub check timeout for task {task_id}")
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        log.exception(f"FoxAIHub check exception: {e}")
        return {"ok": False, "error": "check_exception"}

    if resp.status_code != 200:
        log.error("FoxAIHub check HTTP %s: %s", resp.status_code, resp.text[:500])
        return {"ok": False, "error": f"http_{resp.status_code}"}

    try:
        j = resp.json()
    except Exception:
        log.exception("FoxAIHub check non-JSON: %s", resp.text[:500])
        return {"ok": False, "error": "non_json_response"}

    if isinstance(j, list) and j:
        item = j[0]
    elif isinstance(j, dict):
        item = j.get("item") or j.get("data") or j
    else:
        return {"ok": False, "error": "empty_list", "raw": j}

    status = str(item.get("status") or item.get("state") or "").lower().strip()

    tracks = _extract_tracks_from_fox_item(item)

    if tracks:
        if status in ("completed", "complete", "done", "success", "ok"):
            return {"ok": True, "ready": True, "tracks": tracks}
        else:
            log.info(f"FoxAIHub early links for task {task_id} while status={status}")
            return {"ok": True, "ready": True, "tracks": tracks}

    if status in ("pending", "queued", "processing", "running", "working", "generating"):
        return {"ok": True, "ready": False, "status": status}

    if status in ("completed", "complete", "done", "success", "ok"):
        try:
            resp2 = _get_single_style()
        except requests.exceptions.ReadTimeout:
            log.warning(f"FoxAIHub single check timeout for task {task_id}")
            return {"ok": False, "error": "timeout"}
        except Exception as e:
            log.exception("FoxAIHub single check exception: {e}")
            return {"ok": False, "error": "check_exception"}

        if resp2.status_code == 200:
            try:
                j2 = resp2.json()
            except Exception:
                j2 = None

            if isinstance(j2, dict):
                item2 = j2.get("item") or j2.get("data") or j2
            elif isinstance(j2, list) and j2:
                item2 = j2[0]
            else:
                item2 = j2

            tracks2 = _extract_tracks_from_fox_item(item2 or {})
            if tracks2:
                return {"ok": True, "ready": True, "tracks": tracks2}

        log.warning(f"FoxAIHub completed but no audio urls for task {task_id}")
        return {"ok": False, "error": "completed_but_no_audio_urls", "raw": item}

    return {"ok": False, "error": f"unknown_status_{status or 'none'}", "raw": item}

# =========================================================
# COMET FLOW (Suno v5)
# =========================================================

def comet_submit_music(
    lyrics_text: str,
    style_prompt: str,
    negative_prompt: str,
    cuid: str,
    mv: Optional[str] = None,
) -> Dict[str, Any]:
    if not COMET_API_KEY:
        log.error("Comet: missing COMET_API_KEY")
        return {"ok": False, "error": "no_key"}

    title_guess = _first_line_title(lyrics_text)

    # MV для этого запуска (premium / mini)
    mv_final = (mv or COMET_MODEL_VERSION).strip() or COMET_MODEL_VERSION

    tags = style_prompt or ""
    quality_marker = "high quality song"
    if quality_marker in tags:
        tags = tags.split(quality_marker)[0]
    tags = tags.strip()
    MAX_TAGS_LEN = 450
    if len(tags) > MAX_TAGS_LEN:
        log.info(f"Comet tags too long ({len(tags)}), truncating to {MAX_TAGS_LEN}")
        tags = tags[:MAX_TAGS_LEN].rstrip()

    payload = {
        "prompt": lyrics_text,
        "mv": mv_final,
        "title": title_guess,
        "tags": tags,
        "negative_tags": negative_prompt or "",
    }
    headers = {
        "Authorization": f"Bearer {COMET_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            f"{COMET_BASE}/suno/submit/music",
            headers=headers,
            json=payload,
            timeout=60,
        )
    except Exception as e:
        log.exception("Comet submit exception: %s", e)
        return {"ok": False, "error": "request_exception"}

    if resp.status_code != 200:
        log.error("Comet submit HTTP %s: %s", resp.status_code, resp.text[:500])
        return {"ok": False, "error": f"http_{resp.status_code}", "raw": resp.text[:1000]}

    log.info("Comet submit raw text: %s", resp.text[:500])

    try:
        j = resp.json()
    except Exception:
        log.exception("Comet non-JSON resp: %s", resp.text[:500])
        return {"ok": False, "error": "non_json_response", "raw": resp.text[:1000]}

    task_id = None

    if isinstance(j, str):
        task_id = j.strip().strip('"').strip()
        if not task_id:
            log.error(f"Comet: got empty string JSON response: {j!r}")
            return {"ok": False, "error": "empty_string_response", "raw": j}
        log.info(f"Comet task created (string JSON): {task_id} for cuid={cuid}")
        return {"ok": True, "task_id": task_id, "raw": j}

    if isinstance(j, dict):
        data_field = j.get("data")

        if isinstance(data_field, str) and data_field.strip():
            task_id = data_field.strip()
        elif isinstance(data_field, dict):
            task_id = data_field.get("task_id") or data_field.get("id")

        if not task_id:
            task_id = j.get("task_id") or j.get("id")

        if not task_id:
            log.error(f"Comet: no task_id in response {j}")
            return {"ok": False, "error": "no_task_id", "resp": j}

        log.info(f"Comet task created: {task_id} for cuid={cuid}")
        return {"ok": True, "task_id": task_id, "raw": j}

    log.error(f"Comet submit: unexpected JSON type {type(j)}: {j!r}")
    return {"ok": False, "error": "invalid_json_structure", "raw": j}

def comet_check_task(task_id: str) -> Dict[str, Any]:
    if not COMET_API_KEY:
        return {"ok": False, "error": "no_key"}

    headers = {
        "Authorization": f"Bearer {COMET_API_KEY}",
    }

    try:
        resp = requests.get(
            f"{COMET_BASE}/suno/fetch/{task_id}",
            headers=headers,
            timeout=80,
        )
    except requests.exceptions.ReadTimeout:
        log.warning(f"Comet check timeout for task {task_id}")
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        log.exception(f"Comet check exception: {e}")
        return {"ok": False, "error": "check_exception"}

    if resp.status_code != 200:
        log.error("Comet check HTTP %s: %s", resp.status_code, resp.text[:500])
        return {"ok": False, "error": f"http_{resp.status_code}"}

    log.info("Comet fetch raw for %s: %s", task_id, resp.text[:500])

    try:
        j = resp.json()
    except Exception:
        log.exception("Comet check non-JSON: %s", resp.text[:500])
        return {"ok": False, "error": "non_json_response"}

    root = j
    if isinstance(j, dict) and isinstance(j.get("data"), dict):
        root = j["data"]

    status_raw = ""
    if isinstance(root, dict):
        status_raw = root.get("status") or root.get("state") or root.get("task_status") or ""
    status_lower = str(status_raw).lower().strip()

    clips = None
    if isinstance(root, list):
        clips = root
    elif isinstance(root, dict):
        clips = root.get("data")

    tracks_info: List[Dict[str, Any]] = []

    if isinstance(clips, list):
        for entry in clips:
            if not isinstance(entry, dict):
                continue

            clip_status_raw = entry.get("status") or entry.get("state") or ""
            clip_status = str(clip_status_raw).lower().strip()

            audio_url = (
                entry.get("audio_url")
                or entry.get("audio_url_mp3")
                or entry.get("mp3_url")
                or entry.get("url")
            )

            image_url = (
                entry.get("image_url")
                or entry.get("image_large_url")
                or ""
            )

            title = (
                entry.get("title")
                or entry.get("display_name")
                or "Track"
            )

            duration = (
                entry.get("duration")
                or (entry.get("metadata") or {}).get("duration")
            )

            clip_id = entry.get("clip_id") or entry.get("id")

            tracks_info.append({
                "title": title,
                "audio_url": audio_url,
                "image_url": image_url,
                "duration": duration,
                "status": clip_status,
                "clip_id": clip_id,
            })

    complete_states = {"success", "succeeded", "complete", "completed", "done", "ok"}
    pending_states = {
        "in_progress",
        "running",
        "processing",
        "pending",
        "queued",
        "working",
        "generating",
        "not_start",   # Comet иногда так пишет, это нормальное “ещё не начал”
    }

    ready = False

    # Если корневой статус говорит, что задача завершена и есть треки — всё готово
    if status_lower in complete_states:
        if any(t.get("audio_url") for t in tracks_info):
            ready = True

    # Дополнительная проверка по самим клипам
    if not ready and tracks_info:
        for t in tracks_info:
            st = str(t.get("status") or "").lower().strip()
            if t.get("audio_url") and (st in complete_states or st == ""):
                ready = True
                break

    # Готово — возвращаем треки
    if ready and tracks_info:
        return {
            "ok": True,
            "ready": True,
            "status": status_lower or "success",
            "tracks": tracks_info,
        }

    # Явно ожидающие статусы или пустой статус без треков — просто ждём дальше
    if status_lower in pending_states or (not status_lower and not tracks_info):
        return {
            "ok": True,
            "ready": False,
            "status": status_lower or "pending",
            "tracks": tracks_info,
        }

    # ⚙️ ФОЛБЭК: неизвестный статус БЕЗ явных признаков фейла
    # Например: "UNKNOWN" — считаем, что задача ещё в процессе, и продолжаем опрашивать.
    if status_lower and status_lower not in complete_states and status_lower not in pending_states:
        # Если в названии статуса есть fail/error — считаем это реальным падением
        if any(x in status_lower for x in ("fail", "error")):
            return {
                "ok": False,
                "error": f"failed_status_{status_raw}",
                "raw": j,
            }

        # Иначе — просто логируем и ведём как pending
        log.warning(f"Comet: unknown non-terminal status '{status_raw}', treating as pending")
        return {
            "ok": True,
            "ready": False,
            "status": status_lower or "pending",
            "tracks": tracks_info,
        }

    # Если вообще ничего не поняли — жёсткая ошибка
    return {
        "ok": False,
        "error": f"unknown_status_{status_raw or 'none'}",
        "raw": j,
    }


# =========================================================
# POLLING
# =========================================================

def _send_tracks_to_user(cuid: str, provider: str, task_id: str, tracks: List[Dict[str,Any]]):
    send_message_to_bothelp_via_cuid(
        cuid,
        [{"content": "Держи свои песни❤️"}],
    )

    for i, t in enumerate(tracks, start=1):
        audio_url = t.get("audio_url")
        if not audio_url:
            continue

        variant_num = i
        title = f"Вариант {variant_num}"

        audio_bytes = None
        try:
            dl = requests.get(audio_url, timeout=180)
            if dl.status_code == 200:
                audio_bytes = dl.content
        except Exception:
            log.exception(f"download {provider} audio failed for task {task_id}")

        filename = f"song_variant_{variant_num}.mp3"

        att_id = None
        if audio_bytes:
            att_id = upload_audio_to_bothelp(audio_bytes, filename=filename)

        pretty_title = f"🎧 {title}"

        if att_id:
            send_message_to_bothelp_via_cuid(
                cuid,
                [{
                    "type": "attachment",
                    "attachment_id": att_id,
                    "content": pretty_title
                }],
            )
        else:
            send_message_to_bothelp_via_cuid(
                cuid,
                [{"content": f"{pretty_title}\n{audio_url}"}],
            )

        time.sleep(2)


def _poll_task_and_notify(task_id: str):
    task_info = PENDING_TASKS.get(task_id)
    if not task_info:
        return

    cuid        = task_info["cuid"]
    poll_count  = task_info["poll_count"]
    provider    = task_info["provider"]
    restarts    = task_info.get("restarts", 0)

    max_polls   = FOXAI_MAX_POLLS if provider == "foxai" else COMET_MAX_POLLS
    interval    = FOXAI_POLL_INTERVAL_SEC if provider == "foxai" else COMET_POLL_INTERVAL_SEC

    if poll_count >= max_polls:
        if restarts < 2:
            log.warning(f"{provider} task {task_id}: max polls reached (~6m). Auto-restart attempt #{restarts+1}.")
            send_message_to_bothelp_via_cuid(
                cuid,
                [{"content": "⏳ Трек долго собирается у провайдера. Перезапускаю генерацию заново — пришлю новый результат как будет готов 🙌"}],
            )
            PENDING_TASKS.pop(task_id, None)
            USER_STATE.setdefault(cuid, {}).setdefault("_autorest", 0)
            USER_STATE[cuid]["_autorest"] += 1
            task_info["restarts"] = restarts + 1
            return start_music_generation(cuid=cuid, force=True)
        else:
            log.warning(f"{provider} task {task_id}: max polls reached third time, giving up.")
            send_message_to_bothelp_via_cuid(
                cuid,
                [{"content": "⏳ Я трижды пытался дождаться аудио, но провайдер завис. Попробуй снова чуть позже или нажми «ГЕНЕРИРУЙ» ещё раз 🙏"}],
            )
            PENDING_TASKS.pop(task_id, None)
            return

    if provider == "foxai":
        status_res = foxaihub_check_task(task_id)
    else:
        status_res = comet_check_task(task_id)

    if status_res.get("ok") and status_res.get("ready"):
        tracks = status_res.get("tracks", [])
        if tracks:
            # задержка отправки треков = Type (5m/1h/...) от последней активности юзера
            st = USER_STATE.get(cuid, {}) if isinstance(USER_STATE.get(cuid, {}), dict) else {}
            delay = st.get("autoping_delay_sec") or 0
            if delay <= 0:
                _send_tracks_to_user(cuid, provider, task_id, tracks)
                log.info(f"{provider} task {task_id}: sent {len(tracks)} track(s) to cuid={cuid}")
            else:
                now_ts = time.time()
                last_activity = st.get("last_activity_ts") or now_ts
                desired_ts = last_activity + delay
                send_at_ts = desired_ts
                if send_at_ts <= now_ts:
                    # задержка уже прошла — отдать треки сразу
                    _send_tracks_to_user(cuid, provider, task_id, tracks)
                    log.info(
                        f"{provider} task {task_id}: delay={delay}s already passed since last_activity,"
                        f" sent {len(tracks)} track(s) to cuid={cuid} immediately"
                    )
                else:
                    _schedule_delayed_send(
                        task_id=task_id,
                        cuid=cuid,
                        provider=provider,
                        tracks=tracks,
                        send_at_ts=send_at_ts,
                    )
                    log.info(
                        f"{provider} task {task_id}: scheduling persisted delayed send of {len(tracks)} track(s)"
                        f" to cuid={cuid} at ts={int(send_at_ts)}"
                    )
        else:
            send_message_to_bothelp_via_cuid(
                cuid,
                [{"content": "⚠️ Музыка сгенерилась, но ссылки не пришли 😬"}],
            )
            log.warning(f"{provider} task {task_id}: completed but no tracks")
        PENDING_TASKS.pop(task_id, None)
        return

    if status_res.get("ok") and not status_res.get("ready"):
        PENDING_TASKS[task_id]["poll_count"] = poll_count + 1
        log.info(f"{provider} task {task_id}: still processing ({status_res.get('status')}), poll {poll_count+1}")
        t = threading.Timer(interval, _poll_task_and_notify, args=[task_id])
        t.daemon = True
        t.start()
        return

    soft_errors = ("timeout", "check_exception")
    if status_res.get("error") in soft_errors:
        PENDING_TASKS[task_id]["poll_count"] = poll_count + 1
        log.warning(f"{provider} task {task_id}: soft error {status_res.get('error')}, retrying (poll {poll_count+1})")
        t = threading.Timer(interval, _poll_task_and_notify, args=[task_id])
        t.daemon = True
        t.start()
        return

    if str(status_res.get("error","")).startswith("unknown_status_failed") \
       or str(status_res.get("error","")).startswith("failed") \
       or str(status_res.get("status","")).lower() == "failed":
        log.warning(f"{provider} task {task_id} FAILED on provider side")

        if task_info.get("restarts", 0) < 2:
            send_message_to_bothelp_via_cuid(
                cuid,
                [{"content": "⚠️ Провайдер выдал ошибку, перезапускаю генерацию 🙌"}],
            )
            PENDING_TASKS.pop(task_id, None)
            USER_STATE.setdefault(cuid, {}).setdefault("_autorest", 0)
            USER_STATE[cuid]["_autorest"] += 1
            return start_music_generation(cuid=cuid, force=True)

        send_message_to_bothelp_via_cuid(
            cuid,
            [{"content": "⚠️ Провайдер трижды вернул ошибку. Попробуй нажать «ГЕНЕРИРУЙ» снова 🙏"}],
        )
        PENDING_TASKS.pop(task_id, None)
        return

    errtxt = status_res.get("error", "unknown_error")
    log.error(f"{provider} task {task_id}: failed {errtxt}")
    send_message_to_bothelp_via_cuid(
        cuid,
        [{"content": "⚠️ Не удалось собрать трек 😞 Попробуй ещё раз позже."}],
    )
    PENDING_TASKS.pop(task_id, None)
    return


# =========================================================
# HIGH LEVEL
# =========================================================

def send_song_text_to_user(cuid: str, lyrics_text: str):
    user_view_lyrics = _collapse_ann_for_user(lyrics_text)
    msg = (
        "Твой текст песни готов 🎶\n\n"
        f"{user_view_lyrics}\n\n"
        "📝 Если хочешь что-то поправить — просто напиши правки одним сообщением сюда.\n\n"
        "Если всё ок — нажми кнопку «ГЕНЕРИРУЙ», и я соберу музыку 🎧"
    )
    send_message_to_bothelp_via_cuid(cuid, [{"content": msg}])

def send_waiting_music_msg(cuid: str, provider_name: str, task_id: str, style_prompt: str, negative_text: str, used_model: str):
    waiting_msg = (
        "🎧 Я начал генерацию аудио.\n"
        "Как только трек(и) будут готовы — я скину их сюда 🔥"
    )
    send_message_to_bothelp_via_cuid(cuid, [{"content": waiting_msg}])

    if SHOW_TECH_PROMPT_TO_USER:
        tech_reply = (
            "PROMPT ДЛЯ МУЗЫКИ (style_prompt):\n"
            f"{style_prompt}\n\n"
            "NEGATIVE ДЛЯ МУЗЫКИ:\n"
            f"{negative_text}\n\n"
            f"task_id={task_id}\n"
            f"lyrics_model={used_model}\n"
            f"provider={provider_name}\n"
            "Я слежу за процессом и пришлю ссылки/файлы, когда звук соберётся."
        )
        send_message_to_bothelp_via_cuid(cuid, [{"content": tech_reply}])

def start_music_generation(cuid: str, force: bool = False):
    st = USER_STATE.get(cuid)
    if not st:
        send_message_to_bothelp_via_cuid(
            cuid,
            [{"content": "Мне пока нечего озвучивать 😅 Пришли сначала историю 🙏"}],
        )
        return {"ok": False, "error": "no_state"}

    provider_name = st.get("provider")
    if provider_name not in ("comet", "foxai"):
        provider_name = "comet" if USE_COMET else "foxai"
        st["provider"] = provider_name

    poll_interval = COMET_POLL_INTERVAL_SEC if provider_name == "comet" else FOXAI_POLL_INTERVAL_SEC

    if not force:
        for existing_task_id, info in PENDING_TASKS.items():
            if info.get("cuid") == cuid:
                log.info(
                    f"start_music_generation: skip, already pending task {existing_task_id} "
                    f"for cuid={cuid} (provider={info.get('provider')})"
                )
                send_message_to_bothelp_via_cuid(
                    cuid,
                    [{"content": "Я уже собираю для тебя трек 🎧 Дождись результата, как будет готов — сразу скину 🙌"}],
                )
                return {
                    "ok": False,
                    "error": "already_generating",
                    "task_id": existing_task_id,
                    "provider": info.get("provider"),
                }

    if not force:
        with GENERATING_LOCK:
            if cuid in CURRENTLY_GENERATING:
                log.info(f"start_music_generation: cuid {cuid} is already in CURRENTLY_GENERATING, skip")
                send_message_to_bothelp_via_cuid(
                    cuid,
                    [{"content": "Уже запустил генерацию трека 🎧 Скоро всё прилетит, просто подожди чуть-чуть 🙌"}],
                )
                return {"ok": False, "error": "already_generating_lock"}
            CURRENTLY_GENERATING.add(cuid)

    lyrics_text   = st.get("lyrics","").strip()
    style_prompt  = st.get("suno_prompt","").strip()
    negative_text = st.get("negative","").strip()
    used_model    = st.get("used_model","").strip()

    if not lyrics_text:
        if not force:
            with GENERATING_LOCK:
                CURRENTLY_GENERATING.discard(cuid)
        send_message_to_bothelp_via_cuid(
            cuid,
            [{"content": "Мне пока нечего озвучивать 😅 Пришли сначала историю 🙏"}],
        )
        return {"ok": False, "error": "no_lyrics"}

    comet_mv = st.get("mv") or COMET_MODEL_VERSION

    def _try_generate_music():
        if provider_name == "comet":
            log.info(f"🎧 Using COMET / Suno v5 (mv={comet_mv})")
            return comet_submit_music(
                lyrics_text=lyrics_text,
                style_prompt=style_prompt,
                negative_prompt=negative_text,
                cuid=cuid,
                mv=comet_mv,
            )
        else:
            log.info("🎧 Using FoxAIHub (Suno v4-like)")
            return foxaihub_submit_compose(
                lyrics_text=lyrics_text,
                style_prompt=style_prompt,
                negative_prompt=negative_text,
                cuid=cuid,
            )

    try:
        max_retries = 3
        gen_res = None
        for attempt in range(1, max_retries + 1):
            gen_res = _try_generate_music()
            if gen_res.get("ok"):
                break
            err = gen_res.get("error", "unknown_error")
            log.warning(f"{provider_name} generation attempt {attempt} failed: {err}")
            time.sleep(3)

        if not gen_res or not gen_res.get("ok"):
            err_t = gen_res.get("error", f"{provider_name}_unknown_error") if gen_res else "unknown"
            tech_msg = (
                f"⚠️ Я попытался собрать тебе аудиотрек ({provider_name}), но генерация трижды упала 😞\n"
                f"Причина: {err_t}\n"
                "Я попробую ещё раз чуть позже, либо можешь просто нажать «ГЕНЕРИРУЙ» ещё раз 🙏"
            )
            send_message_to_bothelp_via_cuid(cuid, [{"content": tech_msg}])
            log.info(f"Scheduling auto-retry for {provider_name} after 30s (cuid={cuid}, mv={comet_mv})")
            t = threading.Timer(30, start_music_generation, args=[cuid])
            t.daemon = True
            t.start()
            return {"ok": False, "error": err_t}

        task_id = gen_res.get("task_id")

        send_waiting_music_msg(
            cuid=cuid,
            provider_name=provider_name,
            task_id=task_id,
            style_prompt=style_prompt,
            negative_text=negative_text,
            used_model=used_model,
        )

        if task_id:
            log.info(
                f"Starting polling thread for task {task_id} "
                f"(provider={provider_name}, lyrics_model={used_model}, mv={comet_mv})"
            )
            PENDING_TASKS[task_id] = {
                "cuid": cuid,
                "poll_count": 0,
                "provider": provider_name,
                "restarts": 0,
            }
            t = threading.Timer(poll_interval, _poll_task_and_notify, args=[task_id])
            t.daemon = True
            t.start()

        return {"ok": True, "task_id": task_id, "provider": provider_name, "used_model": used_model}

    finally:
        if not force:
            with GENERATING_LOCK:
                CURRENTLY_GENERATING.discard(cuid)

def handle_new_story(
    cuid: str,
    story_text: str,
    user_name: str,
    use_comet_llm: bool,
    provider_music: str,
    comet_mv: Optional[str],
    comet_llm_model: Optional[str],
):
    send_message_to_bothelp_via_cuid(
        cuid,
        [{"content": "✍️ Подбираю рифмы и ритм… дай пару секунд, сейчас будет черновик 🎶"}],
    )

    pack = generate_song_pack(
        user_name=user_name,
        story=story_text,
        prev_lyrics=None,
        client_edit=None,
        use_comet_llm=use_comet_llm,
        comet_llm_model=comet_llm_model,
    )
    if not pack.get("ok"):
        err_txt = pack.get("error", "model_failed")
        log.error("generate_song_pack (new) failed: %s", err_txt)
        send_message_to_bothelp_via_cuid(
            cuid,
            [{"content": "⚠️ Не смог понять запрос. Напиши простыми словами: для кого песня и какой вайб (грусть, кач, влюблённость, злость) 🙏"}],
        )
        return {"ok": False, "error": err_txt}

    lyrics_text   = pack["lyrics"]
    style_prompt  = pack["suno_prompt"]
    negative_text = pack["suno_negative"]
    used_model    = pack["used_model"]

    USER_STATE[cuid] = {
        "story": story_text,
        "lyrics": lyrics_text,
        "suno_prompt": style_prompt,
        "negative": negative_text,
        "used_model": used_model,
        "provider": provider_music,
        "use_comet_llm": use_comet_llm,
        "comet_llm_model": comet_llm_model,
        # last_activity_ts и autoping_delay_sec уже могут быть выставлены выше в _process_incoming_payload
        "last_activity_ts": USER_STATE.get(cuid, {}).get("last_activity_ts", time.time()),
        "autoping_delay_sec": USER_STATE.get(cuid, {}).get("autoping_delay_sec"),
        "autoping_message": USER_STATE.get(cuid, {}).get("autoping_message"),
    }
    if provider_music == "comet":
        USER_STATE[cuid]["mv"] = (comet_mv or COMET_MODEL_VERSION)

    send_song_text_to_user(cuid, lyrics_text)

    # если настроен Type → запланировать мягкий автопинг
    _schedule_autoping_if_needed(cuid)

    return {"ok": True, "stage": "draft_sent", "lyrics": lyrics_text}

def handle_edit_story(
    cuid: str,
    client_edit_text: str,
    user_name: str,
    use_comet_llm: bool,
    provider_music: str,
    comet_mv: Optional[str],
    comet_llm_model: Optional[str],
):
    st = USER_STATE.get(cuid)
    if not st:
        send_message_to_bothelp_via_cuid(
            cuid,
            [{"content": "Пришли сначала историю: для кого песня, какой вайб и какие имена ❤️"}],
        )
        return {"ok": False, "error": "no_state_for_edit"}

    story_text   = st.get("story","")
    prev_lyrics  = st.get("lyrics","")

    send_message_to_bothelp_via_cuid(
        cuid,
        [{"content": "🛠️ Вношу правки в текст… чуть-чуть магии — и пришлю новую версию ✨"}],
    )

    pack = generate_song_pack(
        user_name=user_name,
        story=story_text,
        prev_lyrics=prev_lyrics,
        client_edit=client_edit_text,
        use_comet_llm=use_comet_llm,
        comet_llm_model=comet_llm_model,
    )
    if not pack.get("ok"):
        err_txt = pack.get("error", "model_failed")
        log.error("generate_song_pack (edit) failed: %s", err_txt)
        send_message_to_bothelp_via_cuid(
            cuid,
            [{"content": "⚠️ Я чуть запутался в правках. Напиши простыми словами, что именно изменить 🙏"}],
        )
        return {"ok": False, "error": err_txt}

    lyrics_text   = pack["lyrics"]
    style_prompt  = pack["suno_prompt"]
    negative_text = pack["suno_negative"]
    used_model    = pack["used_model"]

    USER_STATE[cuid]["lyrics"]      = lyrics_text
    USER_STATE[cuid]["suno_prompt"] = style_prompt
    USER_STATE[cuid]["negative"]    = negative_text
    USER_STATE[cuid]["used_model"]  = used_model
    USER_STATE[cuid]["provider"]    = provider_music
    USER_STATE[cuid]["use_comet_llm"] = use_comet_llm
    USER_STATE[cuid]["comet_llm_model"] = comet_llm_model
    if provider_music == "comet":
        USER_STATE[cuid]["mv"] = (comet_mv or COMET_MODEL_VERSION)

    send_song_text_to_user(cuid, lyrics_text)

    # после обновлённого текста тоже можно мягко напомнить через Type
    _schedule_autoping_if_needed(cuid)

    return {"ok": True, "stage": "draft_updated", "lyrics": lyrics_text}

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

@app.before_request
def _delayed_restore_on_first_request():
    _restore_delayed_sends_once(_send_tracks_to_user)



@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "songbot",
        "primary_model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "use_comet": USE_COMET,
        "use_comet_llm": USE_COMET_LLM,
        "allow_unpaid": ALLOW_UNPAID,
        "bothelp_api": BOTHELP_API_BASE,
        "foxai_poll_interval_s": FOXAI_POLL_INTERVAL_SEC,
        "foxai_max_polls": FOXAI_MAX_POLLS,
        "comet_poll_interval_s": COMET_POLL_INTERVAL_SEC,
        "comet_max_polls": COMET_MAX_POLLS,
        "provider_default": "comet" if USE_COMET else "foxai",
        "user_state_len": len(USER_STATE),
        "pending_tasks_len": len(PENDING_TASKS),
        "delayed_tasks_len": len(DELAYED_TRACKS),
        "show_tech_prompt": SHOW_TECH_PROMPT_TO_USER,
        "comet_model_version": COMET_MODEL_VERSION,
        "comet_model_version_mini": MINI_COMET_MODEL_VERSION,
        "comet_llm_model_premium": COMET_LLM_MODEL_PREMIUM,
        "comet_llm_model_mini": COMET_LLM_MODEL_MINI,
    }), 200

def _process_incoming_payload(
    payload: Dict[str, Any],
    flow_name: str,
    use_comet_llm: bool,
    provider_music: str,
    comet_mv: Optional[str] = None,
    comet_llm_model: Optional[str] = None,
) -> Tuple[Dict[str, Any], int]:
    """
    Общая логика обработки вебхука BotHelp.
    provider_music:
      - "foxai"  -> дешевый тариф
      - "comet"  -> премиум / mini
    flow_name:
      - "main-basic"
      - "premium-v2"
      - "v1-mini"
    """
    tier = "premium" if provider_music == "comet" else "basic"

    cuid = str(payload.get("cuid") or "").strip()
    user_name = (payload.get("Имя клиента") or "Пользователь").strip()

    # ---------- TYPE (АВТОПИНГ) ----------
    type_val = payload.get("Type") or payload.get("type") or ""
    if isinstance(type_val, dict):
        type_val = json.dumps(type_val, ensure_ascii=False)
    type_val = str(type_val).strip()
    autoping_delay = _parse_autoping_delay(type_val) if type_val else 0

    # ---------- СЫРЫЕ ПОЛЯ ИСТОРИИ ----------
    base_form = payload.get("form") or ""
    if isinstance(base_form, dict):
        base_form = json.dumps(base_form, ensure_ascii=False)
    base_form = str(base_form).strip()

    base_dop = (
        payload.get("formdop")
        or payload.get("Formdop")
        or ""
    )
    if isinstance(base_dop, dict):
        base_dop = json.dumps(base_dop, ensure_ascii=False)
    base_dop = str(base_dop).strip()

    pro_form = payload.get("form2") or ""
    if isinstance(pro_form, dict):
        pro_form = json.dumps(pro_form, ensure_ascii=False)
    pro_form = str(pro_form).strip()

    pro_dop = (
        payload.get("formdop2")
        or payload.get("Formv2dop")
        or ""
    )
    if isinstance(pro_dop, dict):
        pro_dop = json.dumps(pro_dop, ensure_ascii=False)
    pro_dop = str(pro_dop).strip()

    mini_form = payload.get("form3") or ""
    if isinstance(mini_form, dict):
        mini_form = json.dumps(mini_form, ensure_ascii=False)
    mini_form = str(mini_form).strip()

    mini_dop = payload.get("formdop3") or ""
    if isinstance(mini_dop, dict):
        mini_dop = json.dumps(mini_dop, ensure_ascii=False)
    mini_dop = str(mini_dop).strip()

    fallback_text = (
        payload.get("text")
        or payload.get("last_prompt")
        or payload.get("message")
        or ""
    )
    if isinstance(fallback_text, dict):
        fallback_text = json.dumps(fallback_text, ensure_ascii=False)
    fallback_text = str(fallback_text).strip()

    # ---------- КНОПКИ / ПРАВКИ ----------
    compform = payload.get("compform") or ""
    if isinstance(compform, dict):
        compform = json.dumps(compform, ensure_ascii=False)
    compform = str(compform).strip()

    compform2 = payload.get("compform2") or ""
    if isinstance(compform2, dict):
        compform2 = json.dumps(compform2, ensure_ascii=False)
    compform2 = str(compform2).strip()

    compform3 = payload.get("compform3") or ""
    if isinstance(compform3, dict):
        compform3 = json.dumps(compform3, ensure_ascii=False)
    compform3 = str(compform3).strip()

    # ---------- ВЫБОР ИСТОЧНИКА ИСТОРИИ ----------
    story_source = "fallback"
    story_text = ""

    if flow_name == "v1-mini":
        # MINI: приоритет form3/formdop3, затем базовые
        merged_mini = mini_form
        if mini_dop:
            merged_mini = (merged_mini + "\n" + mini_dop).strip() if merged_mini else mini_dop

        if merged_mini:
            story_text = merged_mini
            story_source = "mini_fields"
        else:
            merged_base = base_form
            if base_dop:
                merged_base = (merged_base + "\n" + base_dop).strip() if merged_base else base_dop
            if merged_base:
                story_text = merged_base
                story_source = "base_fallback"
            elif fallback_text:
                story_text = fallback_text
                story_source = "fallback"

    elif provider_music == "comet":
        # Премиум: сначала form2/formdop2, fallback в базовые
        merged_pro = pro_form
        if pro_dop:
            merged_pro = (merged_pro + "\n" + pro_dop).strip() if merged_pro else pro_dop

        if merged_pro:
            story_text = merged_pro
            story_source = "pro_fields"
        else:
            merged_base = base_form
            if base_dop:
                merged_base = (merged_base + "\n" + base_dop).strip() if merged_base else base_dop
            if merged_base:
                story_text = merged_base
                story_source = "base_fallback"
            elif fallback_text:
                story_text = fallback_text
                story_source = "fallback"
    else:
        # Бюджетный: игнорируем form2/form3, берём только base
        merged_base = base_form
        if base_dop:
            merged_base = (merged_base + "\n" + base_dop).strip() if merged_base else base_dop

        if merged_base:
            story_text = merged_base
            story_source = "base_fields"
        elif fallback_text:
            story_text = fallback_text
            story_source = "fallback"

    log.info(
        f"[FLOW] via {flow_name}: tier={tier}, story_source={story_source}, "
        f"use_comet_llm={use_comet_llm}, provider_music={provider_music}, comet_mv={comet_mv or '-'}, comet_llm_model={comet_llm_model or '-'}"
    )
    log.info("[INCOMING RAW] %s", json.dumps(payload, ensure_ascii=False))

    if not cuid:
        log.warning("[INCOMING] no cuid in payload")
        return {"ok": False, "error": "no_cuid"}, 200

    user_has_state = cuid in USER_STATE and USER_STATE[cuid]

    # трекаем активность и конфиг автопинга
    now_ts = time.time()
    if not user_has_state:
        USER_STATE[cuid] = {}
    USER_STATE[cuid]["last_activity_ts"] = now_ts

    if autoping_delay > 0:
        prev_delay = USER_STATE[cuid].get("autoping_delay_sec")
        USER_STATE[cuid]["autoping_delay_sec"] = autoping_delay
        log.info(f"[AUTOPING] cuid={cuid} Type='{type_val}' -> delay={autoping_delay}s (was={prev_delay})")

    # Всегда обновляем настройки тарифа для существующего юзера
    if user_has_state:
        USER_STATE[cuid]["provider"] = provider_music
        USER_STATE[cuid]["use_comet_llm"] = use_comet_llm
        USER_STATE[cuid]["comet_llm_model"] = comet_llm_model
        if provider_music == "comet":
            USER_STATE[cuid]["mv"] = (comet_mv or COMET_MODEL_VERSION)

    incoming_new_story = False
    if story_text:
        if (not user_has_state) or (story_text.strip() != USER_STATE[cuid].get("story", "").strip()):
            incoming_new_story = True

    # CASE A: новая история
    if incoming_new_story:
        flow_res = handle_new_story(
            cuid=cuid,
            story_text=story_text,
            user_name=user_name,
            use_comet_llm=use_comet_llm,
            provider_music=provider_music,
            comet_mv=comet_mv,
            comet_llm_model=comet_llm_model,
        )
        return flow_res, 200

    if not user_has_state:
        send_message_to_bothelp_via_cuid(
            cuid,
            [{"content": "Привет 👋 Для кого песня и какой вайб? Напиши короткую историю ❤️"}],
        )
        return {"ok": False, "error": "no_story_yet"}, 200

    # CASE B: нажали «ГЕНЕРИРУЙ»
    if flow_name == "v1-mini":
        button_val = compform3 or compform2 or compform
    elif provider_music == "comet":
        button_val = compform2 or compform
    else:
        button_val = compform

    if button_val.upper().strip() == "ГЕНЕРИРУЙ":
        flow_res = start_music_generation(cuid=cuid)
        return flow_res, 200

    # CASE C: правки
    if flow_name == "v1-mini":
        edit_text = compform3 or compform2 or compform
    elif provider_music == "comet":
        edit_text = compform2 or compform
    else:
        edit_text = compform

    if edit_text:
        flow_res = handle_edit_story(
            cuid=cuid,
            client_edit_text=edit_text,
            user_name=user_name,
            use_comet_llm=use_comet_llm,
            provider_music=provider_music,
            comet_mv=comet_mv,
            comet_llm_model=comet_llm_model,
        )
        return flow_res, 200

    # CASE D: просто повторили историю/пинг
    if story_text:
        send_message_to_bothelp_via_cuid(
            cuid,
            [{"content": "Если всё ок — нажми «ГЕНЕРИРУЙ».\nЕсли хочешь что-то поправить — напиши что именно изменить ❤️"}],
        )
        return {"ok": True, "note": "repeat_story_no_changes"}, 200

    send_message_to_bothelp_via_cuid(
        cuid,
        [{"content": "Если всё ок — нажми «ГЕНЕРИРУЙ».\nЕсли хочешь что-то поправить — напиши что именно изменить ❤️"}],
    )
    return {"ok": True, "note": "no_changes"}, 200

@app.route("/", methods=["POST"])
def incoming_webhook():
    """
    ДЕШЁВЫЙ ТАРИФ (basic):
    - ВСЕГДА OpenAI + FoxAI
    - История берётся только из form/formdop, form2/form3 игнорируем
    """
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        log.exception(f"/ incoming invalid json: {e}")
        return jsonify({"ok": False, "error": "bad_json"}), 400

    resp_body, status = _process_incoming_payload(
        payload,
        flow_name="main-basic",
        use_comet_llm=False,     # дешёвый текст — через OpenAI
        provider_music="foxai",  # музыка — FoxAI
        comet_mv=None,
        comet_llm_model=None,
    )
    return jsonify(resp_body), status

@app.route("/v2", methods=["POST"])
def incoming_webhook_v2():
    """
    ПРЕМИУМ ТАРИФ:
    - Премиум поток: Comet GPT-5.1 + Suno v5 (Comet mv=COMET_MODEL_VERSION), если доступен,
      иначе аккуратный фоллбэк OpenAI + FoxAI.
    - История берётся преимущественно из form2/formdop2.
    """
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        log.exception(f"/v2 incoming invalid json: {e}")
        return jsonify({"ok": False, "error": "bad_json"}), 400

    comet_key_ok = bool(COMET_API_KEY) and _is_ascii(COMET_API_KEY)
    comet_llm_available = USE_COMET_LLM and comet_key_ok
    provider_music = "comet" if (USE_COMET and comet_key_ok) else "foxai"
    comet_mv = COMET_MODEL_VERSION if provider_music == "comet" else None

    resp_body, status = _process_incoming_payload(
        payload,
        flow_name="premium-v2",
        use_comet_llm=comet_llm_available,
        provider_music=provider_music,
        comet_mv=comet_mv,
        comet_llm_model=COMET_LLM_MODEL_PREMIUM,  # gpt-5.1
    )
    return jsonify(resp_body), status

@app.route("/v1", methods=["POST"])
def incoming_webhook_v1():
    """
    MINI ТАРИФ (вебхук /v1):
    - Текст: Comet GPT-5-all (если ключ есть; иначе fallback OpenAI).
    - Музыка: Comet Suno v5 с моделью chirp-auk (MINI_COMET_MODEL_VERSION), если Comet доступен,
      иначе fallback FoxAI.
    - История берётся из form3/formdop3, далее fallback в базовые form/formdop.
    """
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        log.exception(f"/v1 incoming invalid json: {e}")
        return jsonify({"ok": False, "error": "bad_json"}), 400

    comet_key_ok = bool(COMET_API_KEY) and _is_ascii(COMET_API_KEY)
    comet_llm_available = USE_COMET_LLM and comet_key_ok
    provider_music = "comet" if (USE_COMET and comet_key_ok) else "foxai"
    comet_mv = MINI_COMET_MODEL_VERSION if provider_music == "comet" else None

    resp_body, status = _process_incoming_payload(
        payload,
        flow_name="v1-mini",
        use_comet_llm=comet_llm_available,          # текст через Comet, если доступен
        provider_music=provider_music,              # музыка через Comет mini Suno или FoxAI
        comet_mv=comet_mv,
        comet_llm_model=COMET_LLM_MODEL_MINI,       # gpt-5-all для MINI-тарифа
    )
    return jsonify(resp_body), status

@app.route("/song", methods=["POST"])
def create_song():
    """
    Локальный тест:
    curl -X POST http://127.0.0.1:8080/song \
      -H "Content-Type: application/json" \
      -d '{"story":"хочу песню про саню","user_name":"гена","cuid":"local.test"}'
    """
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        log.exception(f"/song invalid json: {e}")
        return jsonify({"ok": False, "error": "bad_json"}), 400

    story = str(payload.get("story", "")).strip()
    user_name = str(payload.get("user_name", "Пользователь")).strip()
    cuid = str(payload.get("cuid", "local.test")).strip()

    if not story:
        return jsonify({"ok": False, "error": "no_story"}), 400

    send_message_to_bothelp_via_cuid(
        cuid,
        [{"content": "✍️ Подбираю рифмы и ритм… дай пару секунд, сейчас будет черновик 🎶"}],
    )

    provider_music = "comet" if USE_COMET else "foxai"
    comet_mv = COMET_MODEL_VERSION if provider_music == "comet" else None

    flow_res = handle_new_story(
        cuid=cuid,
        story_text=story,
        user_name=user_name,
        use_comet_llm=USE_COMET_LLM,
        provider_music=provider_music,
        comet_mv=comet_mv,
        comet_llm_model=COMET_LLM_MODEL_PREMIUM,  # как премиум-поток
    )
    return jsonify(flow_res), 200

@app.route("/suno_callback", methods=["POST"])
def suno_callback_compat():
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        payload = {}
    log.info("[SUNO CALLBACK COMPAT - IGNORED] %s", json.dumps(payload, ensure_ascii=False))
    return jsonify({"ok": True, "note": "callback disabled; using polling now"}), 200

# =========================================================
# ADMIN ROUTES
# =========================================================

def _check_admin_token():
    token = request.headers.get("X-Admin-Token") or request.args.get("token") or (request.get_json(silent=True) or {}).get("token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        abort(403)

@app.route("/admin/get_prompt", methods=["GET"])
def admin_get_prompt():
    _check_admin_token()
    cuid = (request.args.get("cuid") or "").strip()
    if not cuid or cuid not in USER_STATE:
        return jsonify({"ok": False, "error": "unknown_cuid"}), 404
    st = USER_STATE[cuid]
    return jsonify({
        "ok": True,
        "cuid": cuid,
        "story": st.get("story",""),
        "lyrics": st.get("lyrics",""),
        "suno_prompt": st.get("suno_prompt",""),
        "negative": st.get("negative",""),
        "used_model": st.get("used_model",""),
        "provider": st.get("provider",""),
        "use_comet_llm": st.get("use_comet_llm", False),
        "mv": st.get("mv",""),
        "comet_llm_model": st.get("comet_llm_model",""),
        "autoping_delay_sec": st.get("autoping_delay_sec"),
        "last_activity_ts": st.get("last_activity_ts"),
    }), 200

@app.route("/admin/retry_music", methods=["POST"])
def admin_retry_music():
    _check_admin_token()
    payload = request.get_json(silent=True) or {}
    cuid = str(payload.get("cuid") or "").strip()
    if not cuid or cuid not in USER_STATE:
        return jsonify({"ok": False, "error": "unknown_cuid"}), 404
    res = start_music_generation(cuid=cuid, force=True)
    return jsonify({"ok": True, "result": res}), 200


@app.route("/admin/list_tasks", methods=["GET"])
def admin_list_tasks():
    _check_admin_token()
    return jsonify({
        "ok": True,
        "pending_tasks": PENDING_TASKS,
        "delayed_tasks": DELAYED_TRACKS,
    }), 200


@app.route("/admin/force_send_ready", methods=["POST"])
def admin_force_send_ready():
    _check_admin_token()
    payload = request.get_json(silent=True) or {}
    target_cuid = str(payload.get("cuid") or "").strip()
    if not target_cuid:
        return jsonify({"ok": False, "error": "missing_cuid"}), 400

    sent = 0
    to_send = []

    with DELAYED_TRACKS_LOCK:
        for task_id, entry in list(DELAYED_TRACKS.items()):
            if entry.get("cuid") == target_cuid:
                to_send.append((task_id, entry))
                DELAYED_TRACKS.pop(task_id, None)
        _save_delayed_tracks()

    for task_id, entry in to_send:
        try:
            _send_tracks_to_user(entry["cuid"], entry["provider"], task_id, entry["tracks"])
            sent += 1
            log.info(
                f"Admin force_send_ready: sent delayed task {task_id} for cuid={entry['cuid']} (provider={entry['provider']})"
            )
        except Exception:
            log.exception(f"Admin force_send_ready failed for task {task_id}")

    return jsonify({"ok": True, "sent": sent, "cuid": target_cuid}), 200


# =========================================================
# BOOT
# =========================================================

if __name__ == "__main__":
    log.info(
        f"Server on 0.0.0.0:{PORT} | "
        f"ALLOW_UNPAID={ALLOW_UNPAID} | "
        f"USE_COMET={USE_COMET} | "
        f"USE_COMET_LLM={USE_COMET_LLM} | "
        f"COMET_MODEL_VERSION={COMET_MODEL_VERSION} | "
        f"MINI_COMET_MODEL_VERSION={MINI_COMET_MODEL_VERSION} | "
        f"COMET_LLM_MODEL_PREMIUM={COMET_LLM_MODEL_PREMIUM} | "
        f"COMET_LLM_MODEL_MINI={COMET_LLM_MODEL_MINI}"
    )
    _fetch_bothelp_token(force=True)
    _restore_delayed_sends_once(_send_tracks_to_user)
    try:
        from waitress import serve
        log.info(f"Starting waitress on 0.0.0.0:{PORT}")
        serve(app, host="0.0.0.0", port=PORT)
    except Exception:
        log.warning("waitress not available, using Flask dev server instead")
        app.run(host="0.0.0.0", port=PORT, debug=False)
