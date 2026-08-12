"""
Opus-MT translation service — IT -> EN/FR/ES/DE
CTranslate2 int8 on CPU, lazy model loading with idle eviction.

Optional built-in test page at / when ENABLE_UI=true.
"""
import os
import re
import time
import threading
import logging
from contextlib import asynccontextmanager

import ctranslate2
import transformers
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
STATIC_DIR = os.getenv("STATIC_DIR", "/app/static")
TARGETS = set(os.getenv("TARGETS", "en,fr,es,de").split(","))
IDLE_TTL = int(os.getenv("IDLE_TTL", "300"))          # seconds before unloading
INTER_THREADS = int(os.getenv("INTER_THREADS", "2"))  # parallel requests
INTRA_THREADS = int(os.getenv("INTRA_THREADS", "1"))  # threads per translation
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "2"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "20000"))


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


ENABLE_UI = _flag("ENABLE_UI", "false")
API_KEY = os.getenv("API_KEY", "").strip()
TRUST_PROXY = _flag("TRUST_PROXY", "false")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "20"))    # per client IP; 0 = off
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "4"))     # whole process; 0 = off
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("opus-mt")

# --- sentence segmentation ------------------------------------------------
# Opus-MT is trained on single sentences and degrades hard on long paragraphs
# (and silently truncates past ~512 tokens). Split first, translate as a batch,
# rejoin. The negative lookbehinds cover the common Italian abbreviations that
# would otherwise create bogus sentence breaks.
_ABBREV = (
    r"(?<!\bSig\.)(?<!\bSig\.ra\.)(?<!\bDott\.)(?<!\bDott\.ssa\.)(?<!\bProf\.)"
    r"(?<!\bIng\.)(?<!\bAvv\.)(?<!\bon\.)(?<!\becc\.)(?<!\bpag\.)(?<!\bp\.)"
    r"(?<!\bart\.)(?<!\bcomma\.)(?<!\bn\.)(?<!\bnn\.)(?<!\bfig\.)(?<!\btab\.)"
    r"(?<!\bes\.)(?<!\bca\.)(?<!\bvs\.)(?<!\bS\.p\.A\.)(?<!\bS\.r\.l\.)"
)
_SPLIT_RE = re.compile(rf"{_ABBREV}(?<=[.!?…])[\"'»)\]]*\s+(?=[\"'«(\[]*[A-ZÀÈÉÌÒÙ0-9])")


def split_sentences(text: str) -> list[str]:
    """Chunk list where '' and '\\n' entries carry paragraph structure."""
    parts = []
    for block in text.split("\n"):
        if not block.strip():
            parts.append("")  # preserve blank lines / paragraph structure
            continue
        parts.extend(s for s in _SPLIT_RE.split(block) if s.strip())
        parts.append("\n")
    return parts


# --- model pool -----------------------------------------------------------
class ModelPool:
    """Loads models on first use, unloads them after IDLE_TTL of inactivity.

    On a RAM-constrained box this is the difference between ~550 MB resident
    for four languages and ~200 MB average.
    """

    def __init__(self):
        self._models: dict[str, tuple] = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def get(self, tgt: str):
        with self._lock:
            if tgt not in self._models:
                path = os.path.join(MODEL_DIR, f"it-{tgt}")
                if not os.path.isdir(path):
                    raise HTTPException(404, f"model it-{tgt} not installed")
                log.info("loading it-%s", tgt)
                translator = ctranslate2.Translator(
                    path,
                    device="cpu",
                    compute_type="int8",
                    inter_threads=INTER_THREADS,
                    intra_threads=INTRA_THREADS,
                )
                tokenizer = transformers.AutoTokenizer.from_pretrained(path)
                self._models[tgt] = (translator, tokenizer)
            self._last_used[tgt] = time.time()
            return self._models[tgt]

    def _reaper(self):
        while not self._stop.wait(30):
            if IDLE_TTL <= 0:
                continue  # 0 means "keep everything resident"
            now = time.time()
            with self._lock:
                stale = [k for k, t in self._last_used.items() if now - t > IDLE_TTL]
                for k in stale:
                    log.info("unloading it-%s (idle)", k)
                    self._models.pop(k, None)
                    self._last_used.pop(k, None)

    def start(self):
        threading.Thread(target=self._reaper, daemon=True).start()

    def stop(self):
        self._stop.set()

    def loaded(self) -> list[str]:
        with self._lock:
            return sorted(self._models)


pool = ModelPool()


# --- abuse control --------------------------------------------------------
# A public page means a public endpoint. On a shared box the failure mode is
# not a surprise bill, it's the host falling over — so cap both the per-client
# rate and the number of translations in flight.
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.rpm <= 0:
            return
        now = time.time()
        with self._lock:
            if len(self._hits) > 4096:  # cheap guard against unbounded growth
                self._hits = {
                    k: v for k, v in self._hits.items() if v and now - v[-1] < 60
                }
            hits = [t for t in self._hits.get(key, []) if now - t < 60]
            if len(hits) >= self.rpm:
                retry = int(60 - (now - hits[0])) + 1
                raise HTTPException(
                    429,
                    f"rate limit: {self.rpm} requests/min",
                    headers={"Retry-After": str(retry)},
                )
            hits.append(now)
            self._hits[key] = hits


limiter = RateLimiter(RATE_LIMIT_RPM)
inflight = threading.Semaphore(MAX_CONCURRENT) if MAX_CONCURRENT > 0 else None


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def guard(request: Request):
    if API_KEY:
        if request.headers.get("x-api-key", "") != API_KEY:
            raise HTTPException(401, "missing or invalid X-API-Key")
    limiter.check(client_ip(request))


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.start()
    log.info(
        "ready — targets=%s ui=%s auth=%s rpm=%s",
        sorted(TARGETS), ENABLE_UI, bool(API_KEY), RATE_LIMIT_RPM,
    )
    yield
    pool.stop()


app = FastAPI(title="opus-mt-translate-api", lifespan=lifespan)

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


# --- API ------------------------------------------------------------------
class TranslateIn(BaseModel):
    text: str = Field(..., min_length=1)
    targets: list[str] = Field(default_factory=lambda: sorted(TARGETS))
    segments: bool = Field(False, description="also return per-sentence output")


class TranslateOut(BaseModel):
    translations: dict[str, str]
    sentences: int
    ms: int
    source_segments: list[str] | None = None
    segments: dict[str, list[str]] | None = None


def translate_segments(payload: list[str], tgt: str) -> list[str]:
    translator, tokenizer = pool.get(tgt)
    tokenized = [
        tokenizer.convert_ids_to_tokens(
            tokenizer.encode(s, truncation=True, max_length=480)
        )
        for s in payload
    ]
    results = translator.translate_batch(
        tokenized,
        beam_size=BEAM_SIZE,
        max_batch_size=16,
        replace_unknowns=True,
    )
    return [
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True
        )
        for r in results
    ]


def rejoin(chunks: list[str], translated: list[str]) -> str:
    it = iter(translated)
    return "".join(next(it) + " " if c.strip() else c for c in chunks).strip()


@app.post("/translate", response_model=TranslateOut, dependencies=[Depends(guard)])
def translate(req: TranslateIn):
    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, f"text exceeds {MAX_CHARS} chars")
    bad = sorted(set(req.targets) - TARGETS)
    if bad:
        raise HTTPException(400, f"unsupported targets: {bad}")

    chunks = split_sentences(req.text)
    payload = [c for c in chunks if c.strip()]
    if not payload:
        raise HTTPException(400, "nothing to translate")

    if inflight and not inflight.acquire(timeout=20):
        raise HTTPException(503, "server busy, retry shortly")
    try:
        t0 = time.perf_counter()
        per_lang = {tgt: translate_segments(payload, tgt) for tgt in req.targets}
        ms = int((time.perf_counter() - t0) * 1000)
    finally:
        if inflight:
            inflight.release()

    return TranslateOut(
        translations={t: rejoin(chunks, segs) for t, segs in per_lang.items()},
        sentences=len(payload),
        ms=ms,
        source_segments=payload if req.segments else None,
        segments=per_lang if req.segments else None,
    )


@app.get("/health")
def health():
    return {"status": "ok", "loaded": pool.loaded(), "available": sorted(TARGETS)}


@app.get("/config")
def config():
    """What the test page needs in order to render itself."""
    return {
        "targets": sorted(TARGETS),
        "auth_required": bool(API_KEY),
        "max_chars": MAX_CHARS,
        "rate_limit_rpm": RATE_LIMIT_RPM,
    }


# Mounted last so /translate, /health and /config keep priority.
if ENABLE_UI:
    if os.path.isdir(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
        log.info("test page enabled at /")
    else:
        log.warning("ENABLE_UI=true but %s is missing — UI not mounted", STATIC_DIR)
