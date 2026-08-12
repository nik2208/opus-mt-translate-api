"""
Opus-MT translation service — IT -> EN/FR/ES/DE
CTranslate2 int8 on CPU, lazy model loading with idle eviction.
"""
import os
import re
import time
import threading
import logging
from contextlib import asynccontextmanager

import ctranslate2
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
TARGETS = set(os.getenv("TARGETS", "en,fr,es,de").split(","))
IDLE_TTL = int(os.getenv("IDLE_TTL", "300"))          # seconds before unloading
INTER_THREADS = int(os.getenv("INTER_THREADS", "2"))  # parallel requests
INTRA_THREADS = int(os.getenv("INTRA_THREADS", "1"))  # threads per translation
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "2"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "20000"))

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
                continue
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.start()
    yield
    pool.stop()


app = FastAPI(title="opus-mt", lifespan=lifespan)


# --- API ------------------------------------------------------------------
class TranslateIn(BaseModel):
    text: str = Field(..., min_length=1)
    targets: list[str] = Field(default_factory=lambda: sorted(TARGETS))


class TranslateOut(BaseModel):
    translations: dict[str, str]
    sentences: int
    ms: int


def translate_one(text: str, tgt: str) -> str:
    translator, tokenizer = pool.get(tgt)
    chunks = split_sentences(text)
    payload = [c for c in chunks if c.strip()]
    if not payload:
        return text

    tokenized = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(s, truncation=True, max_length=480))
        for s in payload
    ]
    results = translator.translate_batch(
        tokenized,
        beam_size=BEAM_SIZE,
        max_batch_size=16,
        replace_unknowns=True,
    )
    decoded = iter(
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True
        )
        for r in results
    )
    return "".join(next(decoded) + " " if c.strip() else c for c in chunks).strip()


@app.post("/translate", response_model=TranslateOut)
def translate(req: TranslateIn):
    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, f"text exceeds {MAX_CHARS} chars")
    bad = set(req.targets) - TARGETS
    if bad:
        raise HTTPException(400, f"unsupported targets: {sorted(bad)}")

    t0 = time.perf_counter()
    out = {tgt: translate_one(req.text, tgt) for tgt in req.targets}
    return TranslateOut(
        translations=out,
        sentences=len([c for c in split_sentences(req.text) if c.strip()]),
        ms=int((time.perf_counter() - t0) * 1000),
    )


@app.get("/health")
def health():
    return {"status": "ok", "loaded": pool.loaded(), "available": sorted(TARGETS)}
