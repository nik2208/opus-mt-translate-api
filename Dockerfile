# ---------- stage 1: convert the models ----------
# Done at build time so the runtime container never needs network access
# and the image is fully reproducible.
# Base tags pinned to a Debian release on purpose: the bare python:3.12-slim
# tag moves between Debian versions, and a glibc bump is exactly what broke
# ctranslate2 4.5.0 here (see the version note below).
FROM python:3.12-slim-trixie AS converter

# ctranslate2 must be >= 4.6.0. The 4.5.0 wheel ships a shared object marked
# as requiring an executable stack, which glibc 2.41+ (Debian trixie) refuses
# to load:
#   ImportError: libctranslate2-*.so.4.5.0: cannot enable executable stack
# Upstream cleared the flag in 4.6.0 — verified with readelf: 4.5.0 is RWE,
# 4.6.0+ is RW.
RUN pip install --no-cache-dir \
      ctranslate2==4.8.1 \
      transformers==4.46.3 \
      sentencepiece==0.2.0 \
      torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu

ARG TARGETS="en fr es de"
RUN set -eux; \
    for L in $TARGETS; do \
      ct2-transformers-converter \
        --model Helsinki-NLP/opus-mt-it-$L \
        --quantization int8 \
        --output_dir /models/it-$L; \
      python -c "\
import transformers, sys; \
transformers.AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-it-'+sys.argv[1]).save_pretrained('/models/it-'+sys.argv[1])" $L; \
    done

# ---------- stage 2: runtime ----------
# torch is NOT installed here — CTranslate2 runs standalone, which keeps the
# final image around 700 MB instead of ~4 GB.
FROM python:3.12-slim-trixie

RUN pip install --no-cache-dir \
      ctranslate2==4.8.1 \
      transformers==4.46.3 \
      sentencepiece==0.2.0 \
      fastapi==0.115.6 \
      "uvicorn[standard]==0.34.0"

COPY --from=converter /models /models
# The test page ships in the image but is only served when ENABLE_UI=true.
# It is ~12 KB of static HTML, so there is no separate build for the API-only case.
COPY app /app

ENV MODEL_DIR=/models \
    STATIC_DIR=/app/static \
    TARGETS=en,fr,es,de \
    ENABLE_UI=false \
    IDLE_TTL=300 \
    INTER_THREADS=2 \
    INTRA_THREADS=1 \
    OMP_NUM_THREADS=2

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

# One worker only: CTranslate2 handles concurrency internally via inter_threads,
# and extra uvicorn workers would each load their own copy of the models.
CMD ["uvicorn", "main:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
