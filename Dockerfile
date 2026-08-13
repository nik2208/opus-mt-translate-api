# ---------- stage 1: convert the models ----------
# Done at build time so the runtime container never needs network access
# and the image is fully reproducible.
# Base tags name a Debian release on purpose: the bare python:3.12-slim tag
# moves between Debian versions, and a glibc bump is exactly what broke
# ctranslate2 4.5.0 here (see VERSION PAIRING at the bottom).
FROM python:3.12-slim-trixie AS converter

# torch first, from the CPU-only index. --index-url and not --extra-index-url:
# the latter lets pip resolve torch from PyPI instead, dragging in ~2.5 GB of
# CUDA wheels this image never uses.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0

RUN pip install --no-cache-dir \
      ctranslate2==4.8.1 \
      transformers==4.57.1 \
      sentencepiece==0.2.0

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
      transformers==4.57.1 \
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

# ---------- VERSION PAIRING ----------
# The converter stage is a three-way constraint. Change one pin and you have to
# check the other two, or the build breaks in ways that look unrelated.
#
# 1. ctranslate2 >= 4.6.0
#    The 4.5.0 wheel ships a shared object marked as requiring an executable
#    stack, and glibc 2.41+ (Debian trixie) refuses to load it:
#      ImportError: cannot enable executable stack as shared object requires
#    Verified with readelf: 4.5.0 is RWE, 4.6.0+ is RW.
#
# 2. ctranslate2 and transformers must be from the same era
#    transformers renamed the from_pretrained kwarg torch_dtype -> dtype in
#    4.56, and the converter passes whichever name was current when it was
#    built. Mismatched, the kwarg falls through to the model constructor:
#      TypeError: MarianMTModel.__init__() got an unexpected keyword 'dtype'
#    ctranslate2 4.6.0-4.6.2 pass torch_dtype; 4.6.3+ pass dtype.
#
# 3. torch >= 2.6 whenever transformers >= 4.56
#    transformers gates torch.load behind a torch version check
#    (CVE-2025-32434). The Opus-MT checkpoints are .bin, not safetensors, so
#    that load path is unavoidable and an older torch fails with:
#      ValueError: ... we now require users to upgrade torch to at least v2.6
#
# Working combinations:
#   ctranslate2 4.8.1  + transformers 4.57.1 + torch 2.8.0   <- used here
#   ctranslate2 4.6.2  + transformers 4.46.3 + torch 2.5.1   <- older fallback
#
# torch only exists in this stage. The runtime image has no torch at all.
