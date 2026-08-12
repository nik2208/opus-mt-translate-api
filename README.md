# opus-mt-translate-api

Self-hosted translation API: Italian → English, French, Spanish, German.
CTranslate2 + [Opus-MT](https://huggingface.co/Helsinki-NLP) models, int8 on CPU.

Built for the case where the VPS is *already busy* — a few hundred MB of RAM and
two cores, not a dedicated inference box. No GPU, no API keys, no data leaving
the machine.

```bash
git clone https://github.com/nik2208/opus-mt-translate-api.git
cd opus-mt-translate-api
docker compose build      # ~5 min: converts the four models at build time
docker compose up -d
```

```bash
curl -s localhost:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Il contratto decorre dal primo gennaio e si rinnova tacitamente."}'
```

```json
{
  "translations": {
    "en": "The contract runs from January 1 and is tacitly renewed.",
    "fr": "...",
    "es": "...",
    "de": "..."
  },
  "sentences": 1,
  "ms": 61
}
```

Request a subset with `"targets": ["de", "fr"]`. `GET /health` reports which
languages are currently resident.

## Footprint

| | |
|---|---|
| Docker image | ~700 MB |
| Models on disk | ~320 MB (4 × ~80 MB) |
| RSS, one language loaded | ~180 MB |
| RSS, four languages loaded | ~550 MB |
| RSS, average with `IDLE_TTL=300` | ~200-300 MB |
| Latency per sentence | 15-40 ms |

Three things keep it this small:

**torch is excluded from the runtime image.** Stage 1 uses torch to convert the
models; stage 2 copies only `/models` and runs CTranslate2 standalone. Installing
everything in a single stage produces a ~4 GB image.

**Models load on first use and unload when idle.** `IDLE_TTL` (default 300s)
controls eviction. Cold start for one language is 1-2s; set `IDLE_TTL=0` to keep
everything resident and trade ~550 MB for zero cold starts.

**Sentence segmentation before translation.** Opus-MT is trained on single
sentences and silently truncates past ~512 tokens, so feeding it whole paragraphs
degrades output badly. The service splits, translates the sentences as one batch
(much faster than N separate calls), and rejoins preserving line breaks. The
segmenter knows the common Italian abbreviations — `Sig.`, `art. 12`, `S.r.l.`,
`pag. 44` — that would otherwise cause bogus breaks.

## Tuning

| Variable | Default | Notes |
|---|---|---|
| `IDLE_TTL` | `300` | Seconds before unloading a language. `0` disables eviction |
| `INTER_THREADS` | `2` | Concurrent requests |
| `INTRA_THREADS` | `1` | Threads inside a single translation. Raise to 2 if requests are rare and long |
| `BEAM_SIZE` | `2` | `1` is ~40% faster, `4` is marginally better |
| `MAX_CHARS` | `20000` | Request size ceiling |
| `TARGETS` | `en,fr,es,de` | Must match the build arg |

On a shared box, `mem_limit` and `memswap_limit` in `docker-compose.yml` are set
equal on purpose: that denies the container swap entirely. Swapping inference
doesn't slow it down, it stalls it — and drags the rest of the host's I/O with it.
Better that this container gets OOM-killed than the services that matter.

One uvicorn worker, also on purpose: concurrency is handled inside CTranslate2 via
`inter_threads`, while extra workers would each load their own copy of the models.

## Adding languages

Edit `args.TARGETS` (build) and `environment.TARGETS` (runtime) in
`docker-compose.yml`, then rebuild. Available pairs are on
[Helsinki-NLP](https://huggingface.co/Helsinki-NLP) as `opus-mt-it-XX`. Where a
direct pair doesn't exist, `opus-mt-it-en` + `opus-mt-en-XX` in cascade almost
always does — at the cost of a pivot through English.

## Limitations

This is classical NMT, not an LLM. Know what you're giving up:

- Sentence-by-sentence: no terminology consistency across a long document.
- No instructions — you can't ask for a formal register or tell it to leave
  product names untranslated.
- The segmenter is regex-based. For text with unusual abbreviations, swap
  `split_sentences` for [blingfire](https://github.com/microsoft/BlingFire) or
  [sentencex](https://github.com/wikimedia/sentencex) (+~5 MB).
- Sentences over 480 tokens are truncated.

If you need style control or document-level coherence, this isn't the tool — an
instruction-tuned model is, at roughly 30× the memory.

## License

MIT. The Opus-MT models are CC-BY 4.0 and are downloaded at build time, not
redistributed here.
