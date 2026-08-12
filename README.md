# opus-mt-translate-api

Self-hosted translation API: Italian → English, French, Spanish, German.
CTranslate2 + [Opus-MT](https://huggingface.co/Helsinki-NLP) models, int8 on CPU.
Optional built-in test page.

Built for the case where the VPS is *already busy* — a few hundred MB of RAM and
two cores, not a dedicated inference box. No GPU, no API keys, no data leaving
the machine.

```bash
git clone https://github.com/nik2208/opus-mt-translate-api.git
cd opus-mt-translate-api
cp .env.example .env          # edit before first run
docker compose build          # ~5 min: converts the four models at build time
docker compose up -d
```

```bash
curl -s localhost:8000/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Il contratto decorre dal primo gennaio e si rinnova tacitamente."}'
```

```json
{
  "translations": { "en": "The contract runs from January 1 and is tacitly renewed.", "fr": "...", "es": "...", "de": "..." },
  "sentences": 1,
  "ms": 61
}
```

Request a subset with `"targets": ["de","fr"]`, or add `"segments": true` to get
the per-sentence arrays alongside the joined text.

| Endpoint | |
|---|---|
| `POST /translate` | translate; honours `API_KEY` and the rate limit |
| `GET /health` | status and which languages are currently resident |
| `GET /config` | what the test page needs to render itself |
| `GET /docs` | OpenAPI |
| `GET /` | test page, only when `ENABLE_UI=true` |

## The test page

`ENABLE_UI=true` serves a single static HTML file from the same container — no
second service, no build step, ~12 KB. It shows the four translations side by
side, and hovering any sentence highlights the matching sentence in every other
language.

That alignment is not decoration: this service splits the input into sentences
before translating, and the highlight is what that split actually produced. When
output looks wrong, the segmentation is usually why, and the page shows it
directly.

The page ships in the image either way; `ENABLE_UI` only decides whether it gets
mounted. With it off, `/` returns 404 and the API is untouched.

### Before you make it public

A public page is a public endpoint. The failure mode on a shared box is not a
surprise bill, it's the host falling over. Three settings exist for this:

- `API_KEY` — set it and every `/translate` call needs `X-API-Key`. The page
  grows a key field when it's set. Leave it empty only if you genuinely want an
  open endpoint.
- `RATE_LIMIT_RPM` — per client IP, default 20/min.
- `MAX_CONCURRENT` — translations in flight across the process, default 4. Past
  this, callers get a 503 instead of queueing and dragging the host down. This
  is the one that actually protects the box.

Set `TRUST_PROXY=true` when running behind a reverse proxy, so the rate limiter
reads `X-Forwarded-For` rather than the proxy's own IP. Leave it `false` when
the container is exposed directly — that header is trivially spoofed.

## Coolify

Deploy as a public repository, Docker Compose build pack. Set the environment
variables in Coolify's UI rather than committing a `.env`, and:

1. Uncomment `SERVICE_FQDN_OPUSMT_8000` in `docker-compose.yml`. Coolify fills in
   the generated domain and wires up its proxy and TLS.
2. Set `TRUST_PROXY=true` — you're behind Coolify's proxy.
3. Set `ENABLE_UI=true` and `API_KEY` together.

The published `ports:` entry is ignored by Coolify; it's there for everywhere
else. The first build takes a few minutes because the models are converted in
the build stage — subsequent deploys hit the layer cache unless `TARGETS_BUILD`
changes.

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

## Configuration

Everything is env-driven; `.env.example` documents each variable. The ones worth
knowing:

| Variable | Default | Notes |
|---|---|---|
| `ENABLE_UI` | `false` | Serve the test page at `/` |
| `API_KEY` | *(empty)* | Empty means an open endpoint |
| `RATE_LIMIT_RPM` | `20` | Per client IP. `0` disables |
| `MAX_CONCURRENT` | `4` | Translations in flight. `0` disables |
| `TRUST_PROXY` | `false` | Read `X-Forwarded-For` for rate limiting |
| `BIND_HOST` / `PORT` | `127.0.0.1` / `8000` | Where the port is published |
| `IDLE_TTL` | `300` | Seconds before unloading a language. `0` = never |
| `INTER_THREADS` | `2` | Concurrent requests |
| `INTRA_THREADS` | `1` | Threads inside a single translation |
| `BEAM_SIZE` | `2` | `1` is ~40% faster, `4` is marginally better |
| `MAX_CHARS` | `20000` | Request size ceiling |
| `CORS_ORIGINS` | *(empty)* | The built-in page is same-origin and needs none |

On a shared box, `mem_limit` and `memswap_limit` in `docker-compose.yml` are set
equal on purpose: that denies the container swap entirely. Swapping inference
doesn't slow it down, it stalls it — and drags the rest of the host's I/O with it.
Better that this container gets OOM-killed than the services that matter.

One uvicorn worker, also on purpose: concurrency is handled inside CTranslate2 via
`inter_threads`, while extra workers would each load their own copy of the models.

## Adding languages

Edit `TARGETS_BUILD` (build, space-separated) and `TARGETS` (runtime,
comma-separated) in `.env`, then rebuild. Available pairs are on
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
- Rate limiting is per-process and in-memory. Fine for one container; if you ever
  run several, move it to the proxy.

If you need style control or document-level coherence, this isn't the tool — an
instruction-tuned model is, at roughly 30× the memory.

## License

MIT. The Opus-MT models are CC-BY 4.0 and are downloaded at build time, not
redistributed here.
