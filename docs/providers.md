# Asset providers: plugging in your own footage and music

Everything in WevisionCut is self-contained except the two places that have to
reach into whatever media library you own: **footage retrieval** (one search per
shot) and **BGM retrieval** (one search per cut). Both are pluggable.

This document is the contract. Read the first section to pick an approach; read
the rest when you write a provider.

---

## 1. Choosing an approach

| You have | Use | Code needed |
|---|---|---|
| Nothing; just want it to run | `asset_retrieval_tool` (default) | none |
| An HTTP search service | `http_video` / `http_bgm` | none |
| A directory of media files | `local_folder_video` / `local_folder_bgm` | none |
| A database, vector index, internal RPC, paid stock API | a Python plugin | ~20 lines |

The two knobs:

```yaml
services:
  retrieval_provider: <name>   # footage
  bgm_provider: <name>         # music; defaults to retrieval_provider
```

They are independent, so mixing sources is fine — internal footage with
open-web music, for instance.

---

## 2. Bundled providers

### `asset_retrieval_tool` — open-web search

Footage from Bilibili and/or YouTube, BGM from Internet Archive netlabels
(CC-licensed, instrumental-only after a two-stage keyword filter). Standard
library only.

```yaml
services:
  retrieval_provider: asset_retrieval_tool
  asset_retrieval:
    video_sources: [bilibili]        # or [youtube, bilibili]
    bilibili_cookie: ${BILIBILI_COOKIE:-}
    youtube_api_key: ${YOUTUBE_API_KEY:-}
    auto_download: true              # needed for rendering
    max_download_height: 720
    bgm_sources: [internet_archive]
```

Two practical notes. Bilibili's search and playurl endpoints reject anonymous
requests with `HTTP 412`; paste a logged-in cookie or export `BILIBILI_COOKIE`
(treat it as a credential — it is one). And downloading needs `yt-dlp`, because
what search returns are page URLs, not media direct links:

```bash
pip install yt-dlp
```

YouTube needs a Data API v3 key: Google Cloud Console → enable *YouTube Data API
v3* → create an API key. The free quota is 10,000 units/day, about 100 per
search.

### `http_video` / `http_bgm` — your endpoint, config only

```yaml
services:
  retrieval_provider: http_video
  bgm_provider: http_bgm
  providers:
    http_video:
      url: https://footage.internal/search
      method: POST                  # or GET (params go in the query string)
      headers:
        Authorization: "Bearer ${FOOTAGE_TOKEN}"
      timeout_sec: 30
      query_field: query            # request field carrying the text query
      top_k_field: top_k
      duration_field: target_duration_sec
      extra_params:                 # merged into every request
        tenant: video-team
      results_path: data.items      # dotted path to the result array
      field_map:                    # canonical: your field name
        clip_path: playback_url
        duration_sec: seconds
```

The request body is JSON (`POST`/`PUT`) or a query string (`GET`). `${VAR}` in
`url` and `headers` is expanded from the environment, so tokens never have to sit
in the file.

`results_path` may be omitted when the response *is* the array; a response object
containing `results`, `items`, `data`, `candidates` or `list` is also detected
automatically. `field_map` is only needed for fields whose names are not already
recognised — see the alias table in §4.

### `local_folder_video` / `local_folder_bgm` — files on disk

```yaml
services:
  retrieval_provider: local_folder_video
  bgm_provider: local_folder_bgm
  providers:
    local_folder_video:
      roots: [/data/my_footage]
      extensions: [.mp4, .mov]      # optional; defaults cover the common ones
      require_match: false          # true = drop clips matching no query token
    local_folder_bgm:
      roots: [/data/my_music]
```

Scoring is lexical, over the file stem, the two nearest parent directory names,
and an optional sidecar. A sidecar sits next to the media with the same stem:

```
/data/my_footage/city_night_aerial.mp4
/data/my_footage/city_night_aerial.json
```

```json
{
  "caption": "Drone shot over a city at night, neon signage, slow push in",
  "tags": ["city", "night", "aerial", "drone"],
  "duration_sec": 42.0
}
```

A `.txt` sidecar is read as a plain caption. Supplying `duration_sec` is worth
it: without it, duration is probed with `ffprobe` per candidate.

Because matching is lexical, this is a good baseline and a poor semantic search.
For real retrieval quality, write a provider over an embedding index — §3.

---

## 3. Writing a Python provider

A provider is a function from a query object to a list of dicts.

```python
# my_providers.py
from pipeline.providers import (
    BgmQuery, VideoQuery,
    register_bgm_provider, register_video_provider,
)


@register_video_provider("my_footage")
def search_footage(query: VideoQuery) -> list[dict]:
    hits = my_index.search(
        query.query,
        k=query.top_k,
        min_seconds=query.min_duration_sec or None,
    )
    return [
        {
            "clip_path": hit.local_path,     # or an http(s) URL
            "score": hit.similarity,
            "duration_sec": hit.seconds,
            "description": hit.caption,
            "tags": hit.tags,
        }
        for hit in hits
    ]


@register_bgm_provider("my_music")
def search_music(query: BgmQuery) -> list[dict]:
    tracks = my_music_db.search(query.query, k=query.top_k)
    return [
        {"music_path": t.path, "score": t.score,
         "duration_sec": t.seconds, "name": t.title, "caption": t.artist}
        for t in tracks
    ]
```

Register it in the config:

```yaml
services:
  retrieval_provider: my_footage
  bgm_provider: my_music
  providers:
    plugins:
      - my_providers                        # importable module, or
      - /opt/wevisioncut/my_providers.py     # a file path
    my_footage:                             # arrives as query.options
      index_url: https://index.internal
      shard: production
```

Plugins are imported at startup and again on the first retrieval call, so
registration happens before anything needs it. An import that fails is logged and
skipped rather than fatal, and `run_studio.sh --check` reports it as a warning.

### Query objects

`VideoQuery`:

| Field | Meaning |
|---|---|
| `query` | The shot's `visual_query`, written by the policy model |
| `top_k` | How many candidates to return (already widened for filtering) |
| `target_duration_sec` | Length the clip must cover, from the real TTS duration. `0.0` = unconstrained |
| `min_duration_sec` | `target * min_duration_ratio` — shortest useful source |
| `max_duration_sec` | `target * max_loop_count` — longest useful source |
| `output_dir` | A writable directory reserved for this provider |
| `options` | Your `services.providers.<name>` block |
| `pipeline_config` | The whole projected pipeline config, if you need more |
| `shot` | The shot dict that triggered the search |

`BgmQuery` is the same minus the shot-specific fields, with
`target_duration_sec` being the total video duration the playlist must cover.

Return **more** than `top_k` when you can: candidates are filtered afterwards on
duration and orientation, so a provider that returns exactly `top_k` items can
end up contributing nothing.

### Guarantees

- A provider may raise. The exception is logged, treated as "no candidates", and
  the next backend in the cascade runs.
- Returning `[]` is fine and means the same thing.
- Malformed individual results are dropped with a warning; the rest survive.
- Providers are called from a worker thread, possibly several at once, so keep
  them thread-safe.
- Provider names may not shadow a built-in backend (`mmfinder*`,
  `asset_retrieval_tool`, `local_scan`); registration raises if they do.

---

## 4. Result schemas

Results pass through `normalize_video_clip` / `normalize_bgm_track`, which fill
in the canonical fields, coerce types, and preserve any extra keys you set (they
reach the frontend intact). You only have to supply what you actually know.

### Video clip

| Canonical field | Type | Accepted aliases |
|---|---|---|
| `clip_path` | str | `local_video_path`, `local_path`, `path`, `file`, `filepath`, `video_path`, `video_url`, `url`, `source_video`, `src` |
| `clip_id` | str | `id`, `video_id`, `asset_id` (auto-generated if absent) |
| `score` | float | `relevance`, `similarity` (defaults to descending by position) |
| `duration_sec` | float | `duration`, `length_sec`, `length`, `seconds` (0 → probed with ffprobe) |
| `description` | str | `caption`, `caption_zh`, `title`, `text`, `summary`, `name` |
| `tags` | list[str] | a bare string is wrapped |

`clip_path` may be a local path or an `http(s)` URL. If it is a URL, your
provider is responsible for downloading it — rendering needs a local file. Use
`query.output_dir` and put the resulting path in `clip_path` (the pattern
`asset_retrieval_tool` follows).

### BGM track

| Canonical field | Type | Accepted aliases |
|---|---|---|
| `music_path` | str | `local_path`, `audio_path`, `path`, `file`, `filepath`, `download_url`, `audio_url`, `url` |
| `track_id` | str | `id`, `music_id`, `asset_id` (auto-generated if absent) |
| `score` | float | `relevance`, `similarity` |
| `duration_sec` | float | `duration`, `length_sec`, `length`, `seconds` |
| `name` | str | `title`, `track`, `filename` |
| `caption` | str | `artist`, `description`, `album` |
| `cdn_path` | str | `download_url`, `url` |
| `rank` | int | position |

---

## 5. What happens to your results

### Footage

`pipeline/tools/retrieve_visual.py`, in order:

1. **Duration and orientation filter.** A candidate survives if its duration is
   within `[target * min_duration_ratio - tolerance, target * max_loop_count]`
   and, when `require_horizontal` is on, its aspect ratio is at least
   `min_aspect_ratio`. Missing durations and resolutions are probed with
   `ffprobe`. Survivors gain `orientation`, `resolution`, `aspect_ratio`,
   `loop_count`, `duration_shortfall_sec`.
2. **Selection.** With `select_with_llm: true` the policy model picks from the
   filtered candidates; otherwise the highest score wins.
3. **Cascade.** If nothing survives, the next backend in `backend_order` runs.
   With `allow_short_candidate_fallback: true`, the longest rejected candidates
   are used rather than failing the shot.

The knobs live under `services.visual_retrieval` and apply to every provider:

```yaml
services:
  visual_retrieval:
    top_k: 12
    return_candidates: 12
    select_with_llm: true
    require_horizontal: true
    min_aspect_ratio: 1.33
    min_duration_ratio: 0.55
    max_loop_count: 2.0
    duration_tolerance_sec: 1.5
    allow_short_candidate_fallback: false
```

### BGM

`pipeline/tools/music_search.py` greedily takes tracks by score until their
combined duration covers the video, respecting a minimum track count that scales
with length (`min_tracks_short_video` / `long` / `very_long`). Each selected
track gets `offset_sec` and `end_sec`. Tracks with unknown duration are assumed
to be 180 s and flagged `duration_estimated`. If the playlist still falls short,
the search repeats with a varied query.

---

## 6. Chaining providers

`backend_order` accepts registered provider names alongside the built-ins, and is
tried left to right until one yields surviving candidates. This only applies when
`retrieval_provider` is `mmfinder`; any other value replaces the list with that
single provider.

```yaml
services:
  retrieval_provider: mmfinder
  visual_retrieval:
    backend_order:
      - my_footage             # your index first
      - asset_retrieval_tool   # open web when it comes up empty
      - local_scan             # local library as a last resort
```

BGM uses a fixed cascade instead: the selected provider, then a local-directory
scan.

---

## 7. Testing a provider offline

```python
from pipeline.providers import VideoQuery, get_video_provider, load_provider_plugins

load_provider_plugins(["my_providers"])

results = get_video_provider("my_footage")(VideoQuery(
    query="drone shot over a city at night",
    top_k=8,
    target_duration_sec=9.0,
    min_duration_sec=5.0,
    output_dir="/tmp/probe",
    options={"index_url": "https://index.internal"},
))
print(len(results), results[0])
```

To check normalisation as the pipeline sees it, call
`run_video_provider("my_footage", query)` instead — same arguments, canonical
output, and errors swallowed the way they are in production.

To verify config wiring end to end:

```bash
python3 -m studio.config --summary     # lists registered video and bgm providers
python3 -m studio.config --check       # errors on an unknown provider name
```

`tests/test_providers.py` covers the registry, normalisation, and both bundled
providers; it is a reasonable template for testing your own.

---

## 8. Extension points in the source

| What | File | Function |
|---|---|---|
| Registry, schemas, normalisation, plugin loading | `pipeline/providers/registry.py` | — |
| HTTP adapter | `pipeline/providers/http_api.py` | `http_video_provider`, `http_bgm_provider` |
| Local-folder adapter | `pipeline/providers/local_folder.py` | `local_folder_video_provider`, `local_folder_bgm_provider` |
| Footage backend dispatch | `pipeline/tools/retrieve_visual.py` | `search_clips_with_backend` |
| Footage provider bridge | `pipeline/tools/retrieve_visual.py` | `_search_via_registered_provider` |
| BGM cascade | `pipeline/tools/music_search.py` | `search_music_with_backend` |
| BGM provider bridge | `pipeline/tools/music_search.py` | `_search_bgm_via_registered_provider` |
| Config projection | `studio/config.py` | `_build_pipeline_config`, `_validate_providers` |

Adding a provider requires no change to any of these — that is the point. They
are listed for readers who want to see how the dispatch works.
