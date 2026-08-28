# mmfinder — internal retrieval client (optional)

`mmfinder` is a thin client for an embedding-recall service that indexes a
private video and music corpus. The service is not publicly reachable, so this
directory is empty in the released repository.

It is documented because the released policy checkpoint was trained and
evaluated against it, and because the code paths remain in the pipeline:

- `pipeline/tools/retrieve_visual.py` — backends `mmfinder_video_cli`,
  `mmfinder_embedding_recall`, `mmfinder_subprocess`
- `pipeline/tools/music_search.py` — backends `mmfinder_music_cli`,
  `mmfinder_music_svrkit`
- `pipeline/tools/generate_audio.py` — an optional cloud TTS vendor client

## If you have the checkout

Place it so that this path exists:

```
external_tool/mmfinder/WeAgent_WeStoryCut/
```

or export `MMFINDER_ROOT=/path/to/WeAgent_WeStoryCut`. Then:

```yaml
services:
  retrieval_provider: mmfinder
  visual_retrieval:
    backend_order: [mmfinder_video_cli, mmfinder_embedding_recall, mmfinder_subprocess]
```

Relevant environment variables: `AICUT_MMFINDER_PYTHON` (interpreter for the
subprocess calls), `AICUT_MMFINDER_CLI_TIMEOUT`, `AICUT_MMFINDER_EMBED_TIMEOUT`,
`AICUT_MMFINDER_RECALL_TIMEOUT`, `AICUT_FINDER_SEARCH_TIMEOUT`.

## If you do not

You do not need it. The default configuration uses `asset_retrieval_tool`
(open-web search, bundled and dependency-free), and `docs/providers.md` shows how
to attach your own corpus — either with a config-only HTTP endpoint or with a
small Python plugin. Everything downstream of retrieval is provider-agnostic.
