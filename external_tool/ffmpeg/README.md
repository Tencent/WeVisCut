# ffmpeg / ffprobe

Place a **static** `ffmpeg` and `ffprobe` in this directory:

```
external_tool/ffmpeg/ffmpeg
external_tool/ffmpeg/ffprobe
```

`bash scripts/setup_external_tools.sh` does this for you. Manually:

```bash
cd external_tool/ffmpeg
curl -L -o ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ffmpeg.tar.xz --strip-components=1 --wildcards '*/ffmpeg' '*/ffprobe'
chmod +x ffmpeg ffprobe && rm ffmpeg.tar.xz
./ffmpeg -version
```

macOS or arm64: use your platform's build (`brew install ffmpeg`) and point
`services.ffmpeg.*` at it, or export `WEDIRECTOR_FFMPEG` / `WEDIRECTOR_FFPROBE`.

## Why static

The pipeline shells out to ffmpeg from several tools and from subprocesses that
inherit a modified `LD_LIBRARY_PATH`. A dynamically linked binary copied between
machines fails at load time on the first missing `.so`, usually deep inside a
render step where the error is hard to read. A static build removes that class
of failure entirely.

Any reasonably recent build works; the reference deployment uses 7.0.2. The
encoder must support `libx264` and `aac`, which every mainstream build does.
