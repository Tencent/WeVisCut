# Subtitle fonts

Not committed — fonts are large and separately licensed. ~35 MB total.

`bash scripts/setup_external_tools.sh fonts` installs the default one here.

## What is needed

Burned-in subtitles are drawn by ffmpeg's `drawtext` filter, which needs a real
font file, and template layers reference font families by name in CSS. The
default is a CJK serif bold:

```
external_tool/gold_template/fonts/NotoSerifSC-Bold.otf
```

This is the path `render.subtitles.fontfile` resolves to. `run_studio.sh --check`
warns if it is absent; rendering then falls back to whatever ffmpeg finds on the
system, and CJK text often comes out as boxes.

## Manual install

```bash
mkdir -p external_tool/gold_template/fonts
cd external_tool/gold_template/fonts

# Noto Serif CJK SC Bold (SIL Open Font License 1.1)
curl -L -o NotoSerifSC-Bold.otf \
  https://github.com/notofonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Bold.otf
```

Any OTF or TTF works. To use one you already have, point the config at it
instead of copying files around:

```yaml
render:
  subtitles:
    fontfile: /usr/share/fonts/truetype/my-font.ttf
```

or:

```bash
export WEDIRECTOR_SUBTITLE_FONT=/usr/share/fonts/truetype/my-font.ttf
```

For a non-CJK deployment, any Latin font is fine and much smaller — Noto's CJK
coverage is the only reason the default is ~35 MB.

## Licensing

Noto fonts are under the SIL Open Font License 1.1, which permits redistribution
and embedding in rendered video. Check the terms of any font you substitute:
some commercial licences forbid exactly this use.
