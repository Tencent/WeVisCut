#!/usr/bin/env python3
"""Gold Template 编译器 —— 一套代码，把正交三层重组成合法 HyperFrames index.html。

输入: (structure_id, motion_id, visual_id, slots, duration)
读取:
    structures/<structure_id>/structure.html   (含 <style data-hf="structure-css"> 与 <div data-hf="structure-dom">)
    structures/<structure_id>/slot_schema.json  (生成 data-composition-variables + 默认值)
    motions/<motion_id>/motion.js                (挂 window.__hf_motions[<motion_id>])
    visuals/<visual_id>/tokens.css               (:root 令牌 + .stage/.bg-stage 皮肤)
输出:
    一个 standalone index.html（root 直接在 <body>，不用 <template>）

这是 §六 CONTRACT 的参考实现。Stage3 的固定编译器可直接调用本模块。
用法:
    python assemble.py --structure hook_title_intro --motion scale_pop_hold \\
        --visual dark_tech --slots example_inputs.json --duration 6 --out index.html
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Gold_Template/
GEN = ROOT / "generated"  # 所有生成产物在 generated/ 下

GSAP_CDN = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"

# 字体（CJK 不在 HyperFrames 内置列表，必须靠 fonts/ + @font-face 注入）
_FONT_FORMATS = {".woff2": "woff2", ".woff": "woff", ".ttf": "truetype", ".otf": "opentype"}
_FONT_WEIGHTS = {
    "thin": 100, "extralight": 200, "light": 300, "regular": 400, "normal": 400,
    "medium": 500, "semibold": 600, "demibold": 600, "bold": 700, "extrabold": 800, "black": 900,
}
_KNOWN_FAMILIES = {"NotoSansSC": "Noto Sans SC", "NotoSerifSC": "Noto Serif SC"}


def _camel_to_spaced(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).strip()


def _scan_font_faces() -> str:
    """扫描 Gold_Template/fonts/，为每个字体文件生成 @font-face（family/weight/format 自动推断）。"""
    fonts_dir = ROOT / "fonts"
    if not fonts_dir.is_dir():
        return ""
    faces = []
    for f in sorted(fonts_dir.iterdir()):
        fmt = _FONT_FORMATS.get(f.suffix.lower())
        if not fmt:
            continue
        left, _, wtok = f.stem.partition("-")
        family = _KNOWN_FAMILIES.get(left, _camel_to_spaced(left))
        weight = _FONT_WEIGHTS.get(wtok.lower(), 400)
        faces.append(
            f'@font-face{{font-family:"{family}";font-weight:{weight};font-style:normal;'
            f'font-display:block;src:url("fonts/{f.name}") format("{fmt}");}}'
        )
    return "\n".join(faces)

# 抽取结构片段里的两个块
_RE_STRUCT_CSS = re.compile(
    r'<style[^>]*data-hf=["\']structure-css["\'][^>]*>(.*?)</style>', re.S | re.I
)
_RE_STRUCT_DOM = re.compile(
    r'(<[^>]*data-hf=["\']structure-dom["\'][^>]*>.*</[^>]+>)', re.S | re.I
)
_RE_STRUCT_SCRIPT = re.compile(
    r'<script[^>]*data-hf=["\']structure-script["\'][^>]*>(.*?)</script>', re.S | re.I
)


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}（先用对应 prompt 生成该层）")
    return path.read_text(encoding="utf-8")


def _load_slot_schema(structure_id: str) -> dict:
    schema = json.loads(_read(GEN / "structures" / structure_id / "slot_schema.json"))
    return schema


def _extract_structure(structure_id: str) -> tuple[str, str, str]:
    raw = _read(GEN / "structures" / structure_id / "structure.html")
    script_m = _RE_STRUCT_SCRIPT.search(raw)
    structure_script = script_m.group(1).strip() if script_m else ""
    raw_clean = _RE_STRUCT_SCRIPT.sub("", raw)
    css_m = _RE_STRUCT_CSS.search(raw_clean)
    dom_m = _RE_STRUCT_DOM.search(raw_clean)
    if not dom_m:
        raise ValueError(
            f"{structure_id}/structure.html 缺少 <... data-hf=\"structure-dom\"> 块（违反 CONTRACT §4）"
        )
    structure_css = css_m.group(1).strip() if css_m else ""
    structure_dom = dom_m.group(1).strip()
    return structure_css, structure_dom, structure_script


# HyperFrames data-composition-variables 仅认这几种 type；Gold_Template 的抽象数据槽
# （如 type:"json" 的 series/nodes/edges）需在此映射成合法声明，否则 lint 会报
# invalid_composition_variables_declaration。映射策略：非法 type → "string"，
# 且把结构化默认值 JSON.stringify 成字符串；运动层读到后 JSON.parse 还原（沿用「运动读数据」范式）。
_HF_VAR_TYPES = {"string", "number", "color", "boolean", "enum"}


def _build_variables_attr(schema: dict, slots: dict) -> tuple[str, dict]:
    """生成 data-composition-variables JSON 数组 + 合并后的有效值（默认 ⊕ 覆盖）。
    数据槽（type 不在 HyperFrames 合法集内，如 json）：声明降级为 string，默认值序列化为字符串。"""
    decls = []
    effective = {}
    for slot_id, spec in (schema.get("slots") or {}).items():
        raw_type = spec.get("type", "string")
        default = slots.get(slot_id, spec.get("default", ""))
        if raw_type not in _HF_VAR_TYPES:
            # 数据槽：HyperFrames 不识别该 type → 用 string 承载，结构化值转 JSON 字符串
            decl_type = "string"
            decl_default = default if isinstance(default, str) else json.dumps(default, ensure_ascii=False)
        else:
            decl_type = raw_type
            decl_default = default
        decls.append({
            "id": slot_id,
            "type": decl_type,
            "label": spec.get("label", slot_id),
            "default": decl_default,
        })
        effective[slot_id] = decl_default
    return json.dumps(decls, ensure_ascii=False), effective


# ── SFX（可选，向后兼容）────────────────────────────────────────────────────
# 音效来自结构层/运动层的 *_card.json 里的 sfx_cues 数组（assembler 不强制）。
# 每条 cue: {"at": <镜头内相对秒>, "sfx": "<sfx_id>", "volume": <可选>, "dur": <可选>}
# 解析 sfx_id → 文件名 走 registry/sfx.json；找不到则回退 sfx/<id>.mp3。
# 没有任何 cue 时，_render_audio_tags 返回 ""，产物与未引入 SFX 时完全一致。

def _load_sfx_registry() -> dict:
    path = ROOT / "registry" / "sfx.json"
    if not path.exists():
        return {}
    try:
        reg = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for item in reg.get("sfx", []) or []:
        sid = item.get("id")
        if sid:
            out[sid] = item
    out["__track_base__"] = int(reg.get("track_index_base", 102))
    return out


def _read_card_sfx_cues(kind: str, comp_id: str) -> list:
    """读 structures/<id>/structure_card.json 或 motions/<id>/motion_card.json 的 sfx_cues。
    可选文件 / 可选字段，全部缺省时返回 []。"""
    fname = {"structure": "structure_card.json", "motion": "motion_card.json"}.get(kind)
    if not fname:
        return []
    path = GEN / (kind + "s") / comp_id / fname
    if not path.exists():
        return []
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    cues = card.get("sfx_cues")
    return list(cues) if isinstance(cues, list) else []


def _collect_sfx_cues(structure_id: str, motion_id: str, cli_cues: list | None) -> list:
    """合并来源：CLI 显式提供时独占（传 [] 即强制关掉所有音效）；
    CLI 为 None（缺省）时自动读 motion_card + structure_card 的 sfx_cues。"""
    if cli_cues is not None:
        return list(cli_cues)
    return _read_card_sfx_cues("motion", motion_id) + _read_card_sfx_cues("structure", structure_id)


def _render_audio_tags(cues: list, win_start: float, win_dur: float) -> str:
    """把 sfx_cues 渲染成 HyperFrames <audio> 音轨标签（与 narration/bgm 同协议）。
    cue.at 是镜头内相对秒；footage 叠加模式下绝对时间 = win_start + cue.at。
    空 cues → 返回 ""（不注入任何 audio，向后兼容）。"""
    if not cues:
        return ""
    reg = _load_sfx_registry()
    track_base = reg.get("__track_base__", 102)
    win_end = win_start + win_dur
    tags = []
    for i, cue in enumerate(cues):
        try:
            at = win_start + float(cue.get("at", 0.0))
        except (TypeError, ValueError):
            continue
        if at < win_start or at > win_end:   # 越界裁剪（CONTRACT §10）
            at = max(win_start, min(at, win_end))
        sid = str(cue.get("sfx", "")).strip()
        if not sid:
            continue
        meta = reg.get(sid, {}) if isinstance(reg, dict) else {}
        file = meta.get("file", f"{sid}.mp3")
        vol = cue.get("volume", meta.get("default_volume", 0.35))
        dur = cue.get("dur", meta.get("default_dur", 1.0))
        tags.append(
            f'<audio id="sfx-{i}" data-start="{at}" data-duration="{dur}" '
            f'data-track-index="{track_base + i}" src="sfx/{file}" data-volume="{vol}"></audio>'
        )
    return "\n".join(tags)


def _render_footage_chain(src: str, total: float, clip_len: float, media_from: float, loop: bool) -> str:
    """实拍底：在 track 0 上 chain 若干段 <video> 铺满 [0,total]。
    video 必须放在非 timed wrapper 内；框架负责播放/seek，CSS z-index 负责视觉层叠。"""
    import math

    safe_total = max(0.1, float(total))
    if not loop or clip_len <= 0:
        repeat_count = 1
        seg_len = safe_total
    else:
        seg_len = float(clip_len)
        repeat_count = max(1, int(math.ceil(safe_total / seg_len)))

    style = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;"
    tags = []
    for i in range(repeat_count):
        start = round(i * seg_len, 3)
        dur = round(min(seg_len, safe_total - start), 3)
        if dur <= 0:
            break
        media_attr = f' data-media-start="{media_from}"' if (i == 0 and media_from > 0) else ""
        tags.append(
            f'<video id="footage-{i}" data-start="{start}" data-duration="{dur}"{media_attr} '
            f'data-track-index="0" src="{src}" muted playsinline style="{style}"></video>'
        )
    scrim = (
        '<div class="footage-scrim" style="position:absolute;inset:0;z-index:1;pointer-events:none;'
        'background:linear-gradient(180deg,rgba(0,0,0,.10) 0%,rgba(0,0,0,.28) 100%);"></div>'
    )
    return (
        '<div class="bg-footage" style="position:absolute;inset:0;z-index:0;overflow:hidden;background:#000;">\n'
        + _indent("\n".join(tags), 2) + "\n"
        + _indent(scrim, 2) + "\n"
        + "</div>"
    )


def assemble(
    structure_id: str,
    motion_id: str,
    visual_id: str,
    slots: dict,
    duration: float,
    width: int = 1920,
    height: int = 1080,
    sfx_cues: list | None = None,
    bg_video: str | None = None,
    bg_clip_len: float = 0.0,
    bg_media_from: float = 0.0,
    bg_loop: bool = True,
    decor_start: float = 0.0,
    decor_duration: float | None = None,
) -> str:
    schema = _load_slot_schema(structure_id)
    structure_css, structure_dom, structure_script = _extract_structure(structure_id)
    tokens_css = _read(GEN / "visuals" / visual_id / "tokens.css")
    motion_js = _read(GEN / "motions" / motion_id / "motion.js")
    font_faces = _scan_font_faces()
    vars_attr, _effective = _build_variables_attr(schema, slots)

    footage_mode = bool(bg_video)
    if footage_mode:
        scene_start = max(0.0, min(float(decor_start), duration))
        scene_duration = float(decor_duration) if decor_duration is not None else duration - scene_start
        scene_duration = max(0.1, min(scene_duration, duration - scene_start))
    else:
        scene_start, scene_duration = 0.0, duration

    # SFX（可选）：合并 card 里的 sfx_cues（或 CLI 覆盖）→ <audio> 音轨；无 cue 即空串
    cues = _collect_sfx_cues(structure_id, motion_id, sfx_cues)
    audio_tags = _render_audio_tags(cues, scene_start, scene_duration)

    # data-composition-variables 写进属性，需转义单/双引号冲突：用单引号包 JSON，转义内部单引号
    vars_attr_html = vars_attr.replace("'", "&#39;")

    struct_script_block = ""
    if structure_script:
        struct_script_block = (
            f'    <!-- ===== Structure self-animation ({structure_id}) ===== -->\n'
            f'    <script>\n'
            f'{_indent(structure_script, 6)}\n'
            f'    </script>\n'
        )

    if footage_mode:
        fade_in = min(0.4, scene_duration * 0.15)
        fade_out = min(0.4, scene_duration * 0.15)
        fade_out_at = max(scene_start + fade_in + 0.1, scene_start + scene_duration - fade_out)
        motion_runtime = f"""
      const root = document.getElementById('root');
      const scene = document.getElementById('scene');
      scene.style.opacity = 0;
      const tl = gsap.timeline({{ paused: true }});
      tl.to(scene, {{ opacity: 1, duration: {fade_in}, ease: "power2.out" }}, {scene_start});
      tl.to(scene, {{ opacity: 0, duration: {fade_out}, ease: "power2.in" }}, {fade_out_at});
      if (window.__hf_structure_init) {{
        const structTl = gsap.timeline();
        window.__hf_structure_init(structTl, {{ gsap: gsap, root: root, duration: {scene_duration}, isFinal: false }});
        tl.add(structTl, {scene_start});
      }}
      if (window.__hf_motions && window.__hf_motions["{motion_id}"]) {{
        const decor = gsap.timeline();
        window.__hf_motions["{motion_id}"](decor, {{ gsap: gsap, root: root, duration: {scene_duration}, isFinal: false }});
        tl.add(decor, {scene_start});
      }}
      window.__timelines = window.__timelines || {{}};
      window.__timelines["root"] = tl;
        """.strip("\n")
    else:
        motion_runtime = f"""
      const root = document.getElementById('root');
      const tl = gsap.timeline({{ paused: true }});
      if (window.__hf_structure_init) {{
        window.__hf_structure_init(tl, {{ gsap: gsap, root: root, duration: {duration}, isFinal: false }});
      }}
      if (window.__hf_motions && window.__hf_motions["{motion_id}"]) {{
        window.__hf_motions["{motion_id}"](tl, {{ gsap: gsap, root: root, duration: {duration}, isFinal: false }});
      }}
      window.__timelines = window.__timelines || {{}};
      window.__timelines["root"] = tl;
        """.strip("\n")

    bg_block = (
        _render_footage_chain(bg_video or "", duration, bg_clip_len, bg_media_from, bg_loop)
        if footage_mode else
        '<div class="bg-stage" data-anim-role="bg"></div>'
    )
    scene_attrs = (
        f'data-start="{scene_start}" data-duration="{scene_duration}" data-track-index="2" style="position:relative;z-index:2;"'
        if footage_mode else
        f'data-start="0" data-duration="{duration}" data-track-index="1"'
    )

    runtime = f"""
      const V = window.__hyperframes.getVariables();
      document.querySelectorAll('[data-slot]').forEach(function (el) {{
        const k = el.getAttribute('data-slot');
        if (V[k] !== undefined && V[k] !== null) el.textContent = V[k];
      }});
{motion_runtime}
    """.strip("\n")

    html = f"""<!doctype html>
<html lang="zh-CN" data-composition-variables='{vars_attr_html}'>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <script src="{GSAP_CDN}"></script>
    <style>
      /* ===== fonts (CJK 必须本地注入；HyperFrames 内置列表为拉丁字体) ===== */
{_indent(font_faces, 6)}
      /* ===== reset ===== */
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ width: {width}px; height: {height}px; overflow: hidden; }}
      /* ===== centered translucent rounded stage panel (geometry fixed here, look from tokens) ===== */
      #scene {{ display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; }}
      .stage-panel {{
        position: relative; overflow: hidden;
        width: calc(100% - 320px); height: calc(100% - 200px);
        background: var(--stage-bg); border-radius: var(--radius-stage);
        box-shadow: var(--shadow-soft);
        -webkit-backdrop-filter: blur(var(--stage-blur, 0px)); backdrop-filter: blur(var(--stage-blur, 0px));
      }}
      /* ===== Layer 3: visual tokens ({visual_id}) ===== */
{_indent(tokens_css, 6)}
      /* ===== Layer 1: structure css ({structure_id}) ===== */
{_indent(structure_css, 6)}
    </style>
  </head>
  <body class="stage">
    <div id="root" data-composition-id="root" data-start="0" data-duration="{duration}" data-width="{width}" data-height="{height}">
{_indent(bg_block, 6)}
      <div id="scene" class="clip" {scene_attrs}>
        <div class="stage-panel">
{_indent(structure_dom, 10)}
        </div>
      </div>
{_audio_block(audio_tags)}    </div>
{struct_script_block}    <!-- ===== Layer 2: motion ({motion_id}) ===== -->
    <script>
{_indent(motion_js, 6)}
    </script>
    <script>
{_indent(runtime, 6)}
    </script>
  </body>
</html>
"""
    return html


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def _audio_block(audio_tags: str) -> str:
    """音轨块：有 SFX 时输出 <!-- AUDIO --> + 标签；无 SFX 时输出空串（向后兼容，零残留）。"""
    if not audio_tags.strip():
        return ""
    return "      <!-- ===== SFX audio tracks ===== -->\n" + _indent(audio_tags, 6) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold Template 三层重组编译器")
    ap.add_argument("--structure", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--visual", required=True)
    ap.add_argument("--slots", default="", help="JSON 文件路径或内联 JSON 字符串；缺省用 slot_schema 默认值")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--sfx-cues", default="", dest="sfx_cues",
                    help="可选。JSON 文件路径或内联 JSON 数组 [{at,sfx,volume?,dur?}]；"
                         "缺省时自动读 motion_card/structure_card 的 sfx_cues。传 '[]' 可强制关掉音效。")
    ap.add_argument("--bg-video", default="", dest="bg_video",
                    help="可选。视频底片路径（相对 Gold_Template 根），给定后动画作为 overlay 叠在循环视频上。")
    ap.add_argument("--bg-clip-len", type=float, default=0.0, dest="bg_clip_len",
                    help="视频底片素材时长（秒），用于循环 chain；0 表示不按素材时长切段。")
    ap.add_argument("--bg-media-from", type=float, default=0.0, dest="bg_media_from",
                    help="首段视频的 data-media-start（秒），默认 0。")
    ap.add_argument("--no-bg-loop", action="store_false", dest="bg_loop",
                    help="关闭视频底片循环，默认开启。")
    ap.add_argument("--decor-start", type=float, default=0.0, dest="decor_start",
                    help="overlay 动画开始时间（秒，相对完整视频），默认 0。")
    ap.add_argument("--decor-duration", type=float, default=None, dest="decor_duration",
                    help="overlay 动画持续时长（秒），默认到完整视频结束。")
    ap.set_defaults(bg_loop=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    slots: dict = {}
    if args.slots:
        p = Path(args.slots)
        slots = json.loads(p.read_text(encoding="utf-8")) if p.exists() else json.loads(args.slots)

    sfx_cues = None
    if args.sfx_cues:
        sp = Path(args.sfx_cues)
        sfx_cues = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else json.loads(args.sfx_cues)

    html = assemble(
        args.structure, args.motion, args.visual, slots, args.duration,
        sfx_cues=sfx_cues,
        bg_video=(args.bg_video or None),
        bg_clip_len=args.bg_clip_len,
        bg_media_from=args.bg_media_from,
        bg_loop=args.bg_loop,
        decor_start=args.decor_start,
        decor_duration=args.decor_duration,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    mode = (
        f"footage[{args.bg_video}] decor@{args.decor_start}+{args.decor_duration}"
        if args.bg_video else "standalone"
    )
    print(f"[assemble] {args.structure} × {args.motion} × {args.visual} ({mode}) -> {out}")


if __name__ == "__main__":
    main()
