# 18 · cycle_loop + venn_compare × warm_editorial　【diagram · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。本步**一次产出两个 diagram 结构**（各含内嵌动画脚本）+ 复用视觉 warm_editorial。
> **不产出独立运动层**。编译时搭配通用叙事运动。
> mother_id = `18_cycle_warm`（以 cycle_loop 为成品验收），duration 默认 7.0s。

## 18A · 结构 generated/structures/cycle_loop/（family=diagram）
> 结构层必须内嵌默认数据的静态占位图形。

循环/周期机制镜头：`title` → 圆周排布的 node（含 `node-label`）+ 弧形 `edge`（首尾闭合，`marker` 箭头）→ `loop_note`。
- 数据：slot `nodes` JSON `[{"label":"蒸发"},{"label":"凝结"},{"label":"降水"},{"label":"径流"}]`
- slot_schema：`headline`≤18、`nodes`(json)、`loop_note`≤24
- default：headline=「水循环」、loop_note=「水在地球上永不停歇地循环」
- structure_card：use_for=[循环过程,闭环,自然循环]，spotlight_level=mid，duration_range=[6.0,9.0]

### 内嵌动画脚本
定义 `window.__hf_structure_init`：
- 读 nodes → 清空占位 → 按数量等角排布 node + 重建 SVG 弧线
- `node`：沿圆周依次出现（back.out，stagger，整体 -8°→0° 旋入）
- `edge`：弧线 `stroke-dashoffset` 顺时针生长
- `marker`：闭合处箭头弹现

## 18B · 结构 generated/structures/venn_compare/（family=diagram）
集合关系镜头：`title` → 两/三个交叠圆（每圆是一个 `cluster`，含 `node-label`），交集区放 `overlap_label` → `accent-line`。
- 圆用 `--accent`/`--accent-2` 低透明叠加
- slot_schema：`headline`≤18、`left_label`≤10、`right_label`≤10、`overlap_label`≤10

### 内嵌动画脚本
定义 `window.__hf_structure_init`：
- `cluster`：缩放渐显（power2.out，stagger 0.15）

## 运动层
**不产出独立运动**。编译时使用通用叙事运动（推荐 `parallax_pan`）。

### sfx_cues（写在 structure_card.json）
`[{"at":0.3,"sfx":"whoosh"},{"at":1.4,"sfx":"sparkle"}]`

## 编译命令
```bash
python assembler/assemble.py --structure cycle_loop --motion parallax_pan --visual warm_editorial \
  --slots generated/templates/18_cycle_warm/example_inputs.json --duration 7.0 \
  --out generated/templates/18_cycle_warm/index.html
```

## 验收
按 00 自检；分别 assemble cycle_loop 与 venn_compare 两个结构（均配通用运动 × warm_editorial）确认都能渲染、不溢出。
