# T19 · chart_callout（类别4 对比与证据展示）　【dataviz · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 chart_callout（family=dataviz，含内嵌动画脚本）**。
> **不产出独立运动**，图表生长 + 关键点标注内嵌在 `<script data-hf="structure-script">`。
> 编译时搭配通用叙事运动（推荐 `slide_in_hold`，仅负责 title）。
> 视觉层复用现有（推荐 `dark_tech` 或 `clean_saas`）。
> mother_id = `T19_chart_callout`，duration 默认 7.0s。先读 `references/data-in-motion.md` + `references/css-patterns.md`。

## 数据契约
- slot `series`（JSON）：折线数据点，default
  `[{"label":"1970","value":14.0},{"label":"1980","value":14.2},{"label":"1990","value":14.4},{"label":"2000","value":14.5},{"label":"2010","value":14.7},{"label":"2020","value":14.9}]`
- 结构层渲染**静态占位图**（坐标轴 + 占位折线/区域），不搭配运动时画面仍完整。
- 内嵌脚本读 `series` 生成 SVG path（折线 + 渐变面积），path draw 生长，末点 `chart-point` 高亮，`callout` 标注卡指向该点。

## 结构层 generated/structures/chart_callout/
> 结构层必须内嵌默认数据的静态占位图形，确保不搭配运动时画面仍完整可见。

图表（折线）+ 旁侧标注。上 `title` → 图表区（`chart-axis` 基线/网格 + `chart-line` 折线 + `chart-area` 填充 + 若干 `chart-point` + `chart-value`）+ 指向关键点的 `callout`（含 `callout_text`）→ 底 `source_note` + `accent-line`。
- anim-role：`title` / `chart-axis` / `chart-line` / `chart-area` / `chart-point`（每点）/ `chart-value`（每数值）/ `callout` / `accent-line`
- 折线 `stroke=var(--accent)`，面积渐变低透明；callout 卡用 `--stage-bg`/`--radius-card`，箭头指向末点
- slot_schema：`headline`≤18、`series`(json)、`callout_text`≤22、`source_note`≤24；`data_slots:["series"]`
- default：headline=「过去五十年全球平均气温变化」、callout_text=「1998年后升温速率明显加快」、source_note=「数据来源：NASA GISS」
- structure_card：use_for=[数据趋势,图表标注,关键拐点]，spotlight_level=high，duration_range=[5.5,8.5]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- `chart-axis`：scaleX 画出基线
- 读 `series` → 生成折线 path → `chart-line` 用 strokeDashoffset path draw 生长（power1.inOut）
- `chart-area`：跟随渐显；`chart-point` 逐点 pop；`chart-value` 渐显
- **趣味 motif（科普推荐）**：放一个 inline SVG **小火箭**（或小箭头）`marker`，让它**顺着折线 path 向上飞**（用 `MotionPathPlugin` 或按采样点逐段 `tl.to` 沿 path 前进 + 轻微抖动/尾焰），与 `chart-line` 的 path draw 同步推进；趋势下降段则换成「向下回落的小指标/落叶」。motif 元素也 `removeAttribute` 自管。
- `callout`：在折线收尾后从侧边滑入并指向末点，轻微 pulse 强调
- 完成后对所有 chart-*/marker/callout/accent-line `removeAttribute("data-anim-role")`，**不处理 title**

### sfx_cues（可选，趣味匹配）
`[{"at":0.4,"sfx":"rise","comment":"火箭顺折线上升"},{"at":1.8,"sfx":"pop","comment":"callout 指向末点"}]`（上升趋势用 `rise`；若为下降趋势改用 `whoosh`）

## 运动层
**不产出独立运动**。通用运动负责 title。

## 视觉层
复用 `generated/visuals/dark_tech/`。

## 编译命令
```bash
python assembler/assemble.py --structure chart_callout --motion slide_in_hold --visual dark_tech \
  --slots generated/templates/T19_chart_callout/example_inputs.json --duration 7.0 \
  --out generated/templates/T19_chart_callout/index.html
```

## 多样性变体矩阵（一个 prompt 产出 2 个模版，均 bound 各自内嵌脚本）
按 00「多样性变体矩阵」，产出 2 个空间布局变体（各独立 id，追加 `variant_of:"chart_callout"` + `variant_label`）：
- A｜`chart_callout` · 叠注型：图表满幅 + 关键点旁注卡浮在图上（「图上标注」）→ `T19_chart_callout`
- B｜`chart_callout_split` · 分栏型：左图表 + 右大标注/结论栏（「左图右注」）→ `T19_chart_callout_split`

## 验收
按 00 自检；确认 4/5/7 个点都不溢出图表区、callout 不遮挡折线、render 下 path draw 确定性收尾；viewer 自动跳过通用运动。
