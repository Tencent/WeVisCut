# 19 · icon_overview 图标概要展示　【icon 家族 · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。本步产出一个**图标概要展示结构** `icon_overview`，含内嵌的图标弹现/强调动画。
> **不产出独立运动层**。编译时搭配通用叙事运动（推荐 `scale_pop_hold`）。
> mother_id = `19_icon_demo`，duration 默认 5.5s。

## 设计原则
这是一个**多面板布局**的结构，展示多个图标+关键指标的概要画面：
- **不是单个居中面板**，而是在画面上分散布局多个小信息卡片
- 每个卡片含：一个 inline SVG 语义图标 + 一个关键数字/标签
- 卡片用 `position:absolute` 或 CSS grid 放置在画面不同位置
- 整体有一个 `title` 作为概要标题

## 结构层 generated/structures/icon_overview/
### 布局方案（多面板）
画面分为：
- 顶部：`title`（概要标题）
- 主体：3~4 个 icon 卡片散布在画面中，每个含：
  - `<svg data-anim-role="icon">` 语义图标（如放大镜、灯泡、心形、齿轮）
  - `<span data-anim-role="stat">` 关键数字
  - `<span data-anim-role="caption">` 简短说明
- 底部：`accent-line` 装饰线

### CSS 布局
结构 CSS 可重写 `.stage-panel` 为全宽全高无圆角，或保持默认面板但内部用 grid/flexbox 分 2×2 网格。
图标用 `var(--accent)` 填充/描边，一律 inline SVG（不外链图标库）。

- anim-role：`title` / `icon`（每个图标）/ `stat`（每个数字）/ `caption`（每个说明）/ `accent-line`
- slot_schema：`headline`≤18、`icon1_value`≤8、`icon1_label`≤12、`icon2_value`≤8、`icon2_label`≤12、`icon3_value`≤8、`icon3_label`≤12
- default：headline=「太阳系的三个关键数字」、icon1_value=「8」、icon1_label=「大行星 🪐」、icon2_value=「1」、icon2_label=「恒星 ☀️」、icon3_value=「5」、icon3_label=「矮行星 🌑」（图标建议用 emoji 🪐 ⚡ 🧬 等代替图片）
- structure_card：display_name=「图标概要卡」，use_for=[关键数字配图标,概要卡片,emoji图标展示]，spotlight_level=mid，duration_range=[4.5,7.0]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init`：
- `icon`：scale 0.6→1 弹现（back.out(1.6)，stagger 0.1）
- 可选附加装饰动画（callout-circle 描边画出、underline-mark 横向擦出等）

### sfx_cues（写在 structure_card.json）
`[{"at":0.2,"sfx":"pop"},{"at":0.5,"sfx":"sparkle"}]`

## 运动层
**不产出独立运动**。编译时使用通用叙事运动（如 `scale_pop_hold`），负责 title/stat/caption 入场。

## 编译命令
```bash
python assembler/assemble.py --structure icon_overview --motion scale_pop_hold --visual clean_saas \
  --slots generated/templates/19_icon_demo/example_inputs.json --duration 5.5 \
  --out generated/templates/19_icon_demo/index.html
```

## 验收
assemble icon_overview × scale_pop_hold × clean_saas，确认：图标正确弹现、多面板布局美观、render 下确定。
再配一个其他通用运动确认兼容。
