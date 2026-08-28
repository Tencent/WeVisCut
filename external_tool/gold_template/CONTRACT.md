# Gold Template 三层解耦铁律（CONTRACT）

> 这是整套系统的**唯一权威契约**。所有母模板由 Gemini/Claude 生成时，必须严格遵守本文件，
> 否则 `assembler/assemble.py` 无法把「结构 × 运动 × 视觉」稳定重组。
> 一句话目标：**任意 `(structure_id, motion_id, visual_id)` 三元组都能拼出一个能 `hyperframes render` 的 1920×1080 镜头。**
> 三层彼此独立，每一层单独使用或与任意其它两层搭配都必须有效、不报错。

---

## 0. 三层职责（正交，零耦合）

| 层 | 产物 | 只负责 | 绝对禁止 |
|---|---|---|---|
| **结构层 Structure** | `generated/structures/<id>/structure.html` + `slot_schema.json` | 画面 DOM 排版、语义槽位、动画角色锚点 | 写死颜色/字体/阴影；写任何 GSAP；写 `data-start/duration` |
| **运动层 Motion** | `generated/motions/<id>/motion.js` | 一段 GSAP 入场+保持动画，只按 `data-anim-role` 选择元素 | 引用具体结构 class；引用具体颜色；写死时长 |
| **视觉层 Visual** | `generated/visuals/<id>/tokens.css` | `:root` 设计令牌 + 背景/全局字体皮肤 | 写内容排版；写动画；引用具体结构 class |

三层任何一层都**不准知道另外两层的存在**。结构靠 `var(--token)` 拿颜色、靠 `data-anim-role` 暴露动画锚点；运动只认 `data-anim-role`；视觉只改令牌值和背景皮肤。

### 独立有效性（核心原则）

- **结构层必须自带可见静态内容**：即使没有任何运动层加持，渲染出来也是一个完整可阅读的画面（有标题、有数据图形、有文字）。dataviz/diagram 的结构层**必须内嵌默认数据的静态占位图形**（如 5 根默认高度的柱、折线路径、环形切片），不得留空容器。
- **运动层必须对缺失角色安全跳过**：搭配非匹配结构时，选不到的 `data-anim-role` 不动即可，不报错。能选到的通用角色（title/accent-line 等）仍正常入场。
- **视觉层纯令牌**：搭配任意结构都生效，因为所有结构都只用 `var(--token)`。

---

## 1. 设计令牌词表（视觉层唯一可定义的东西）

每个 `generated/visuals/<id>/tokens.css` **必须且只能**定义下面这套令牌（值可不同，键必须齐全）：

```css
:root {
  /* 背景 / 表面 */
  --bg: #060d1a;          --bg-2: #0a1525;
  --surface: rgba(20,38,60,.55);   --surface-2: rgba(20,38,60,.8);
  /* 文字三级 */
  --ink-1: #eef6fb;       --ink-2: #b9d3e6;       --ink-3: #7d96ad;
  /* 强调色 */
  --accent: #7fc7d9;      --accent-2: #f5b14a;
  --danger: #ff7a7a;      --success: #6ee7b7;
  /* 线条 */
  --line: rgba(127,199,217,.18);   --line-strong: rgba(127,199,217,.42);
  /* 字体 */
  --font-display: "Noto Sans SC", system-ui, sans-serif;
  --font-body:    "Noto Sans SC", system-ui, sans-serif;
  --font-caption: "Noto Sans SC", system-ui, sans-serif;
  --weight-display: 800;  --weight-body: 500;
  /* 圆角 / 阴影 / 栅格 */
  --radius-card: 22px;    --radius-pill: 999px;
  --shadow-soft: 0 12px 36px rgba(0,0,0,.45);
  --shadow-glow: 0 0 36px rgba(127,199,217,.35);
  --grid-color: rgba(127,199,217,.04);
  /* 居中舞台面板（半透明 + 圆边）—— alpha 0~0.6，0 = 无蒙版 */
  --stage-bg: rgba(10,21,37,.5);   /* 面板填充，alpha 在 0~0.6 范围，按需调节 */
  --radius-stage: 48px;
  --stage-blur: 8px;               /* 0px = 无模糊 */
}
```

`tokens.css` 还**只允许**额外定义这 2 个全局皮肤类：

```css
.stage { font-family: var(--font-body); color: var(--ink-1); background: var(--bg); }
.bg-stage { /* 外层全屏氛围底：渐变 / 网格 / 噪点 / 暗角，只用上面的令牌 */ }
.caption-bar .caption-text { /* 字幕条皮肤，只用令牌 */ }
```

> 视觉层禁止出现任何内容布局。

### 背景与蒙版

- **外层 `.bg-stage`**：全屏 1920×1080 氛围底。
- **内层 `.stage-panel`**：居中圆角内容面板，几何由 `assemble.py` 统一提供。
  **`--stage-bg` 的 alpha 值在 0~0.6 范围自由设定**：
  - alpha=0：完全透明（无蒙版效果，内容直接显示在底层上）
  - alpha=0.5~0.6：半透明蒙版（适合叠在视频底片上增加文字可读性）
  - 不同视觉主题可自由选择是否加蒙版以及透明度程度。

---

## 2. 动画角色词表（运动层的唯一选择器）

结构层给需要动的元素打 `data-anim-role`，运动层**只能**用这套角色做选择器：

```
# —— 通用角色（叙事类结构）——
bg            背景层（视差/缓动用）
eyebrow       主标题上方的小标签
title         主标题
subtitle      副标题 / 一句话
stat          大数字（number_countup 的目标）
unit          数字单位 / 后缀
card          可重复的卡片块（每张卡都打这个）
list-item     需要 stagger 的列表项（每项都打）
decoration    装饰性 emoji / Lottie 图标（不承载关键信息，纯视觉趣味）
accent-line   分隔线 / 下划线 / 进度条
caption       下三分之一字幕
badge         角标 / 来源标签

# —— DataViz 角色（图表/数字/曲线，family=dataviz）——
chart-axis    坐标轴 / 网格线
chart-bar     柱状条（每根都打；bar_grow 逐根升起）
chart-line    折线路径 <path>（line_draw 用 stroke-dashoffset 画出）
chart-area    面积填充
chart-slice   饼/环图扇区（每片都打；pie_sweep 扫过）
chart-point   折线/散点上的数据点
chart-value   数值标签（可叠 number_countup 滚动）
chart-label   分类/坐标标签
chart-legend  图例项

# —— Diagram 角色（关系/流程/树，family=diagram）——
node          流程/网络/树的节点（每个都打）
edge          连接线 / 箭头（每条都打；可用 stroke-dashoffset 生长）
node-label    节点文字
cluster       分组/泳道容器
marker        箭头头 / 端点装饰

# —— Iconography 角色（图标/强调装饰，family=icon）——
icon          语义图标（svg/字体图标）
callout-circle 手绘风圈出
callout-arrow  指向箭头
underline-mark 荧光笔下划线
highlight-block 文字背景高亮块
check-mark     ✓/✗ 判定标记
stamp          角标 / 印章 / "重点"戳
```

规则：
- 运动层用 `root.querySelectorAll('[data-anim-role="chart-bar"]')` 取元素，**绝不**用结构 class。
- **对空集合安全跳过**（`.length===0` 时不执行）。这保证任意 structure×motion 组合不报错。
- 一个元素可叠多个语义，但 `data-anim-role` 只给一个主角色。

---

## 3. 槽位词表（结构层暴露给模型/编译器的可填内容）

结构层用 `data-slot="<id>"` 标出所有可变文本，并在 `slot_schema.json` 声明：

```json
{
  "structure_id": "hook_title_intro",
  "duration_default": 6.0,
  "slots": {
    "eyebrow":  {"type":"string","label":"小标签","default":"权威综述 · 2025","max_chars":14},
    "headline": {"type":"string","label":"主标题","default":"维生素B6 的双面真相","max_chars":18}
  }
}
```

规则：
- 每个 `data-slot` 必在 `slot_schema.slots` 里有一条，含 `default` + `max_chars`。
- 数据槽（dataviz 的 `series`/`nodes`/`edges`）用 `"type":"json"`。

---

## 4. 结构文件格式（`structure.html`）

`structure.html` 是一个**片段**，不是完整 HTML。固定两块：`<style data-hf="structure-css">` + `<div data-hf="structure-dom">`。

硬规则：
1. 不含 `<html>/<head>/<body>`、不含 gsap、不含 `<script>`、不含 `data-start/duration`。
2. 不出现任何十六进制颜色——颜色/字体/阴影/圆角只用 `var(--token)`。
3. **dataviz/diagram 结构必须内嵌默认数据的静态占位图形**（基于 slot_schema 的 default series/nodes 数据），让结构层独立即可见。运动层的数据驱动代码（如 bar_grow）会先清空再重注入，两种场景都正常。
4. 文字靠 `max-width` 自然换行，**禁止 `<br>`**。

---

## 5. 运动文件格式（`motion.js`）

```js
window.__hf_motions = window.__hf_motions || {};
window.__hf_motions["scale_pop_hold"] = function (tl, ctx) {
  var gsap = ctx.gsap, root = ctx.root;
  var $ = function(role) { return root.querySelectorAll('[data-anim-role="' + role + '"]'); };
  if ($("title").length) tl.from($("title"), { scale: .8, opacity: 0, duration: .7, ease: "back.out(1.4)" }, 0.3);
  // ... 对空集合安全跳过
};
```

硬规则：
1. 挂到 `window.__hf_motions["<motion_id>"]`，签名 `(tl, ctx)`。
2. **只用 `[data-anim-role]` 选择器**；**对空集合（`.length===0`）必须安全跳过**。
3. 只做入场+保持；确定性（无 `Math.random()/Date.now()/repeat:-1`）。
4. dataviz/diagram motion 读数据驱动图形时，若容器元素（如 `chart-axis`）不存在则直接跳过整段逻辑。

---

## 6. 编译器 = `assembler/assemble.py`

输入 `(structure_id, motion_id, visual_id, slots, duration)`，输出合法 HyperFrames `index.html`。
生成产物在 `generated/` 目录下。

**解耦判定**：换 `visual_id` 只换 tokens；换 `motion_id` 只换 `<script>`；换 `structure_id` 只换 DOM + structure-css。三者互不牵连即合规。

---

## 7. 每个母模板生成时要落的文件

所有生成产物统一放在 `generated/` 目录：

```
generated/structures/<structure_id>/structure.html
generated/structures/<structure_id>/slot_schema.json
generated/structures/<structure_id>/structure_card.json
generated/motions/<motion_id>/motion.js
generated/motions/<motion_id>/motion_card.json
generated/visuals/<visual_id>/tokens.css
generated/visuals/<visual_id>/visual_card.json
generated/templates/<mother_id>/index.html       # assemble.py 产的成品
generated/templates/<mother_id>/example_inputs.json
```

> **要重新生成所有模板，只需删除 `generated/` 目录。**

若某层在更早的 prompt 已生成，后续 prompt **复用，不重复生成**。

---

## 8. 验收（每个新母模板都要过）

```
1. assemble.py 能拼出 index.html
2. npx hyperframes lint && npx hyperframes validate 通过
3. npx hyperframes inspect 无非预期溢出
4. npx hyperframes render 出 mp4
5. 三层零耦合自查：结构无硬编码颜色/字体、运动只用 data-anim-role、视觉只动 tokens
6. 至少 3 组不同 slots 输入都能正常渲染
7. 独立性验证：结构搭配非原配 motion 仍有可见画面、motion 搭配非原配结构不报错
```

---

## 9. 装饰家族（DataViz / Diagram / Iconography）—— 不新增第 4 层

三个语义家族在 `registry/decorations.json` 登记，仅作目录 + 打标用，不改变编译路径。

**关键原则：所有 dataviz/diagram 结构层必须自带静态占位图形**，保证在搭配任意 motion 时画面仍然可见（非空白）。

数据驱动动画有两种合规落地，二选一：
- **(旧) 运动层驱动**：data 驱动代码写在 motion.js（如 `bar_grow` 读 series 先清空再重注入）。
- **(推荐·bound) 结构内嵌**：与 DOM/数据紧耦合的专属动画（countup、图表生长、关系图、多米诺因果链、趣味 motif 如小火箭顺折线上升）写在结构层的 `<script data-hf="structure-script">` 里，定义 `window.__hf_structure_init(tl, ctx)`，处理完对相关元素 `removeAttribute("data-anim-role")`；assembler 自动在通用运动前执行。此类结构 `structure_card.json` 标 `"bound": true`，编译/预览时**自动跳过通用运动**，只保留其内嵌动画 + 通用运动负责剩余叙事角色（title 等）。

---

## 10. 音效绑定（SFX）—— 可选，additive

- 在 motion_card.json **或** structure_card.json 声明 `"sfx_cues":[{"at":0.30, "sfx":"pop"}]`，`assemble.py` 自动注入 `<audio>`。
- sfx id 在 `registry/sfx.json` 登记（含 `scene_map` 语义映射）。**音效要与画面语义匹配且有趣**，绑在动画发生的同一拍点；单镜头 ≤2 个。
- 库里缺贴切音效时，可用音频技能生成一个短音效落到 `sfx/` 并在 `registry/sfx.json` 追加一条再引用。
- 无 `sfx_cues` 即不注入，向后兼容。

---

## 11. 防过度装饰治理

组件库与选用解耦。生成母模板时只产组件，何时选用由下游 Planner 治理。
Emoji / Lottie 图标是允许的装饰元素（带 `data-anim-role="decoration"`），但需克制——每个结构最多 1~2 个装饰图标。

---

## 12. Emoji / Lottie 图标约定

- **Emoji** 作为图标位直接用文字渲染（如 🧬 🪐 ⚡），font-size 80~200px，不用 `<img>`。
- **Lottie** 动画用 `<lottie-player>` 或 `<dotlottie-player>`，src 指向 JSON/dotlottie 文件，`loop=false`，用 `hf-seek` 控制播放进度。
- 二者均带 `data-anim-role="decoration"`，可带 `data-slot` 使 emoji 内容可替换。
- **禁止** `<img>` / `<video>` / `<canvas>`（除 inline SVG 和 Lottie player）。

---

## 13. 夸张渲染模式（全屏无蒙版）

部分结构（`structure_card.json` 标 `"exaggerated_render": true`）不使用 `.stage-panel`，内容直接铺在 `.bg-stage` 上全屏呈现：
- 字号可大到 120~200px，emoji 可达画面 30% 尺寸。
- 适合单个数字/概念/问题的冲击式呈现（如 fun_fact、emoji_concept、myth_busting、question_pause、open_question）。
- 结构层 CSS 把 `.stage-panel` 设为 `width:100%;height:100%;border-radius:0;background:transparent` 或直接不依赖 `.stage-panel`。
