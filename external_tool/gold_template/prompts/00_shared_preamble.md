# 00 · 母模板生成 · 共用前置（每个 prompt 开头都先贴这段）

> 把本段 + 对应的 `NN_*.md` 一起发给 Gemini 3.1 Pro / Claude（IDE 内，确保已装 `hyperframes` skill）。

## 角色与目标
你是 HyperFrames 高级模板工程师。请生成**一个知识科普 spotlight 镜头母模板**（金融/科技/历史/哲学通用），1920×1080、16:9。
最关键：必须**正交三层拆分**，让下游编译器能**自由重组**「任意结构 × 任意运动 × 任意视觉」。
每一层独立使用都有效果：结构层不靠运动也有完整可见画面，运动层搭配非原配结构不报错，视觉层搭配任意结构都生效。

## 舞台面板与布局多样性
背景分两层：
- **外层 `.bg-stage`**：全屏氛围底（视觉层 `tokens.css` 写治理：渐变/网格/暗角）。
- **内层 `.stage-panel`**：默认居中圆角内容面板。但 **不是所有模板都必须用单个居中面板**。

### 布局多样性指引（结构层可通过 CSS 重写 `.stage-panel` 布局）
- **单面板居中**（默认）：适合标题、金句、数据高亮等焦点型内容
- **多面板/分栏**：结构 CSS 可把 `.stage-panel` 改为 `display:grid` 分多个区域，或用 `position:absolute` 放置多个小面板
- **全屏铺满**：结构 CSS 可把 `.stage-panel` 设为 `width:100%;height:100%;border-radius:0` 实现全屏
- **侧边栏 + 主区**：左/右 30% 面板 + 主内容区
- **卡片矩阵**：多个小卡片在画面不同位置展示概要信息
- **浮动元素**：图标/徽标等小元素用 `position:absolute` 散布在画面各处

面板的默认几何由 `assemble.py` 提供，视觉层给三个令牌：
- `--stage-bg`：面板填充色（alpha 0~0.6）
- `--radius-stage`：面板圆角
- `--stage-blur`：玻璃模糊

## Emoji / Lottie 图标规范
- **emoji 作为图标**：直接用文字渲染（如 🧬 🪐 ⚡），`font-size: 80-200px`，不用 `<img>`。
- **Lottie 动画**：用 `<lottie-player>` 或 `dotlottie-player`，`src` 指向 JSON/dotlottie 文件，`loop=false`，用 `hf-seek` 事件控制播放进度。
- 二者均带 `data-anim-role="decoration"` 或 `data-anim-role="icon"`。
- **禁止**：不能有 `<img>` / `<video>` / `<canvas>`（除了 inline SVG）。

## 夸张渲染模式（全屏无蒙版）
- 部分模板（如 `fun_fact`、`emoji_concept`、`myth_busting`、`question_pause`、`open_question`）可不用 `.stage-panel`。
- 结构层直接在 `.bg-stage` 上放置内容，全屏铺满。
- 字号可大到 120-200px，emoji 可达画面 30% 尺寸。
- 适合单个数字/概念/问题的冲击式呈现。
- 在 `structure_card.json` 中标注 `"exaggerated_render": true`。

## 必须先读
1. 本仓 `Gold_Template/CONTRACT.md` —— 三层解耦铁律，**逐条遵守**。
2. `hyperframes` skill 主文件 + 这些 reference（按需）：
   - `references/video-composition.md`、`references/motion-principles.md`、`references/typography.md`（**必读**）
   - `gsap` skill（写 motion.js 时）
   - 数字类读 `data-in-motion.md`；字幕/打字机类读 `references/captions.md` + `references/dynamic-techniques.md`；
     遮罩/擦出类读 `references/css-patterns.md`；视差/图层读 `references/techniques.md`。
3. 本仓 `registry/visuals.json` 里对应 visual 的 palette（**直接用其令牌值，不要另造颜色**）。

## 输出文件（统一放在 `generated/` 目录，严格按 CONTRACT §7；已存在的层就复用、不要重复生成）
- `generated/structures/<structure_id>/structure.html`（CONTRACT §4 片段格式）
- `generated/structures/<structure_id>/slot_schema.json`
- `generated/structures/<structure_id>/structure_card.json`
- `generated/motions/<motion_id>/motion.js`（挂 `window.__hf_motions["<motion_id>"]`）——**仅叙事类通用运动需要**
- `generated/motions/<motion_id>/motion_card.json`——**仅叙事类通用运动需要**
- `generated/visuals/<visual_id>/tokens.css`（`:root` 令牌全集 + `.stage`/`.bg-stage`/`.caption-bar`）
- `generated/visuals/<visual_id>/visual_card.json`
- `generated/templates/<mother_id>/example_inputs.json`（一组真实感中文示例 slots）
- `generated/templates/<mother_id>/index.html`（**用 `assembler/assemble.py` 生成**，作为成品验收）

## 幂等生成 · 跳过已存在（省 token，重跑必读）
> 适用于全部 prompt（含 00~20 旧母模板与 T/V 系列新母模板）。统一重跑时，**先检测后生成**：

1. **逐层检测**：开始写任何一层前，先看 `generated/` 是否已有同名产物：
   - 结构层：存在 `generated/structures/<structure_id>/structure.html` → **跳过该结构**（连带其 slot_schema/structure_card 一并跳过）。
   - 运动层：存在 `generated/motions/<motion_id>/motion.js` → **跳过该运动**。
   - 视觉层：存在 `generated/visuals/<visual_id>/tokens.css` → **跳过该视觉**。
2. **只补缺口**：该 prompt 涉及的三层里，只生成尚不存在的那几层；已存在的直接复用其文件名做编译。
3. **变体同理**：本 prompt 若产出多个变体结构（见下「多样性变体矩阵」），逐个变体 id 各自做上面的存在性检测，已存在的变体跳过。
4. **编译产物**：`generated/templates/<mother_id>/index.html` 若已存在且三层未变，可跳过重编译；如需强制刷新再重跑 `assemble.py`。
5. 在回复开头先列「检测结果：跳过 X / 新建 Y」，再开始动手，避免重复生成浪费 token。

## 多样性变体矩阵（一个 prompt 产出多个模版）
> 同一个内容信息，**空间位置与风格要有多样性**（知识科普调性，趣味但不杂乱）。每个结构 prompt 默认产出 **2~3 个布局变体**，各自是独立 `structure_id` + 独立 `mother_id` 成品。

- **命名约定**：规范结构用基名（如 `problem_hook`）作为变体 A（默认/聚焦型）；其余变体加后缀，如 `problem_hook_scatter`（散布型）、`problem_hook_split`（分栏型）。
- **复用上面「布局多样性指引」的 6 种骨架**，每个变体选不同骨架，确保空间位置不同：
  - 变体 A｜聚焦型：单面板居中（焦点强）
  - 变体 B｜散布型：多个小蒙版/卡片 `position:absolute` 散布画面四角/错落（复杂空间、趣味性）
  - 变体 C｜分栏/侧边型：左右分栏或侧边栏+主区（信息层次）
- **三层零耦合不变**：每个变体仍是合法结构层（零硬值、可独立预览）；bound 变体各自内嵌 `structure-script` + `removeAttribute`。
- **registry 登记**：每个变体在 `registry/structures.json` 追加一条，带 `category`、`variant_of`(=基名/T-code 结构 id)、`variant_label`(中文，如「散布蒙版」)、`family`；变体不单独占 `template_code`（继承父 T-code）。viewer 会把变体缩进归到父结构下展示。
- **示例输入**：每个变体配一组 `example_inputs.json`（可同内容不同排布），命名 `generated/templates/<mother_id>/`。

## 三层零耦合红线（违反即返工）
- 结构层：**零十六进制颜色 / 零字体名 / 零阴影硬值**，全部 `var(--token)`；不写 `data-start/duration`。有专属动画的结构可包含 `<script data-hf="structure-script">`（见下方「结构绑定专属动画」）。
- 运动层 = **纯入场方式**：只定义"元素怎么出现在屏幕上"（方向、缓动、时序），**不操作 DOM、不读数据、不做内容变换**。只用 `[data-anim-role=...]` 选择器；对空集合安全跳过；只入场+保持、不退场；确定性。每个通用运动须覆盖全部 12 叙事角色。
- 视觉层：只定义 CONTRACT §1 的令牌全集 + 3 个皮肤类；**不写内容排版**。`--stage-bg` 的 alpha 在 0~0.6 范围。

## 结构绑定专属动画（`<script data-hf="structure-script">`）

当某个结构有**与 DOM 紧耦合的专属动画**（如数字滚动、打字机、快闪切换、图表生长等），这些动画**不作为独立运动注册**，而是**内嵌到结构层**中。

**适用场景**（以下结构已内嵌专属动画）：
- `stat_highlight`：stat 数字 0→N countup 滚动
- `bar_chart` / `line_chart` / `donut_pie` / `flow_steps` / `cycle_loop`：dataviz/diagram 动画

**做法**：
1. 在 `structure.html` 末尾加 `<script data-hf="structure-script">` 块
2. 定义 `window.__hf_structure_init = function(tl, ctx) { ... }`（签名与 motion 相同）
3. 完成专属动画后，对已处理的元素 **`removeAttribute("data-anim-role")`**，避免通用运动重复动画
4. **不处理其他叙事角色**（那些由通用运动层处理）
5. assembler 自动提取并注入，在通用运动之前执行

编译时，这些结构搭配任意通用入场运动（如 `scale_pop_hold`、`slide_in_hold`），运动负责剩余叙事元素的入场。

## 音效绑定 SFX（每个 prompt 都要在合适拍点配，详见 CONTRACT §10）
> 目标：音效**跟当前画面语义匹配、且有趣**，绑在动画发生的同一时刻。生成本模板时**必须**在关键拍点声明 `sfx_cues`（bound 结构写在 structure_card.json，generic 结构搭配的运动写在 motion_card.json，或直接写在 structure_card.json）。

1. **先查库**：从 `registry/sfx.json` 选 id。常用语义映射（见 sfx.json 的 `scene_map`）：
   - 数字蹦出/计数 → `pop` / `rise`；上升·发射·增长 → `rise`（趣味首选）；逐项逐阶段 → `tick`
   - 图表画出/柱升/折线生长 → `whoosh` + `rise`；高亮·结论 → `sparkle` / `chime`
   - 收尾·确认 → `chime`；聚焦·放大·切换 → `click` / `whoosh`；转场·擦出 → `whoosh` / `sparkle`
2. **缺就生成**：若某语义库里没有贴切音效，**运行 prompt 时用音频技能生成一个短音效**（落到 `sfx/<id>.mp3`）并在 `registry/sfx.json` 追加一条（含 desc/energy/default_dur/default_volume），再在 `sfx_cues` 引用。
3. **拍点对齐**：`cue.at` = 该动画发生秒（如数字 0.8s 蹦出 → `{"at":0.8,"sfx":"pop"}`）。
4. **预算**：单镜头 sfx ≤ 2 个，避免吵；`track_index` 从 102 起。无 `sfx_cues` 则不注入（向后兼容）。

## 趣味化语义动效 motif（科普调性，建议主动用）
> 让科普教学画面更有趣：用**与内容语义绑定的小动效隐喻**承载数据/概念，而不是干巴巴的入场。motif 属于结构层自有动画——**需要 bespoke 动画的 motif 一律写进 `<script data-hf="structure-script">`（该结构即为 bound，自动跳过通用运动）**，且 inline SVG、确定性、克制不喧宾夺主。

示例（可按内容套用）：
- **折线上升趋势**：一个小火箭/小箭头**顺着折线 path 向上飞**（沿 path 运动 + 轻微抖动尾焰），配 `rise`。
- **折线下降**：一片落叶/向下的小指标**随折线下落**，配 `whoosh`。
- **柱状增长**：柱子像幼苗一样**从底拔高**，末根最高处冒个小高亮点，配 `rise`。
- **计数/指标**：数字滚动如里程表，到位时打个 ✓ 或冒一下高亮，配 `pop`。
- **流程推进**：一个发光小信号点**沿连线 edge 跑过**每个节点点亮它，每过一节点 `tick`。
- **环形占比**：扇区像扇子**依次扫开**，主扇区到位 `sparkle`。
- **聚焦放大**：镜头/取景框**咔哒落位**框住目标，配 `click`。

要求：motif 的“可见静态兜底”仍要满足——不放 motif 时结构本身就是完整可读画面（motif 只是锦上添花的运动）。

## 防过度装饰（CONTRACT §11）
"不装饰"是一等默认；生成母模板时只产组件，何时选用由下游 Planner 治理。
**例外**：emoji 文字图标与 Lottie 动画图标属于允许的装饰元素（带 `data-anim-role="decoration"` 或 `"icon"`），不受此条限制。

## 生成后自检（在回复里附 checklist）
1. `python assembler/assemble.py --structure <s> --motion <m> --visual <v> --slots generated/templates/<mother_id>/example_inputs.json --duration <d> --out generated/templates/<mother_id>/index.html` 能跑通；
2. `npx hyperframes lint && npx hyperframes validate` 通过；
3. 三层文件互相不引用对方私有名；
4. 至少 3 组不同 slots 都不溢出；
5. **独立性验证**：结构搭配非原配 motion 仍有完整可见画面，motion 搭配非原配结构不报错。

## 通用约束
- 中文正文用 `Noto Sans SC`，衬线场景用 `Noto Serif SC`。
- 标题 ≥60px，正文 ≥20px，数据标签 ≥16px；数字列加 `font-variant-numeric: tabular-nums`。
- 首个入场 0.1~0.3s 起，单镜头内 ≥3 种不同 ease。
