# T12 · vertical_flow（类别3 流程与结构关系）　【diagram · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 vertical_flow（family=diagram，含内嵌动画脚本）**。
> **不产出独立运动**，纵向阶段链 + 连线生长内嵌在 `<script data-hf="structure-script">`。
> 编译时搭配通用叙事运动（推荐 `slide_in_hold`，仅负责 title）。
> 视觉层复用现有（推荐 `clean_saas` 或 `dark_tech`）。
> mother_id = `T12_vertical_flow`，duration 默认 7.0s。先读 `references/data-in-motion.md` + `references/css-patterns.md`。

## 数据契约
- slot `nodes` 是 **JSON 数据槽**：`"nodes": {"type":"json", ...}`，default 给 3~5 个阶段：
  `[{"label":"口腔"},{"label":"食道"},{"label":"胃"},{"label":"小肠"},{"label":"大肠"}]`
- 结构层渲染**静态占位骨架**（固定 N 个纵向节点 + 节点间竖向连线），不搭配运动时画面仍完整。
- 内嵌脚本读 `nodes` 生成/填充每个阶段节点，再做自上而下连线生长 + 节点依次入场。

## 结构层 generated/structures/vertical_flow/
> 结构层必须内嵌默认数据的静态占位图形，确保不搭配运动时画面仍完整可见。

上到下纵向阶段链。上 `title` → 中部纵向链（若干 `node` 横条 + 节点间竖向 `edge` + 每节点 `node-label`，可带序号 `marker`）。
- anim-role：`title` / `node`（每阶段）/ `edge`（每段连线）/ `node-label` / `marker`（序号点，可选）
- 节点纵向居中排列（flex column 或绝对定位），连线用 SVG 竖向 `<line>`/`<path>`（`stroke=var(--line-strong)`）
- slot_schema：`headline`≤16、`nodes`(json)；`data_slots:["nodes"]`
- default：headline=「食物在消化系统的旅程」、nodes=上方示例
- structure_card：use_for=[纵向流程,阶段推进,自然过程]，spotlight_level=mid，duration_range=[6.0,9.0]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- 读 `nodes` → 清空占位 → 创建每个阶段 `node` + `node-label` + 段间 `edge`
- 第一个 `node`：y-20 + scale .9→1 + opacity，power3.out，0.1s
- 逐段：`edge` 用 `scaleY 0→1`（transform-origin:top）或 strokeDashoffset 向下生长，紧接下一 `node` 弹入，整体 stagger 0.5s/段
- `node-label`：随各自 node 渐显；`marker` 序号点 pop
- 完成后对 node/edge/node-label/marker `removeAttribute("data-anim-role")`，**不处理 title**

### sfx_cues（可选）
`[{"at":0.5,"sfx":"tick","comment":"逐段推进"}]`

## 运动层
**不产出独立运动**。通用运动负责 title。

## 视觉层
复用 `generated/visuals/clean_saas/`。

## 编译命令
```bash
python assembler/assemble.py --structure vertical_flow --motion slide_in_hold --visual clean_saas \
  --slots generated/templates/T12_vertical_flow/example_inputs.json --duration 7.0 \
  --out generated/templates/T12_vertical_flow/index.html
```

## 多样性变体矩阵（一个 prompt 产出 2 个模版，均 bound 各自内嵌脚本）
按 00「多样性变体矩阵」，产出 2 个空间布局变体（各独立 id，追加 `variant_of:"vertical_flow"` + `variant_label`）：
- A｜`vertical_flow` · 直列型：节点居中竖直串联（「居中直列」）→ `T12_vertical_flow`
- B｜`vertical_flow_zigzag` · 错列型：节点左右交错下行、连线折线（「左右错列」）→ `T12_vertical_flow_zigzag`

## 验收
按 00 自检；确认 3/4/5 个阶段纵向不溢出、连线对齐节点中线、render 下逐段确定性；viewer 自动跳过通用运动。
