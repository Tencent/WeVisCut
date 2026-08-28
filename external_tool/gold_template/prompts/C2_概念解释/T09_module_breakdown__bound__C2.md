# T09 · module_breakdown（类别2 概念与信息解释）　【diagram · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 module_breakdown（family=diagram，含内嵌动画脚本）**。
> **不产出独立运动**，放射布局 + 连线生长动画内嵌在结构的 `<script data-hf="structure-script">`。
> 编译时搭配一个通用叙事运动（推荐 `scale_pop_hold`，仅负责 title 等叙事角色）。
> 视觉层复用现有（推荐 `dark_tech` 或 `clean_saas`）。
> mother_id = `T09_module_breakdown`，duration 默认 7.0s。先读 `references/data-in-motion.md` + `references/css-patterns.md`。

## 数据契约
- slot `modules` 是 **JSON 数据槽**：`slot_schema` 里 `"modules": {"type":"json", ...}`，default 给 4~6 个子模块：
  `[{"label":"T细胞"},{"label":"B细胞"},{"label":"巨噬细胞"},{"label":"树突细胞"},{"label":"自然杀伤细胞"}]`
- 结构层渲染**静态占位骨架**（中心节点 + 固定 N 个放射占位节点 + 占位连线），不搭配运动时画面仍完整。
- 结构内嵌脚本读 `modules`（`window.__hyperframes.getVariables().modules`）生成/填充周围节点并径向布局，再做连线生长 + 节点弹入。

## 结构层 generated/structures/module_breakdown/
> 结构层必须内嵌默认数据的静态占位图形，确保不搭配运动时画面仍完整可见。

中心模块 + 周围子模块放射布局。上 `title` → 中部放射图（中心 `node` + 周围若干 `node` + 连线 `edge` + 每个 `node-label`）。
- anim-role：`title` / `node`（中心+子）/ `edge`（每条连线）/ `node-label`（每节点文字）/ `marker`（端点/箭头，可选）
- 连线用内联 `<svg>` 的 `<line>`/`<path>`（`stroke=var(--line-strong)`），节点用绝对定位圆/圆角块（`var(--accent)` / `var(--accent-2)` 交替）
- slot_schema：`headline`≤16、`center_label`≤8、`modules`(json)；`data_slots:["modules"]`
- default：headline=「人体免疫系统」、center_label=「免疫响应」、modules=上方示例
- structure_card：use_for=[系统分解,模块拆解,组成部分]，spotlight_level=mid，duration_range=[6.0,9.0]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- 读 `modules` → 清空占位 → 按数量等角度径向布点（围绕中心），创建子 `node` + `node-label` + 中心到子节点的 `edge`（SVG line/path）
- 中心 `node`：scale .6→1 + opacity，back.out(1.5)，起手 0.1s
- `edge`：用 `strokeDashoffset`（path draw）或 scaleX 沿径向生长，逐条 stagger 0.08
- 子 `node`：跟随对应 edge 末端弹入 scale .4→1 back.out(1.4)
- `node-label`：opacity+y 渐显
- 完成后对所有上述元素 `removeAttribute("data-anim-role")`，**不处理 title**（由通用运动负责）

### sfx_cues（可选，写在 structure_card.json）
`[{"at":0.4,"sfx":"whoosh"},{"at":1.2,"sfx":"pop","comment":"末节点弹入"}]`

## 运动层
**不产出独立运动**。编译时用通用叙事运动负责 title 入场。

## 视觉层
复用 `generated/visuals/dark_tech/`。

## 编译命令
```bash
python assembler/assemble.py --structure module_breakdown --motion scale_pop_hold --visual dark_tech \
  --slots generated/templates/T09_module_breakdown/example_inputs.json --duration 7.0 \
  --out generated/templates/T09_module_breakdown/index.html
```

## 多样性变体矩阵（一个 prompt 产出 2 个模版，均 bound 各自内嵌脚本）
按 00「多样性变体矩阵」，产出 2 个空间布局变体（各独立 id，追加 `variant_of:"module_breakdown"` + `variant_label`；变体各自内嵌 `structure-script` + `removeAttribute`）：
- A｜`module_breakdown` · 放射型：中心 + 周围子模块径向布点（「中心放射」）→ `T09_module_breakdown`
- B｜`module_breakdown_grid` · 网格型：中心模块 + 子模块网格环绕、连线为正交折线（「网格环绕」）→ `T09_module_breakdown_grid`

## 验收
按 00 自检；额外确认：4/5/6 个子模块都不重叠、连线指向准确、render 下确定性生长；含内嵌动画结构在 viewer 自动跳过通用运动。
