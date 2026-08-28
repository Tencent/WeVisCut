# T13 · branching_flow（类别3 流程与结构关系）　【diagram · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 branching_flow（family=diagram，含内嵌动画脚本）**。
> **不产出独立运动**，主干 + 多分支线条生长内嵌在 `<script data-hf="structure-script">`。
> 编译时搭配通用叙事运动（推荐 `scale_pop_hold`，仅负责 title）。
> 视觉层复用现有（推荐 `dark_tech` 或 `clean_saas`）。
> mother_id = `T13_branching_flow`，duration 默认 7.0s。先读 `references/data-in-motion.md` + `references/css-patterns.md`（SVG path draw）。

## 数据契约
- slot `branches` 是 **JSON 数据槽**：`"branches": {"type":"json", ...}`，default 给 2~4 条分支：
  `[{"label":"反射"},{"label":"折射"},{"label":"吸收"}]`
- 结构层渲染**静态占位骨架**（左侧主干节点 + 右侧 N 个分支节点 + 主干到各分支的占位曲线），不搭配运动时画面仍完整。
- 内嵌脚本读 `branches` 生成/填充分支节点，再做主干→分支的 SVG path draw 生长 + 分支节点弹入。

## 结构层 generated/structures/branching_flow/
> 结构层必须内嵌默认数据的静态占位图形，确保不搭配运动时画面仍完整可见。

一个主干分出多条分支（左→右）。上 `title` → 左侧主干 `node`（slot=`root_label`）→ 右侧若干分支 `node` + 弯曲 `edge` 连接 + 每分支 `node-label`。
- anim-role：`title` / `node`（主干+分支）/ `edge`（每条分支曲线）/ `node-label` / `marker`（分支端点，可选）
- 主干节点纵向居中靠左，分支节点纵向均布靠右；连线用 SVG `<path>` 贝塞尔曲线（`stroke=var(--line-strong)`，主干用 `var(--accent)`）
- slot_schema：`headline`≤16、`root_label`≤10、`branches`(json)；`data_slots:["branches"]`
- default：headline=「光遇到物体后的三种命运」、root_label=「入射光」、branches=上方示例
- structure_card：use_for=[分支决策,分类,多结果展示]，spotlight_level=mid，duration_range=[6.0,9.0]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- 主干 `node`：scale .8→1 + opacity，back.out(1.4)，0.1s
- 读 `branches` → 清空占位 → 创建每条分支 `node` + `node-label` + 主干到该分支的 `edge`（path）
- 每条 `edge`：`strokeDasharray/strokeDashoffset` path draw 从主干向分支生长，stagger 0.12
- 分支 `node`：跟随 path 末端弹入；`node-label` 渐显
- 完成后对 node/edge/node-label/marker `removeAttribute("data-anim-role")`，**不处理 title**

### sfx_cues（可选）
`[{"at":0.6,"sfx":"whoosh","comment":"分支生长"}]`

## 运动层
**不产出独立运动**。通用运动负责 title。

## 视觉层
复用 `generated/visuals/dark_tech/`。

## 编译命令
```bash
python assembler/assemble.py --structure branching_flow --motion scale_pop_hold --visual dark_tech \
  --slots generated/templates/T13_branching_flow/example_inputs.json --duration 7.0 \
  --out generated/templates/T13_branching_flow/index.html
```

## 多样性变体矩阵（一个 prompt 产出 2 个模版，均 bound 各自内嵌脚本）
按 00「多样性变体矩阵」，产出 2 个空间布局变体（各独立 id，追加 `variant_of:"branching_flow"` + `variant_label`）：
- A｜`branching_flow` · 横展型：左主干 → 右多分支（「左干右枝」）→ `T13_branching_flow`
- B｜`branching_flow_radial` · 放射型：中心主干 → 四周放射分支（「中心放射」）→ `T13_branching_flow_radial`

## 验收
按 00 自检；确认 2/3/4 条分支不重叠、曲线从主干平滑连到分支、render 下 path draw 确定性；viewer 自动跳过通用运动。
