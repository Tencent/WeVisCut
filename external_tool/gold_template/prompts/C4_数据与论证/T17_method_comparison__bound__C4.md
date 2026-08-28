# T17 · method_comparison（类别4 对比与证据展示）　【dataviz · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 method_comparison（family=dataviz，含内嵌动画脚本）**。
> **不产出独立运动**，表格逐列/逐行 reveal 内嵌在 `<script data-hf="structure-script">`。
> 编译时搭配通用叙事运动（推荐 `slide_in_hold`，仅负责 title）。
> 视觉层复用现有（推荐 `minimal_academic` 或 `clean_saas`）。
> mother_id = `T17_method_comparison`，duration 默认 7.0s。先读 `references/data-in-motion.md`。

## 数据契约
- slot `columns`（JSON）：方法列名，default `["牛顿力学","相对论"]`（2~3 列）
- slot `rows`（JSON）：每行一个维度 + 各列取值，default
  `[{"dim":"适用范围","values":["低速宏观","高速/强引力"]},{"dim":"时空观","values":["绝对时空","相对时空"]},{"dim":"质能关系","values":["独立","E=mc²"]}]`
- 结构层渲染**静态占位表格**（表头 + N 行 × M 列单元格占位），不搭配运动时画面仍完整。
- 内嵌脚本读 `columns`/`rows` 生成/填充表格，再逐列或逐行 reveal；可高亮「我方」列（`var(--accent)` 底）。

## 结构层 generated/structures/method_comparison/
> 结构层必须内嵌默认数据的静态占位表格，确保不搭配运动时画面仍完整可见。

2~3 列多行对比表。上 `title` → 表格（列头 `col-header` + 行标 `row-label` + 单元格 `cell`）→ `accent-line`。
- anim-role：`title` / `col-header`（每列）/ `row-label`（每行维度）/ `cell`（每单元格）/ `accent-line`
- 表格用 CSS grid（`grid-template-columns`），优胜列整列底色 `var(--accent)` 浅化；数字 `tabular-nums`
- slot_schema：`headline`≤16、`columns`(json)、`rows`(json)；`data_slots:["columns","rows"]`
- default：headline=「牛顿力学 vs 相对论」、columns/rows 用上方示例
- structure_card：display_name=「对比表格」，use_for=[多维度对比,理论方法对比,表格比较]，spotlight_level=high，duration_range=[5.5,8.5]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- 读 `columns`/`rows` → 清空占位 → 生成表格 DOM
- `col-header`：从上滑入 stagger 0.1
- `row-label` + 该行 `cell`：逐行 reveal（y+12 opacity），stagger 0.18/行
- 优胜列单元格末尾轻微 pulse/底色加深强调
- 完成后对 col-header/row-label/cell/accent-line `removeAttribute("data-anim-role")`，**不处理 title**

### sfx_cues（可选）
`[{"at":0.4,"sfx":"tick"},{"at":1.6,"sfx":"pop","comment":"优胜列强调"}]`

## 运动层
**不产出独立运动**。通用运动负责 title。

## 视觉层
复用 `generated/visuals/minimal_academic/`。

## 编译命令
```bash
python assembler/assemble.py --structure method_comparison --motion slide_in_hold --visual minimal_academic \
  --slots generated/templates/T17_method_comparison/example_inputs.json --duration 7.0 \
  --out generated/templates/T17_method_comparison/index.html
```

## 多样性变体矩阵（一个 prompt 产出 2 个模版，均 bound 各自内嵌脚本）
按 00「多样性变体矩阵」，产出 2 个空间布局变体（各独立 id，追加 `variant_of:"method_comparison"` + `variant_label`）：
- A｜`method_comparison` · 表格型：列头 + 多行单元格对比表（「对比表格」）→ `T17_method_comparison`
- B｜`method_comparison_cards` · 卡片型：每个方法一张大卡并列、维度做卡内条目（「并列卡片」）→ `T17_method_comparison_cards`

## 验收
按 00 自检；确认 2 列×3 行 与 3 列×4 行都对齐不溢出、长文本单元格换行不破格、render 下逐行确定性；viewer 自动跳过通用运动。
