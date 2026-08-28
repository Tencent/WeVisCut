# T18 · metric_cards（类别4 对比与证据展示）　【dataviz · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 metric_cards（family=dataviz，含内嵌动画脚本）**。
> **不产出独立运动**，数字 count-up + 卡片 pop 内嵌在 `<script data-hf="structure-script">`（滚动逻辑同 `stat_highlight`）。
> 编译时搭配通用叙事运动（推荐 `scale_pop_hold`，仅负责 title）。
> 视觉层复用现有（推荐 `dark_tech` 或 `clean_saas`）。
> mother_id = `T18_metric_cards`，duration 默认 6.0s。先读 `references/data-in-motion.md`（数字滚动）。

## 数据契约
- slot `metrics`（JSON）：2~4 个指标，default
  `[{"value":71,"unit":"%","label":"地球表面被水覆盖"},{"value":97.5,"unit":"%","label":"是不能直接饮用的海水"},{"value":0.3,"unit":"%","label":"可供人类直接利用"}]`
- 结构层渲染**静态占位卡片**（固定 N 张卡，数字显示为 default 终值），不搭配运动时画面仍完整。
- 内嵌脚本读 `metrics` 生成/填充每张卡，再做卡片 pop + 数字 0→N count-up。

## 结构层 generated/structures/metric_cards/
> 结构层必须内嵌默认数据，确保不搭配运动时画面仍完整（数字直接显示终值）。

多指标卡并列。上 `title` → 一行 2~4 张指标 `card`（每张：大数字 `stat` + 单位 `unit` + 标签 `card-label`）→ `accent-line`。
- anim-role：`title` / `card`（每张）/ `stat`（每数字）/ `unit`（每单位）/ `card-label`（每标签）/ `accent-line`
- 卡片用 grid 等宽；数字 `font-variant-numeric: tabular-nums`，font-size 64~88px，色 `var(--accent)`
- slot_schema：`headline`≤16、`metrics`(json)；`data_slots:["metrics"]`
- default：headline=「关于水的三个数字」、metrics 用上方示例
- structure_card：display_name=「关键数字」，use_for=[关键数字,惊人比例,核心指标]，spotlight_level=high，duration_range=[5.0,8.0]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- 读 `metrics` → 清空占位 → 生成每张 `card`
- `card`：scale .85→1 + opacity，back.out(1.4)，stagger 0.15
- `stat`：每张卡数字 0→value count-up（保留小数位与原值一致，用 gsap 对象 + onUpdate 写 textContent）
- `unit`/`card-label`：随卡渐显
- **趣味 motif（科普推荐）**：数字滚动做成**里程表/翻牌**式向上跳动；到位瞬间冒一下高亮点或打个 ✓，升幅大的指标可加一个向上的小箭头窜一下。motif 元素自管 `removeAttribute`。
- 完成后对 card/stat/unit/card-label/accent-line `removeAttribute("data-anim-role")`，**不处理 title**

### sfx_cues（可选，趣味匹配）
`[{"at":0.3,"sfx":"pop","comment":"卡片弹出"},{"at":1.0,"sfx":"rise","comment":"数字飙升到位"}]`

## 运动层
**不产出独立运动**。通用运动负责 title。

## 视觉层
复用 `generated/visuals/dark_tech/`。

## 编译命令
```bash
python assembler/assemble.py --structure metric_cards --motion scale_pop_hold --visual dark_tech \
  --slots generated/templates/T18_metric_cards/example_inputs.json --duration 6.0 \
  --out generated/templates/T18_metric_cards/index.html
```

## 多样性变体矩阵（一个 prompt 产出 2 个模版，均 bound 各自内嵌脚本）
按 00「多样性变体矩阵」，产出 2 个空间布局变体（各独立 id，追加 `variant_of:"metric_cards"` + `variant_label`）：
- A｜`metric_cards` · 单行型：指标卡一行并排（「一行并排」）→ `T18_metric_cards`
- B｜`metric_cards_grid` · 矩阵型：2×2 卡片矩阵、大小错落散布（「矩阵散布」）→ `T18_metric_cards_grid`

## 验收
按 00 自检；确认 2/3/4 张卡等宽不溢出、整数与小数 count-up 都正确收尾、render 下数字确定停在终值；viewer 自动跳过通用运动。
