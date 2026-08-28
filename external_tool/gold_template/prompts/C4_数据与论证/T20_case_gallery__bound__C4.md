# T20 · case_gallery（类别4 对比与证据展示）　【dataviz · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 case_gallery（family=dataviz，含内嵌动画脚本）**。
> **不产出独立运动**，宫格 reveal + 逐个 active zoom 内嵌在 `<script data-hf="structure-script">`。
> 编译时搭配通用叙事运动（推荐 `scale_pop_hold`，仅负责 title）。
> 视觉层复用现有（推荐 `clean_saas` 或 `warm_editorial`）。
> mother_id = `T20_case_gallery`，duration 默认 7.0s。先读 `references/data-in-motion.md` + `references/css-patterns.md`。

## 数据契约
- slot `cases`（JSON）：4~6 个案例，default
  `[{"caption":"干涉条纹 🌊"},{"caption":"光电效应 ⚡"},{"caption":"双缝实验 🔬"},{"caption":"康普顿散射 💫"}]`
- 结构层渲染**静态宫格**（2×2 / 3×2 纯色块 + emoji + 文字单元格），不搭配运动时画面仍完整。
- 内嵌脚本读 `cases` 生成/填充每格 `cell` + `cell-label`，再 grid reveal + 逐个 active zoom 强调。

## 结构层 generated/structures/case_gallery/
> 结构层必须内嵌默认数据的静态宫格（纯色块 + emoji + 文字单元格），确保不搭配运动时画面仍完整可见。

宫格陈列。上 `title` → 2×2 / 3×2 宫格（每格 `cell` + 底部 `cell-label`）→ `accent-line`。
- anim-role：`title` / `cell`（每格）/ `cell-label`（每格标签）/ `accent-line`
- 宫格用 CSS grid 等比格；每格为纯色块背景 + 居中 emoji + 底部文字标签，色块用 `var(--accent)`/`var(--accent-2)`/`var(--ink-3)` 低透明，圆角 `--radius-card`
- slot_schema：`headline`≤16、`cases`(json)；`data_slots:["cases"]`
- default：headline=「光的波粒二象性证据」、cases 用上方示例
- structure_card：display_name=「案例宫格」，use_for=[多案例展示,现象分类,证据宫格]，spotlight_level=mid，duration_range=[5.5,8.5]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- 读 `cases` → 清空占位 → 生成每格 `cell` + `cell-label`
- `cell`：scale .9→1 + opacity，stagger 0.1（grid reveal 波次）
- `cell-label`：随各自 cell 由下滑入
- 收尾：逐个 `cell` 轻微 scale 1→1.04→1 active 强调（stagger 顺序），其余略降透明
- 完成后对 cell/cell-label/accent-line `removeAttribute("data-anim-role")`，**不处理 title**

### sfx_cues（可选）
`[{"at":0.3,"sfx":"pop"},{"at":1.2,"sfx":"tick"}]`

## 运动层
**不产出独立运动**。通用运动负责 title。

## 视觉层
复用 `generated/visuals/clean_saas/`。

## 编译命令
```bash
python assembler/assemble.py --structure case_gallery --motion scale_pop_hold --visual clean_saas \
  --slots generated/templates/T20_case_gallery/example_inputs.json --duration 7.0 \
  --out generated/templates/T20_case_gallery/index.html
```

## 多样性变体矩阵（一个 prompt 产出 2 个模版，均 bound 各自内嵌脚本）
按 00「多样性变体矩阵」，产出 2 个空间布局变体（各独立 id，追加 `variant_of:"case_gallery"` + `variant_label`）：
- A｜`case_gallery` · 宫格型：2×2 / 3×2 等比宫格（「规整宫格」）→ `T20_case_gallery`
- B｜`case_gallery_masonry` · 瀑布型：大小错落瀑布流散布、逐个 active 放大（「错落瀑布」）→ `T20_case_gallery_masonry`

## 验收
按 00 自检；确认 4 格（2×2）与 6 格（3×2）都不溢出、格内渐变占位无破图、render 下 reveal/zoom 确定性；viewer 自动跳过通用运动。
