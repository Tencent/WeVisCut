# 16 · donut_pie × minimal_academic　【dataviz · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 donut_pie（family=dataviz，含内嵌动画脚本）+ 视觉 minimal_academic（若未生成则一并产出）**。
> **不产出独立运动层**。编译时搭配通用叙事运动（推荐 `mask_reveal`）。
> mother_id = `16_donut_academic`，duration 默认 6.5s。读 `references/data-in-motion.md`。

## 数据契约
- slot `series` JSON：`[{"label":"N₂","value":78},{"label":"O₂","value":21},{"label":"Ar","value":0.9},{"label":"CO₂","value":0.04},{"label":"其他","value":0.06}]`。
- 结构层用 inline SVG 环形占位；内嵌脚本读 `series` 计算各片弧长并扫出。

## 结构层 generated/structures/donut_pie/
> 结构层必须内嵌默认数据的静态占位图形。

构成占比镜头：`title` → 环形图（多 `chart-slice` + 环心 `chart-value`/`chart-label`）+ `chart-legend`（每项 label+占比）→ `source_note`。
- slice 颜色：`var(--accent)`/`var(--accent-2)`/`var(--ink-3)`/`var(--success)`（最多 4~6 片）
- slot_schema：`headline`≤18、`series`(json)、`center_value`≤6、`center_label`≤10、`source_note`≤24
- default：headline=「地球大气成分占比」、series 用上方示例
- structure_card：use_for=[饼图/环形图,占比分布,成分构成]，spotlight_level=high，duration_range=[5.0,8.0]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init`：
- 读 `series` → 清空占位 → 创建 SVG 环形切片
- `chart-slice`：按角度依次扫出（stroke-dashoffset，power2.out，stagger 0.12）
- `center_value`：countup 0→末值
- `chart-legend`：右侧逐项 stagger 渐显

### sfx_cues（写在 structure_card.json）
`[{"at":0.3,"sfx":"whoosh"},{"at":1.5,"sfx":"sparkle"}]`

## 运动层
**不产出独立运动**。编译时使用通用叙事运动（如 `mask_reveal`）。

## 编译命令
```bash
python assembler/assemble.py --structure donut_pie --motion mask_reveal --visual minimal_academic \
  --slots generated/templates/16_donut_academic/example_inputs.json --duration 6.5 \
  --out generated/templates/16_donut_academic/index.html
```

## 验收
按 00 自检；确认占比总和归一化正确、4~6 片不重叠、环心数值不溢出。
