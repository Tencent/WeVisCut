# 15 · line_chart × dark_tech　【dataviz · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 line_chart（family=dataviz，含内嵌动画脚本）+ 视觉 dark_tech（复用）**。
> **不产出独立运动层**。编译时搭配通用叙事运动（推荐 `slide_in_hold`）。
> mother_id = `15_line_dark`，duration 默认 7.0s。读 `references/data-in-motion.md`（SVG path + stroke-dashoffset 画线）。

## 数据契约
- slot `series` JSON：`[{"x":"1800","y":10},{"x":"1930","y":20},{"x":"1960","y":30},{"x":"1975","y":40},{"x":"1987","y":50},{"x":"2022","y":80}]`。
- 结构层用 **inline SVG** 画占位坐标系 + 占位折线/数据点；
  内嵌脚本读 `series` 重新计算真实 path 和数据点，再用 `stroke-dashoffset` 画出。

## 结构层 generated/structures/line_chart/
> 结构层必须内嵌默认数据的静态占位图形。

时间趋势折线镜头：`title` → SVG 图表区（`chart-axis` 网格 + `chart-area` 渐变填充 + `chart-line` 折线 + 若干 `chart-point` + `chart-value` 末点数值）→ `x_label`/`y_label` 轴名 → `source_note`。
- 线条 `stroke:var(--accent)`，面积 `fill` 用 `var(--accent)` 低透明；点用 `var(--accent-2)`
- slot_schema：`headline`≤18、`series`(json)、`x_label`≤8、`y_label`≤8、`source_note`≤24
- default：headline=「全球人口增长曲线」、series 用上方示例、x_label=「年份」、y_label=「亿人」
- structure_card：use_for=[折线图,趋势变化,时间序列]，spotlight_level=high，duration_range=[5.5,8.5]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init`：
- `chart-axis`：网格淡入
- 读 `series` → 清空占位 → 重建 SVG path / circles / area
- `chart-line`：`stroke-dashoffset` 全长→0 画出（power2.inOut，~1.4s）
- `chart-area`：线画完后淡入
- `chart-point`：沿线依次弹现（back.out，stagger）
- `chart-value`：末点数值渐显
- `chart-label`：x 轴标签淡入

### sfx_cues（写在 structure_card.json）
`[{"at":0.4,"sfx":"whoosh"},{"at":1.8,"sfx":"pop"}]`

## 运动层
**不产出独立运动**。编译时使用通用叙事运动（如 `slide_in_hold`）。

## 编译命令
```bash
python assembler/assemble.py --structure line_chart --motion slide_in_hold --visual dark_tech \
  --slots generated/templates/15_line_dark/example_inputs.json --duration 7.0 \
  --out generated/templates/15_line_dark/index.html
```

## 验收
按 00 自检；确认 path 在 render 下确定可 seek；3 组不同点数（4/7/12 点）不溢出。
