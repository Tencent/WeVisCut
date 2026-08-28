# 14 · bar_chart × clean_saas　【dataviz · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 bar_chart（family=dataviz，含内嵌动画脚本）+ 视觉 clean_saas（已存在则复用）**。
> **不产出独立运动层**，图表动画内嵌在结构的 `<script data-hf="structure-script">` 中。
> 编译时搭配一个通用叙事运动（推荐 `scale_pop_hold`）。
> mother_id = `14_bar_saas`，duration 默认 6.5s。先读 `references/data-in-motion.md` + `references/css-patterns.md`。

## 数据契约（dataviz 关键）
- slot `series` 是 **JSON 数据槽**：`slot_schema` 里 `"series": {"type":"json", ...}`，default 给一组真实感示例，如
  `[{"label":"O","value":46},{"label":"Si","value":28},{"label":"Al","value":8},{"label":"Fe","value":5},{"label":"Ca","value":4}]`。
- 结构层渲染**占位骨架**（坐标轴容器 + 固定 N 根占位 bar）；
  结构内嵌脚本读 `series` 数据（`window.__hyperframes.getVariables().series`）生成/填充每根 `chart-bar`，再做生长动画。

## 结构层 generated/structures/bar_chart/
> 结构层必须内嵌默认数据的静态占位图形（柱/标签），确保不搭配运动时画面仍完整可见。

类目柱状图镜头：上 `title` → 主体图表区（`chart-axis` 基线/网格 + 若干 `chart-bar`，每根带 `chart-value` 数值标签 + `chart-label` 类目名）→ 底 `source_note` + `accent-line`。
- anim-role：`title` / `chart-axis` / `chart-bar`（每根）/ `chart-value`（每根）/ `chart-label`（每根）/ `accent-line`
- 每根 bar：`transform-origin:bottom`；颜色 `var(--accent)`，可交替 `var(--accent-2)`；圆角取小值
- slot_schema：`headline`≤18、`series`(json)、`unit`≤4、`source_note`≤24；`data_slots:["series"]`
- default：headline=「各元素在地壳中的丰度百分比」、series 用上方示例
- structure_card：use_for=[柱状图,数量对比,排名可视化]，spotlight_level=high，duration_range=[5.0,8.0]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- `chart-axis`：scaleX/opacity 画出基线（power2.out）
- 读 `series` → 清空占位 → 创建每根 bar DOM → 设高度
- `chart-bar`：`from {scaleY:0}` 逐根 stagger 升起（back.out(1.2)，stagger 0.08）
- `chart-value`：opacity+y 渐显
- `chart-label`：stagger 渐显
- **不处理叙事角色**（title/eyebrow 等由通用运动负责）

### sfx_cues（写在 structure_card.json）
`[{"at":0.35,"sfx":"whoosh"},{"at":1.1,"sfx":"pop","comment":"末根数值蹦出"}]`

## 运动层
**不产出独立运动**。编译时使用通用叙事运动（如 `scale_pop_hold`），负责 title/accent-line 等入场。

## 视觉层 generated/visuals/clean_saas/
已存在则**直接复用**，不重造。

## 编译命令
```bash
python assembler/assemble.py --structure bar_chart --motion scale_pop_hold --visual clean_saas \
  --slots generated/templates/14_bar_saas/example_inputs.json --duration 6.5 \
  --out generated/templates/14_bar_saas/index.html
```

## 验收
按 00 自检；额外确认：3 组不同 series（3/5/7 根、含小数/大数）都不溢出、bar 不超出图表区；render 下生长确定。
