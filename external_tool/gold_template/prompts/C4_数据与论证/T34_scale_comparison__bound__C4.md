# T34 · scale_comparison（类别4 数据与论证）　【bound】

> 先贴 `00_shared_preamble.md`。新增：**结构 scale_comparison（family=argument）**。
> **Bound 结构**：内嵌 structure-script 实现尺度缩放动画。
> 视觉层复用现有（推荐 `clean_saas` 或 `dark_tech`）。
> mother_id = `T34_scale_comparison`，duration 默认 6.0s。

## 结构层 generated/structures/scale_comparison/
尺度对比可视化。"如果太阳是篮球大小，地球就只有玻璃珠大"。左右两个 emoji/圆形 + 大小比例 + 文字标注。

- DOM（`data-anim-role` / `data-slot`）：
  - 小标签：顶部（anim-role=`eyebrow`, slot=`eyebrow`）
  - 主标题（anim-role=`title`, slot=`headline`）
  - 左侧大物体：
    - emoji/圆形：根据 scale 渲染大尺寸，如 200px（anim-role=`decoration`, slot=`big_emoji`）
    - 标签：（anim-role=`list-item`, slot=`big_label`）
  - 右侧小物体：
    - emoji/圆形：根据比例渲染小尺寸，如 30px（anim-role=`decoration`, slot=`small_emoji`）
    - 标签：（anim-role=`list-item`, slot=`small_label`）
  - 比例文字：居中显示比例数值（anim-role=`body`, slot=`ratio_text`）
- slot_schema：`eyebrow`≤10、`headline`≤16、`big_emoji`≤2、`big_label`≤10、`small_emoji`≤2、`small_label`≤10、`ratio_text`≤20
- default：eyebrow="尺度感知"、headline="太阳与地球的大小差距"、big_emoji="☀️"、big_label="太阳（篮球）"、small_emoji="🌍"、small_label="地球（玻璃珠）"、ratio_text="体积是地球的 130 万倍"
- structure_card：use_for=[尺度对比, 大小比较, 量级感知]，spotlight_level=medium，duration_range=[5.0,7.0]

## 结构绑定动画（structure-script）
1. 大物体先出现：emoji scale 0→1（渐大）+ 标签淡入
2. 小物体出现：emoji scale 0→目标小尺寸 + 标签淡入
3. 比例文字 countup 式出现 + pulse 强调
4. 可选：大物体做轻微 pulse/glow 强调差异

## 音效
- sfx_cues：`[{"at":0.5,"sfx":"rise","comment":"大物体出现"},{"at":1.8,"sfx":"pop","comment":"小物体出现"},{"at":2.8,"sfx":"sparkle","comment":"比例揭示"}]`

## 多样性变体矩阵（3 个模版）
- A｜`scale_comparison` · 左右并排：大物体左+小物体右（variant_label「左右并排」）→ mother_id `T34_scale_comparison`
- B｜`scale_comparison_nested` · 嵌套式：小物体嵌入大物体内部（variant_label「嵌套对比」）→ `T34_scale_comparison_nested`
- C｜`scale_comparison_stack` · 上下堆叠：上方大物体+下方小物体（variant_label「上下堆叠」）→ `T34_scale_comparison_stack`

## 验收
按 00 自检；确认比例视觉正确；文字不溢出；不同尺度参数下渲染正常。
