# T33 · myth_busting（类别4 数据与论证）　【bound · 夸张渲染模式】

> 先贴 `00_shared_preamble.md`。新增：**结构 myth_busting（family=argument）**。
> **Bound 结构**：内嵌 structure-script 实现误区→事实翻转动画。
> **夸张渲染模式**：不使用 `.stage-panel`，全屏戏剧性转折。
> 视觉层复用现有（推荐 `dark_tech`）。
> mother_id = `T33_myth_busting`，duration 默认 6.5s。

## 结构层 generated/structures/myth_busting/
误区纠正。先展示"错误认知"（大红 ✗ + 误区文字），翻转/擦除 reveal"正确事实"（大绿 ✓ + 事实文字）。

- **夸张渲染**：不用 `.stage-panel`，全屏布局。
- DOM（`data-anim-role` / `data-slot`）：
  - 小标签：顶部（anim-role=`eyebrow`, slot=`eyebrow`）
  - 误区面板（初始可见）：
    - 大 "✗" 符号：font-size 120px，红色 `color: #ef4444`（anim-role=`decoration`）
    - 误区文字：font-size 56~72px（anim-role=`title`, slot=`wrong_text`）
  - 事实面板（初始隐藏，structure-script 控制 reveal）：
    - 大 "✓" 符号：font-size 120px，绿色 `color: #22c55e`（anim-role=`decoration`）
    - 事实文字：font-size 56~72px（anim-role=`body`, slot=`right_text`）
- slot_schema（max_chars）：`eyebrow`≤10、`wrong_text`≤24、`right_text`≤24
- default：eyebrow="误区纠正"、wrong_text="人类只使用了大脑的10%"、right_text="大脑各区域全天候都在活动"
- structure_card：use_for=[误区纠正, 辟谣, 事实核查]，spotlight_level=high，duration_range=[5.5,7.5]，exaggerated_render=true

## 结构绑定动画（structure-script）
1. 误区阶段（0~3s）：✗ 从 scale 0 弹出 + 红色脉冲，误区文字淡入
2. 转折（3~3.5s）：整个误区面板 rotateY 90°（翻转中间态）
3. 事实阶段（3.5~5s）：从 rotateY -90° → 0°，✓ 弹出 + 绿色脉冲，事实文字淡入
4. removeAttribute("data-anim-role")

## 音效
- sfx_cues：`[{"at":0.5,"sfx":"pop","comment":"✗弹出"},{"at":3.2,"sfx":"whoosh","comment":"翻转"},{"at":3.8,"sfx":"chime","comment":"✓揭示"}]`

## 多样性变体矩阵（3 个模版）
- A｜`myth_busting` · 翻转式：误区→翻转→事实（variant_label「翻转揭示」）→ mother_id `T33_myth_busting`
- B｜`myth_busting_wipe` · 擦除式：误区被从左到右擦除，事实从下方推入（variant_label「擦除推入」）→ `T33_myth_busting_wipe`
- C｜`myth_busting_split` · 左右对比：左误区（红底）+ 右事实（绿底）同时展示（variant_label「左右对比」）→ `T33_myth_busting_split`

## 验收
按 00 自检；确认翻转动画流畅；24 字文本不溢出；✗/✓ 颜色对比度达标。
