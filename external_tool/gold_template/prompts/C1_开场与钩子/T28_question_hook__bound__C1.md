# T28 · question_hook（类别1 开场与钩子）　【bound · 夸张渲染模式】

> 先贴 `00_shared_preamble.md`。新增：**结构 question_hook（family=hook）**。
> **Bound 结构**：内嵌 structure-script 实现答案延迟 reveal 动画。
> **夸张渲染模式**：不使用 `.stage-panel`，内容直接铺在 `.bg-stage` 上，全屏冲击。
> 视觉层复用现有（推荐 `dark_tech` 或 `neon_cyber`）。
> mother_id = `T28_question_hook`，duration 默认 6.0s。

## 结构层 generated/structures/question_hook/
"你知道吗？"式问答钩子开场。全屏无蒙版，巨大问号 emoji + 大字疑问句 + 下方答案预告（延迟 reveal）。

- **夸张渲染**：不用 `.stage-panel`，结构直接在全屏区域布局（`.bg-stage` 的子级）。
- DOM（`data-anim-role` / `data-slot`）：
  - 巨大 emoji 问号「❓」：font-size 160px，居上区域（anim-role=`decoration`, 无 slot，纯装饰）
  - 问句主标题：font-size 100~130px，居中偏上，`var(--ink-1)` 色（anim-role=`title`, slot=`question_text`）
  - 答案预告：font-size 36~48px，底部区域，初始 opacity=0 由 structure-script 控制 reveal（anim-role=`body`, slot=`answer_hint`）
  - 小标签：上方角落（anim-role=`eyebrow`, slot=`eyebrow`）
- slot_schema（max_chars）：`eyebrow`≤10、`question_text`≤20、`answer_hint`≤30
- default：eyebrow="你知道吗"、question_text="人一生心跳约 25 亿次"、answer_hint="平均每分钟 72 次，日夜不停跳动 80 年"
- structure_card：use_for=[问答式开场, 你知道吗, 惊人事实引入]，spotlight_level=high，duration_range=[5.0,7.0]，exaggerated_render=true

## 结构绑定动画（structure-script）
1. 问号 emoji scale 0→1 + rotation -15°→0°，弹性缓动（back.out）
2. 问句 title 从 opacity 0 + y+40 → 显现
3. 延迟 2.5s 后，answer_hint 从 opacity 0 + blur(10px) → 清晰显现（reveal 效果）
4. 完成后 removeAttribute("data-anim-role") 避免通用运动重复

## 音效
- sfx_cues：`[{"at":0.3,"sfx":"pop","comment":"问号弹出"},{"at":2.8,"sfx":"sparkle","comment":"答案揭晓"}]`

## 多样性变体矩阵（3 个模版）
- A｜`question_hook` · 居中聚焦：问号和问句垂直居中（variant_label「居中聚焦」）→ mother_id `T28_question_hook`
- B｜`question_hook_top` · 顶部布局：问号和问句在上 1/3，答案在下 2/3 大字显示（variant_label「上问下答」）→ `T28_question_hook_top`
- C｜`question_hook_split` · 左右分栏：左大问号+右问句，底部答案横贯（variant_label「左右分栏」）→ `T28_question_hook_split`

## 验收
按 00 自检；确认全屏无 `.stage-panel` 时内容不溢出；答案 reveal 动画正常；搭配不同 visual 主题效果一致。
