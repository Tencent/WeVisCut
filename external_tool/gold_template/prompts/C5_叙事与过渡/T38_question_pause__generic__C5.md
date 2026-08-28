# T38 · question_pause（类别5 叙事与过渡）　【narrative · 通用入场运动 · 夸张渲染模式】

> 先贴 `00_shared_preamble.md`。新增：**结构 question_pause（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `scale_pop_hold`）。
> **夸张渲染模式**：不使用 `.stage-panel`，全屏大问号 + 思考题。
> 视觉层复用现有（推荐 `dark_tech`）。
> mother_id = `T38_question_pause`，duration 默认 5.0s。

## 结构层 generated/structures/question_pause/
互动提问暂停点。大问号 + 思考题 + 短暂留白。用在视频中段引发观众思考。

- **夸张渲染**：不用 `.stage-panel`，全屏布局。
- DOM（`data-anim-role` / `data-slot`）：
  - 小标签：顶部（anim-role=`eyebrow`, slot=`eyebrow`）
  - 巨大问号 "❓"：font-size 140~180px（anim-role=`decoration`, 无 slot）
  - 思考题：font-size 56~72px，居中（anim-role=`title`, slot=`question_text`）
  - 引导语：font-size 24~28px，底部，`var(--ink-2)` 色（anim-role=`subtitle`, slot=`guide_text`）
- slot_schema：`eyebrow`≤8、`question_text`≤24、`guide_text`≤20
- default：eyebrow="想一想"、question_text="如果没有月球，地球会怎样？"、guide_text="暂停三秒，想想你的答案"
- structure_card：use_for=[互动提问, 思考暂停, 观众互动]，spotlight_level=high，duration_range=[4.0,6.0]，exaggerated_render=true

## 运动层
**不产出独立运动**。通用运动负责 decoration→title→subtitle 入场。

## 音效
- sfx_cues：`[{"at":0.3,"sfx":"pop","comment":"问号弹出"},{"at":1.5,"sfx":"tick","comment":"思考提示"}]`

## 多样性变体矩阵（2 个模版）
- A｜`question_pause` · 居中聚焦：问号和问题垂直居中（variant_label「居中聚焦」）→ mother_id `T38_question_pause`
- B｜`question_pause_corner` · 角落问号：大问号在左上角，问题在中下方（variant_label「角落问号」）→ `T38_question_pause_corner`

## 验收
按 00 自检；确认全屏无蒙版时对比度达标；问题文本不溢出。
