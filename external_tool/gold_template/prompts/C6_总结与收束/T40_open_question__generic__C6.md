# T40 · open_question（类别6 总结与收束）　【narrative · 通用入场运动 · 夸张渲染模式】

> 先贴 `00_shared_preamble.md`。新增：**结构 open_question（family=recap）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `scale_pop_hold`）。
> **夸张渲染模式**：不使用 `.stage-panel`，全屏大 "?" + 开放问题。
> 视觉层复用现有（推荐 `dark_tech`）。
> mother_id = `T40_open_question`，duration 默认 5.0s。

## 结构层 generated/structures/open_question/
开放问题/思想实验。收束时留给观众的思考题。全屏大 "?" + 思考性问题 + 底部引导语。

- **夸张渲染**：不用 `.stage-panel`，全屏布局。
- DOM（`data-anim-role` / `data-slot`）：
  - 小标签（anim-role=`eyebrow`, slot=`eyebrow`）
  - 巨大 "?" 符号：font-size 200px，`var(--accent)` 色半透明（anim-role=`decoration`, 无 slot）
  - 开放问题：font-size 48~64px，居中（anim-role=`title`, slot=`question_text`）
  - 引导语：font-size 22~28px，底部（anim-role=`subtitle`, slot=`guide_text`）
- slot_schema：`eyebrow`≤8、`question_text`≤30、`guide_text`≤20
- default：eyebrow="思想实验"、question_text="如果人类能活一千年，社会结构会如何变化？"、guide_text="评论区说说你的看法"
- structure_card：use_for=[开放问题, 思想实验, 互动收尾]，spotlight_level=high，duration_range=[4.0,6.0]，exaggerated_render=true

## 运动层
**不产出独立运动**。通用运动负责 decoration→title→subtitle 入场。

## 音效
- sfx_cues：`[{"at":0.3,"sfx":"whoosh","comment":"?符号入场"},{"at":1.2,"sfx":"sparkle","comment":"问题显现"}]`

## 多样性变体矩阵（2 个模版）
- A｜`open_question` · 居中聚焦：?居中+问题+引导纵向排列（variant_label「居中聚焦」）→ mother_id `T40_open_question`
- B｜`open_question_bg` · 背景大字：巨大 ? 做背景水印，问题在前景居中（variant_label「背景水印」）→ `T40_open_question_bg`

## 验收
按 00 自检；确认全屏无蒙版时对比度达标；30 字问题不溢出；非 CTA 风格。
