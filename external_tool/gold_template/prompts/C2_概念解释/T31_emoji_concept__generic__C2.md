# T31 · emoji_concept（类别2 概念解释）　【narrative · 通用入场运动 · 夸张渲染模式】

> 先贴 `00_shared_preamble.md`。新增：**结构 emoji_concept（family=concept）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `scale_pop_hold`）。
> **夸张渲染模式**：不使用 `.stage-panel`，全屏一个超大 emoji + 概念名 + 一句话解释。
> 视觉层复用现有（推荐 `dark_tech`）。
> mother_id = `T31_emoji_concept`，duration 默认 4.5s。

## 结构层 generated/structures/emoji_concept/
全屏夸张渲染：一个超大 emoji（150~200px）+ 下方概念名 + 一句话解释。适合快节奏知识点串联。

- **夸张渲染**：不用 `.stage-panel`，全屏布局。
- DOM（`data-anim-role` / `data-slot`）：
  - 超大 emoji：font-size 150~200px，居中偏上（anim-role=`decoration`, slot=`emoji_icon`）
  - 概念名：font-size 64~80px，居中，粗体（anim-role=`title`, slot=`concept_name`）
  - 一句话解释：font-size 28~36px，居中偏下（anim-role=`subtitle`, slot=`explain_text`）
- slot_schema（max_chars）：`emoji_icon`≤2、`concept_name`≤8、`explain_text`≤24
- default：emoji_icon="🧬"、concept_name="基因突变"、explain_text="DNA复制时的随机抄写错误"
- structure_card：use_for=[快节奏概念, emoji科普, 单概念展示]，spotlight_level=high，duration_range=[3.5,5.5]，exaggerated_render=true

## 运动层
**不产出独立运动**。通用运动负责 decoration→title→subtitle 的 stagger 入场。

## 音效
- sfx_cues：`[{"at":0.3,"sfx":"pop","comment":"emoji弹出"},{"at":0.8,"sfx":"tick","comment":"概念名出现"}]`

## 多样性变体矩阵（3 个模版）
- A｜`emoji_concept` · 居中堆叠：emoji→概念名→解释纵向居中（variant_label「居中堆叠」）→ mother_id `T31_emoji_concept`
- B｜`emoji_concept_side` · 左图右文：左大 emoji + 右侧概念名和解释（variant_label「左图右文」）→ `T31_emoji_concept_side`
- C｜`emoji_concept_bottom` · 底部锚定：emoji 在中上，概念名和解释锚定底部（variant_label「底部锚定」）→ `T31_emoji_concept_bottom`

## 验收
按 00 自检；确认 emoji 渲染正常；概念名 8 字不溢出；全屏无蒙版对比度。
