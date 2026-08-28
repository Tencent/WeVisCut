# T37 · historical_moment（类别5 叙事与过渡）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 historical_moment（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `slide_in_hold` 或 `mask_reveal`）。
> 视觉层复用现有（推荐 `warm_editorial` 或 `minimal_academic`）。
> mother_id = `T37_historical_moment`，duration 默认 5.5s。

## 结构层 generated/structures/historical_moment/
历史时刻卡。大年份数字 + 人物名 + 事件描述 + 时代标签。科普常见的"时间节点+人物+事件"叙事。

- DOM（`data-anim-role` / `data-slot`）：
  - 时代标签：小胶囊（anim-role=`eyebrow`, slot=`era_tag`）
  - 大年份：font-size 120~160px，`var(--accent)` 色，粗体（anim-role=`title`, slot=`year`）
  - 人物/主体名：font-size 40~56px（anim-role=`subtitle`, slot=`person_name`）
  - 事件描述：font-size 24~32px，`var(--ink-2)` 色（anim-role=`body`, slot=`event_desc`）
  - accent-line（anim-role=`accent-line`，装饰分割线）
- slot_schema：`era_tag`≤10、`year`≤8、`person_name`≤12、`event_desc`≤40
- default：era_tag="科学革命"、year="1905"、person_name="阿尔伯特·爱因斯坦"、event_desc="发表狭义相对论、光电效应论文，被称为"奇迹年""
- structure_card：use_for=[历史时刻, 人物故事, 时间节点]，spotlight_level=medium，duration_range=[4.5,6.5]

## 运动层
**不产出独立运动**。通用运动负责 eyebrow→title(年份)→subtitle(人物)→body(事件)→accent-line 的时序入场。

## 音效
- sfx_cues：`[{"at":0.3,"sfx":"whoosh","comment":"年份入场"},{"at":1.0,"sfx":"tick","comment":"人物出现"}]`

## 多样性变体矩阵（3 个模版）
- A｜`historical_moment` · 居中经典：年份居中+人物+事件纵向排列（variant_label「居中经典」）→ mother_id `T37_historical_moment`
- B｜`historical_moment_side` · 左年份右叙事：大年份占左 1/3，人物+事件在右 2/3（variant_label「左年右事」）→ `T37_historical_moment_side`
- C｜`historical_moment_card` · 卡片式：年份+人物做成卡片，事件在卡片下方（variant_label「卡片叙事」）→ `T37_historical_moment_card`

## 验收
按 00 自检；确认年份大数字渲染正确；40 字事件描述不溢出；时代标签可见。
