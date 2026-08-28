# T29 · fun_fact（类别1 开场与钩子）　【narrative · 通用入场运动 · 夸张渲染模式】

> 先贴 `00_shared_preamble.md`。新增：**结构 fun_fact（family=hook）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `scale_pop_hold`）。
> **夸张渲染模式**：不使用 `.stage-panel`，全屏大 emoji + 惊人事实。
> 视觉层复用现有（推荐 `dark_tech`）。
> mother_id = `T29_fun_fact`，duration 默认 5.0s。

## 结构层 generated/structures/fun_fact/
冷知识弹出卡。全屏无蒙版，超大 emoji + 惊人数字/事实 + 一句解释。类似 stat_highlight 但更趣味化。

- **夸张渲染**：不用 `.stage-panel`，全屏布局。
- DOM（`data-anim-role` / `data-slot`）：
  - 超大 emoji 图标：font-size 120~180px，居中偏上（anim-role=`decoration`, slot=`emoji_icon`）
  - 事实主文：font-size 80~100px，`var(--accent)` 色，粗体（anim-role=`title`, slot=`fact_text`）
  - 解释副文：font-size 28~36px，`var(--ink-2)` 色（anim-role=`subtitle`, slot=`explain_text`）
  - 小标签：左上角（anim-role=`eyebrow`, slot=`eyebrow`）
- slot_schema（max_chars）：`eyebrow`≤8、`emoji_icon`≤2、`fact_text`≤16、`explain_text`≤30
- default：eyebrow="冷知识"、emoji_icon="🐙"、fact_text="章鱼有三颗心脏"、explain_text="两颗负责鳃部供血，一颗负责全身循环"
- structure_card：use_for=[冷知识, 趣味事实, 惊人发现]，spotlight_level=high，duration_range=[4.0,6.0]，exaggerated_render=true

## 运动层
**不产出独立运动**。通用运动负责 decoration/title/subtitle/eyebrow 入场。

## 音效
- sfx_cues：`[{"at":0.4,"sfx":"pop","comment":"emoji弹出"},{"at":1.0,"sfx":"sparkle","comment":"事实高亮"}]`

## 多样性变体矩阵（3 个模版）
- A｜`fun_fact` · 居中聚焦：emoji 和事实垂直居中堆叠（variant_label「居中聚焦」）→ mother_id `T29_fun_fact`
- B｜`fun_fact_left` · 左对齐：emoji 在左大区域，事实和解释在右侧（variant_label「左图右文」）→ `T29_fun_fact_left`
- C｜`fun_fact_stack` · 纵向铺满：emoji 占上半屏，文字占下半屏（variant_label「上下铺满」）→ `T29_fun_fact_stack`

## 验收
按 00 自检；确认 emoji 在各平台渲染一致；16 字事实不溢出；全屏无蒙版下对比度达标。
