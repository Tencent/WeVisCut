# T30 · analogy_card（类别2 概念解释）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 analogy_card（family=concept）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `slide_in_hold`）。
> 视觉层复用现有（推荐 `clean_saas` 或 `warm_editorial`）。
> mother_id = `T30_analogy_card`，duration 默认 5.5s。

## 结构层 generated/structures/analogy_card/
科普最核心手法——类比卡。左半=抽象概念（术语+简述），中间大 "≈" 符号，右半=日常类比（emoji 图标+通俗说法）。

- DOM（`data-anim-role` / `data-slot`）：
  - 小标签：顶部（anim-role=`eyebrow`, slot=`eyebrow`）
  - 左区概念面板：
    - 概念名：font-size 48~60px（anim-role=`title`, slot=`concept_name`）
    - 概念简述：font-size 22~28px（anim-role=`body`, slot=`concept_desc`）
  - 中间符号 "≈"：font-size 80~120px，`var(--accent)` 色（anim-role=`accent-line`, 无 slot）
  - 右区类比面板：
    - emoji 图标：font-size 80~100px（anim-role=`decoration`, slot=`analogy_emoji`）
    - 类比说法：font-size 36~48px（anim-role=`subtitle`, slot=`analogy_text`）
- `.stage-panel` 内两列 grid 布局，中间 "≈" 绝对定位居中
- slot_schema（max_chars）：`eyebrow`≤10、`concept_name`≤8、`concept_desc`≤30、`analogy_emoji`≤2、`analogy_text`≤16
- default：eyebrow="类比理解"、concept_name="DNA"、concept_desc="储存生物遗传信息的双螺旋分子"、analogy_emoji="📖"、analogy_text="一本生命说明书"
- structure_card：use_for=[类比解释, 概念通俗化, 抽象→具象]，spotlight_level=medium，duration_range=[4.5,6.5]

## 运动层
**不产出独立运动**。通用运动负责入场：左区先入→"≈" 展开→右区入场。

## 音效
- sfx_cues：`[{"at":0.5,"sfx":"whoosh","comment":"左侧概念入场"},{"at":1.2,"sfx":"click","comment":"≈符号出现"},{"at":1.8,"sfx":"sparkle","comment":"类比揭示"}]`

## 多样性变体矩阵（3 个模版）
- A｜`analogy_card` · 左右对称：左概念+中≈+右类比（variant_label「左右对称」）→ mother_id `T30_analogy_card`
- B｜`analogy_card_vertical` · 上下布局：上概念→中间≈→下类比，纵向堆叠（variant_label「上下堆叠」）→ `T30_analogy_card_vertical`
- C｜`analogy_card_arrow` · 箭头引导：左概念→大箭头→右类比，强调转化方向（variant_label「箭头引导」）→ `T30_analogy_card_arrow`

## 验收
按 00 自检；确认概念名 8 字 + 类比 16 字不溢出；≈ 符号居中对齐；三层解耦。
