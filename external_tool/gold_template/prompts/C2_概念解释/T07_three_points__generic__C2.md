# T07 · three_points（类别2 概念与信息解释）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 three_points（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `card_stack_sequence`），三张卡靠 `card` stagger 入场。
> 视觉层复用现有（推荐 `clean_saas`）。
> mother_id = `T07_three_points`，duration 默认 6.0s。

## 结构层 generated/structures/three_points/
三点并列解释。上 `title` → 下方三列卡片（每张：序号/图标 + 小标题 + 一句说明）→ accent-line。
- DOM：
  - `title`：标题（anim-role=`title`, slot=`headline`）
  - 三张 `card`（anim-role=`card`），用 `display:grid; grid-template-columns: repeat(3,1fr); gap`：
    - 卡内：序号徽标（01/02/03，CSS 写死）+ 小标题（slot=`point1_title`…）+ 说明（slot=`point1_desc`…）
  - `accent-line`：标题下短线（anim-role=`accent-line`）
  - 卡片用 `--stage-bg`/`--radius-card`/`--shadow-soft`
- slot_schema：`headline`≤16；`pointN_title`≤8；`pointN_desc`≤22
- default：headline=「理解量子纠缠的三把钥匙」、point1_title=「叠加态」、point1_desc=「粒子可以同时处于多个状态」、point2_title=「测量坍缩」、point2_desc=「观察会让叠加态随机坍缩为一种」、point3_title=「非局域关联」、point3_desc=「纠缠粒子即使相隔光年也瞬间关联」
- structure_card：use_for=[多要点解释,概念分解,三个关键点]，spotlight_level=mid，duration_range=[5.0,8.0]

## 运动层
**不产出独立运动**。通用运动负责 title 入场 + card stagger（三张卡逐张 reveal）+ accent-line。

## 视觉层
复用 `generated/visuals/clean_saas/`。

## 编译命令
```bash
python assembler/assemble.py --structure three_points --motion card_stack_sequence --visual clean_saas \
  --slots generated/templates/T07_three_points/example_inputs.json --duration 6.0 \
  --out generated/templates/T07_three_points/index.html
```

## 音效与趣味 motif
- **sfx_cues（写进 structure_card.json）**：`[{"at":0.4,"sfx":"tick","comment":"卡片逐张"},{"at":1.0,"sfx":"pop","comment":"第三张落定"}]`
- **趣味 motif（可选）**：每张卡序号 01/02/03 数字翻动；卡内小图标弹现。bespoke → 内嵌 `structure-script`（变 bound）。

## 多样性变体矩阵（一个 prompt 产出 3 个模版）
按 00「多样性变体矩阵」，产出 3 个空间布局变体（各独立 id，追加 `variant_of:"three_points"` + `variant_label`）：
- A｜`three_points` · 三列型：三张卡并排（「三列并排」）→ `T07_three_points`
- B｜`three_points_diagonal` · 散布型：三卡对角错落散布（「对角错落」）→ `T07_three_points_diagonal`
- C｜`three_points_stack` · 纵列型：纵向编号堆叠条（「纵向堆叠」）→ `T07_three_points_stack`

## 验收
按 00 自检；确认三列等宽对齐、长说明换行卡片等高、搭配非原配运动不报错。
