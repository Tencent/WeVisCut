# T04 · agenda_roadmap（类别1 开场与观点引入）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 agenda_roadmap（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `card_stack_sequence` 或 `slide_in_hold`），章节逐条 stagger 出现。
> 视觉层复用现有（推荐 `clean_saas`）。
> mother_id = `T04_agenda_roadmap`，duration 默认 6.5s。

## 结构层 generated/structures/agenda_roadmap/
内容目录/章节预告。左上 `title` → 下方 3~5 个章节节点纵向列表（每条带序号徽标 + 章节名）→ `accent-line`。
- DOM：
  - `title`：标题「本期目录 / 我们会讲」（anim-role=`title`, slot=`headline`）
  - 章节列表：3~5 条 `list-item`（anim-role=`list-item`，slot=`item1`…`item5`），每条左侧一个序号圆徽（01/02/03，CSS counter 或写死），右侧章节名；纵向 flex，行距舒适
  - `accent-line`：标题下短线（anim-role=`accent-line`）
- slot_schema：`headline`≤16、`item1`…`item5`≤18（item4/item5 可空）
- default：headline=「今天讲什么」、item1=「什么是熵」、item2=「为什么房间会越来越乱」、item3=「熵与时间箭头」、item4=「」、item5=「」
- structure_card：use_for=[议程目录,内容提纲,本期大纲]，spotlight_level=mid，duration_range=[5.0,8.0]

## 运动层
**不产出独立运动**。通用运动负责 title 入场 + list-item stagger（章节逐条 reveal）。

## 视觉层
复用 `generated/visuals/clean_saas/`。

## 编译命令
```bash
python assembler/assemble.py --structure agenda_roadmap --motion card_stack_sequence --visual clean_saas \
  --slots generated/templates/T04_agenda_roadmap/example_inputs.json --duration 6.5 \
  --out generated/templates/T04_agenda_roadmap/index.html
```

## 音效与趣味 motif
- **sfx_cues（写进 structure_card.json）**：`[{"at":0.5,"sfx":"tick","comment":"章节逐条出现"},{"at":1.2,"sfx":"tick"}]`
- **趣味 motif（可选）**：序号徽标 01/02/03 翻牌式跳出；当前章节高亮。bespoke 翻牌 → 内嵌 `structure-script`（变 bound）。

## 多样性变体矩阵（一个 prompt 产出 3 个模版）
按 00「多样性变体矩阵」，产出 3 个空间布局变体（各独立 id，追加 `variant_of:"agenda_roadmap"` + `variant_label`）：
- A｜`agenda_roadmap` · 纵列型：标题 + 纵向章节列表（「纵向列表」）→ `T04_agenda_roadmap`
- B｜`agenda_roadmap_grid` · 矩阵型：章节做卡片矩阵（2×2/3×1）散布（「卡片矩阵」）→ `T04_agenda_roadmap_grid`
- C｜`agenda_roadmap_timeline` · 横轴型：横向时间轴节点串联（「横向时间轴」）→ `T04_agenda_roadmap_timeline`

## 验收
按 00 自检；确认 3 条与 5 条两种数量都纵向不溢出、序号徽标对齐、空 item 不留空行。
