# T06 · definition_card（类别2 概念与信息解释）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 definition_card（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `slide_in_hold`），关键词标签靠 `list-item` stagger。
> 视觉层复用现有（推荐 `minimal_academic` 或 `clean_saas`）。
> mother_id = `T06_definition_card`，duration 默认 5.5s。

## 结构层 generated/structures/definition_card/
术语/概念解释卡。单面板居中：小标签 → 概念名（大）→ 定义正文（多行）→ 关键词标签行 → accent-line。
- DOM：
  - `eyebrow`：标签「定义 / 概念」（anim-role=`eyebrow`, slot=`eyebrow`）
  - 概念名：font-size 64~88px，可含 `.accent`（anim-role=`title`, slot=`term`）
  - 定义正文：2~3 行说明，font-size ≥24px，行高舒适（anim-role=`subtitle`, slot=`definition`）
  - 关键词标签：2~3 个小 pill（anim-role=`list-item`，slot=`keyword1`/`keyword2`/`keyword3`）
  - `accent-line`：分隔短线（anim-role=`accent-line`）
- slot_schema：`eyebrow`≤10、`term`≤14、`definition`≤44、`keyword1/2/3`≤8
- default：eyebrow=「名词解释」、term=「熵」、definition=「热力学中衡量系统混乱程度的物理量」、keyword1=「不可逆」、keyword2=「概率」、keyword3=「信息论」
- structure_card：use_for=[术语解释,概念定义,名词科普]，spotlight_level=mid，duration_range=[4.5,7.0]

## 运动层
**不产出独立运动**。通用运动负责 eyebrow→title→subtitle→list-item(stagger)→accent-line。

## 视觉层
复用 `generated/visuals/minimal_academic/`。

## 编译命令
```bash
python assembler/assemble.py --structure definition_card --motion slide_in_hold --visual minimal_academic \
  --slots generated/templates/T06_definition_card/example_inputs.json --duration 5.5 \
  --out generated/templates/T06_definition_card/index.html
```

## 音效与趣味 motif
- **sfx_cues（写进 structure_card.json）**：`[{"at":0.3,"sfx":"pop","comment":"概念名弹出"},{"at":1.2,"sfx":"tick","comment":"关键词逐个"}]`
- **趣味 motif（可选）**：术语下高亮块/下划线擦出强调。bespoke 擦出 → 内嵌 `structure-script`（变 bound）。

## 多样性变体矩阵（一个 prompt 产出 3 个模版）
按 00「多样性变体矩阵」，产出 3 个空间布局变体（各独立 id，追加 `variant_of:"definition_card"` + `variant_label`）：
- A｜`definition_card` · 聚焦型：居中卡（概念名+定义+关键词行）（「居中卡片」）→ `T06_definition_card`
- B｜`definition_card_split` · 分栏型：左术语 + 右定义体（「左右分栏」）→ `T06_definition_card_split`
- C｜`definition_card_float` · 散布型：关键词标签 `position:absolute` 浮动散布概念名四周（「关键词浮动」）→ `T06_definition_card_float`

## 验收
按 00 自检；确认定义正文 44 字换行不溢出面板、关键词 pill 横向居中不错位。
