# T03 · thesis_statement（类别1 开场与观点引入）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 thesis_statement（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `slide_in_hold` 或 `mask_reveal`）。
> 视觉层复用现有（推荐 `warm_editorial` 或 `minimal_academic`）。
> mother_id = `T03_thesis_statement`，duration 默认 5.5s。

## 结构层 generated/structures/thesis_statement/
亮明核心观点。垂直居中：小标签 → 大观点句 → 简短解释 → accent-line。
- DOM：
  - `eyebrow`：上方标签「核心观点 / 本期主张」（anim-role=`eyebrow`, slot=`eyebrow`）
  - 观点句：font-size 76~104px，句中关键词用 `.accent`（颜色 `var(--accent)`，可选发光）（anim-role=`title`, slot=`thesis_text`）
  - 解释行：一行小字补充（anim-role=`subtitle`, slot=`explain_text`）
  - `accent-line`：观点句下短线（anim-role=`accent-line`）
- slot_schema：`eyebrow`≤14、`thesis_text`≤26、`explain_text`≤30
- default：eyebrow=「核心论点」、thesis_text=「复利不是第八大奇迹，但它解释了大多数财富差距」、explain_text=「让我们用三个模型来拆解这个被神化的概念」
- structure_card：use_for=[核心论点,观点陈述,科普立论]，spotlight_level=high，duration_range=[4.5,7.0]

## 运动层
**不产出独立运动**。通用运动负责 eyebrow→title→subtitle→accent-line 的入场。

## 视觉层
复用 `generated/visuals/warm_editorial/`（或任意现有皮肤）。

## 编译命令
```bash
python assembler/assemble.py --structure thesis_statement --motion slide_in_hold --visual warm_editorial \
  --slots generated/templates/T03_thesis_statement/example_inputs.json --duration 5.5 \
  --out generated/templates/T03_thesis_statement/index.html
```

## 音效与趣味 motif
- **sfx_cues（写进 structure_card.json）**：`[{"at":0.3,"sfx":"whoosh","comment":"观点句滑入"},{"at":1.0,"sfx":"chime","comment":"观点落定"}]`
- **趣味 motif（可选）**：观点句里的 `.accent` 关键词用荧光笔下划线擦出/高亮块亮起。bespoke 擦出动画 → 内嵌 `structure-script`（变 bound）。

## 多样性变体矩阵（一个 prompt 产出 3 个模版）
按 00「多样性变体矩阵」，产出 3 个空间布局变体（各独立 id，追加 `variant_of:"thesis_statement"` + `variant_label`）：
- A｜`thesis_statement` · 聚焦型：居中大观点句 + 下方解释行（「居中聚焦」）→ `T03_thesis_statement`
- B｜`thesis_statement_split` · 分栏型：左侧巨大观点句 + 右侧解释栏（「左右分栏」）→ `T03_thesis_statement_split`
- C｜`thesis_statement_fullbleed` · 全屏型：全屏铺满巨字观点 + 底部解释条 + 角标 eyebrow（「全屏巨字」）→ `T03_thesis_statement_fullbleed`

## 验收
按 00 自检；确认观点句 26 字不溢出、`.accent` 关键词颜色取自令牌、解释行不喧宾夺主。
