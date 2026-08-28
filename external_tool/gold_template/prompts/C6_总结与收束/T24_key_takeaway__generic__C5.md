# T24 · key_takeaway（类别5 场景展示与总结收束）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 key_takeaway（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `scale_pop_hold`），takeaway 靠 `list-item` stagger，结尾可强调。
> 视觉层复用现有（推荐 `warm_editorial` 或 `dark_tech`）。
> mother_id = `T24_key_takeaway`，duration 默认 5.5s。

## 结构层 generated/structures/key_takeaway/
总结核心结论。上方大结论句 → 下方 3 条 takeaway 要点（带勾选/序号）→ accent-line。
- DOM：
  - `title`：大结论句，font-size 60~84px，可含 `.accent`（anim-role=`title`, slot=`headline`）
  - 3 条 `list-item`（anim-role=`list-item`，slot=`takeaway1`/`takeaway2`/`takeaway3`），每条左侧勾选/序号徽标
  - `accent-line`：结论句下短线（anim-role=`accent-line`）
- slot_schema：`headline`≤22、`takeaway1/2/3`≤18
- default：headline=「记住这三点」、takeaway1=「复利的力量在于时间而非利率」、takeaway2=「大多数人高估一年、低估十年」、takeaway3=「开始比完美更重要」
- structure_card：use_for=[要点总结,核心收获,知识回顾]，spotlight_level=high，duration_range=[4.5,7.0]

## 运动层
**不产出独立运动**。通用运动负责 title 入场 + list-item stagger（要点逐条 reveal）+ accent-line；末条可由运动收尾保持。

## 视觉层
复用 `generated/visuals/warm_editorial/`。

## 编译命令
```bash
python assembler/assemble.py --structure key_takeaway --motion scale_pop_hold --visual warm_editorial \
  --slots generated/templates/T24_key_takeaway/example_inputs.json --duration 5.5 \
  --out generated/templates/T24_key_takeaway/index.html
```

## 音效与趣味 motif
- **sfx_cues（写进 structure_card.json）**：`[{"at":0.5,"sfx":"tick","comment":"要点逐条"},{"at":1.6,"sfx":"chime","comment":"结论落定"}]`
- **趣味 motif（可选）**：每条 takeaway 前的 ✓ 勾逐个描画出来；结论句收尾轻闪。bespoke 画勾 → 内嵌 `structure-script`（变 bound）。

## 多样性变体矩阵（一个 prompt 产出 3 个模版）
按 00「多样性变体矩阵」，产出 3 个空间布局变体（各独立 id，追加 `variant_of:"key_takeaway"` + `variant_label`）：
- A｜`key_takeaway` · 句+列型：大结论句 + 下方 3 条要点列（「结论+要点列」）→ `T24_key_takeaway`
- B｜`key_takeaway_cards` · 散布型：结论居中 + 3 张 takeaway 卡散布（「结论+散布卡」）→ `T24_key_takeaway_cards`
- C｜`key_takeaway_center` · 聚焦型：单条核心结论巨字居中（「单句聚焦」）→ `T24_key_takeaway_center`

## 验收
按 00 自检；确认结论句 22 字不溢出、3 条 takeaway 对齐不溢出、搭配非原配运动不报错。
