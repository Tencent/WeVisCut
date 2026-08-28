# T05 · chapter_transition（类别1 开场与观点引入）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 chapter_transition（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `mask_reveal` 或 `slide_in_hold`），配擦出/滑入转场感。
> 视觉层复用现有（推荐 `dark_tech`）。
> mother_id = `T05_chapter_transition`，duration 默认 4.5s。

## 结构层 generated/structures/chapter_transition/
段落转场。超大编号 + 小标题 + 一行说明。左侧巨大章节号（如 `02`），右侧标题与副标题；底部 accent-line。
- DOM：
  - 巨大编号：font-size 180~260px，低对比/描边或填充 `var(--accent)`（anim-role=`badge`, slot=`chapter_no`）
  - 章节标题：font-size 64~88px（anim-role=`title`, slot=`chapter_title`）
  - 副标题：一行说明（anim-role=`subtitle`, slot=`chapter_sub`）
  - `accent-line`：标题下短线（anim-role=`accent-line`）
  - 布局：可用 `display:grid; grid-template-columns: auto 1fr` 让编号与文字左右排
- slot_schema：`chapter_no`≤3、`chapter_title`≤14、`chapter_sub`≤22
- default：chapter_no=「02」、chapter_title=「微观世界」、chapter_sub=「从原子到夸克」
- structure_card：use_for=[章节过渡,段落切换,内容分段]，spotlight_level=mid，duration_range=[3.5,6.0]

## 运动层
**不产出独立运动**。通用运动负责 badge（编号）/title/subtitle/accent-line 的入场（`mask_reveal` 擦出感最佳）。

## 视觉层
复用 `generated/visuals/dark_tech/`。

## 编译命令
```bash
python assembler/assemble.py --structure chapter_transition --motion mask_reveal --visual dark_tech \
  --slots generated/templates/T05_chapter_transition/example_inputs.json --duration 4.5 \
  --out generated/templates/T05_chapter_transition/index.html
```

## 音效与趣味 motif
- **sfx_cues（写进 structure_card.json）**：`[{"at":0.2,"sfx":"whoosh","comment":"编号擦入"},{"at":0.8,"sfx":"chime","comment":"标题落定"}]`
- **趣味 motif（可选）**：巨大编号像翻页/里程表跳到当前章号。bespoke 跳号 → 内嵌 `structure-script`（变 bound）。

## 多样性变体矩阵（一个 prompt 产出 3 个模版）
按 00「多样性变体矩阵」，产出 3 个空间布局变体（各独立 id，追加 `variant_of:"chapter_transition"` + `variant_label`）：
- A｜`chapter_transition` · 分栏型：左巨大编号 + 右标题副标题（「左号右题」）→ `T05_chapter_transition`
- B｜`chapter_transition_center` · 聚焦型：居中巨号叠半透明 + 标题压在其上（「居中叠字」）→ `T05_chapter_transition_center`
- C｜`chapter_transition_side` · 侧边型：左侧竖排编号条 + 主区标题（「侧边竖排」）→ `T05_chapter_transition_side`

## 验收
按 00 自检；确认巨大编号不压标题、左右栅格在长标题下不挤压；搭配 `slide_in_hold` 也成立。
