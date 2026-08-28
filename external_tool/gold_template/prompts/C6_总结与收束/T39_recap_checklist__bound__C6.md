# T39 · recap_checklist（类别6 总结与收束）　【bound】

> 先贴 `00_shared_preamble.md`。新增：**结构 recap_checklist（family=recap）**。
> **Bound 结构**：内嵌 structure-script 实现逐条 checkmark pop 动画。
> 视觉层复用现有（推荐 `clean_saas` 或 `warm_editorial`）。
> mother_id = `T39_recap_checklist`，duration 默认 7.0s。

## 结构层 generated/structures/recap_checklist/
要点回顾清单。收束时逐条回顾要点，每条带 ✓ 标记逐个点亮。

- DOM（`data-anim-role` / `data-slot`）：
  - 小标签（anim-role=`eyebrow`, slot=`eyebrow`）
  - 主标题（anim-role=`title`, slot=`headline`）
  - 清单容器：纵向列表，每条含：
    - ✓ 标记圆形（初始灰色空圆，点亮后绿色填充 + checkmark SVG）
    - 要点文字（anim-role=`list-item`）
  - 数据来自 slot `items`（JSON 数组），structure-script 动态渲染
- slot_schema：`eyebrow`≤10、`headline`≤12、`items`= JSON 字符串数组，3~5 项，每项≤30字
- default：eyebrow="本期回顾"、headline="记住这三点"、items=["熵是混乱度的度量","熵只增不减（热力学第二定律）","这就是时间箭头的方向"]
- structure_card：use_for=[要点回顾, 知识清单, 总结列表]，spotlight_level=medium，duration_range=[6.0,8.0]

## 结构绑定动画（structure-script）
1. 标题入场
2. 逐条 reveal：
   - 每条：文字从 opacity 0 + x-20 滑入 → ✓ 标记从灰色空圆 scale 0→1 + 填充绿色 + checkmark pop（overshoot 弹性）
   - 间隔 1.0~1.5s
3. 全部完成后：所有 ✓ 同时轻 pulse 一下
4. removeAttribute("data-anim-role")

## 音效
- sfx_cues：`[{"at":1.0,"sfx":"tick","comment":"第一条✓"},{"at":2.2,"sfx":"tick","comment":"第二条✓"},{"at":3.4,"sfx":"tick","comment":"第三条✓"},{"at":4.5,"sfx":"chime","comment":"全部完成"}]`

## 多样性变体矩阵（3 个模版）
- A｜`recap_checklist` · 居中列表：垂直居中列表（variant_label「居中列表」）→ mother_id `T39_recap_checklist`
- B｜`recap_checklist_cards` · 卡片式：每条做成独立小卡片（variant_label「卡片清单」）→ `T39_recap_checklist_cards`
- C｜`recap_checklist_grid` · 网格式：2×2 或 3列网格排列（variant_label「网格排列」）→ `T39_recap_checklist_grid`

## 验收
按 00 自检；确认 3~5 条回顾不溢出；checkmark 动画流畅；时序正确。
