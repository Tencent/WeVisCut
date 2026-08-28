# T35 · ranking_list（类别4 数据与论证）　【bound】

> 先贴 `00_shared_preamble.md`。新增：**结构 ranking_list（family=argument）**。
> **Bound 结构**：内嵌 structure-script 实现倒计时逐条 reveal。
> 视觉层复用现有（推荐 `dark_tech` 或 `clean_saas`）。
> mother_id = `T35_ranking_list`，duration 默认 8.0s。

## 结构层 generated/structures/ranking_list/
排行榜/Top N。倒计时式排行（5→4→3→2→1），逐条 reveal，每条带序号+emoji+标签+数值条。

- DOM（`data-anim-role` / `data-slot`）：
  - 小标签（anim-role=`eyebrow`, slot=`eyebrow`）
  - 主标题（anim-role=`title`, slot=`headline`）
  - 排行容器：纵向列表，每条 rank-item 包含：
    - 序号徽章（1~5）
    - emoji 图标
    - 名称标签
    - 数值条（宽度按百分比）
  - 数据来自 slot `items`（JSON 数组），structure-script 动态渲染
- slot_schema：`eyebrow`≤10、`headline`≤16、`items`= JSON 数组，每项 {emoji, label, value, unit}，3~5 项
- default：eyebrow="Top 5"、headline="全球最深的五个海沟"、items=[{emoji:"🌊",label:"马里亚纳海沟",value:10994,unit:"m"},{emoji:"🌊",label:"汤加海沟",value:10882,unit:"m"},{emoji:"🌊",label:"菲律宾海沟",value:10540,unit:"m"},{emoji:"🌊",label:"千岛海沟",value:10542,unit:"m"},{emoji:"🌊",label:"克马德克海沟",value:10047,unit:"m"}]
- structure_card：use_for=[排行榜, Top N, 排名展示]，spotlight_level=medium，duration_range=[7.0,10.0]

## 结构绑定动画（structure-script）
1. 标题先入场
2. 从最后一名到第一名逐条 reveal：
   - 每条：序号 scale 0→1 弹出 → emoji 淡入 → 标签滑入 → 数值条从宽度 0 grow 到目标宽度
   - 间隔 1.0~1.5s 
3. 第一名 reveal 时特殊强调（scale pulse + glow + 延长停留）
4. removeAttribute("data-anim-role")

## 音效
- sfx_cues：`[{"at":1.0,"sfx":"tick","comment":"第5名"},{"at":2.2,"sfx":"tick","comment":"第4名"},{"at":3.4,"sfx":"tick","comment":"第3名"},{"at":4.6,"sfx":"tick","comment":"第2名"},{"at":5.8,"sfx":"sparkle","comment":"第1名揭晓"}]`

## 多样性变体矩阵（3 个模版）
- A｜`ranking_list` · 纵向列表：从上到下排列（variant_label「纵向列表」）→ mother_id `T35_ranking_list`
- B｜`ranking_list_bars` · 横向条形图：每条用水平条形图表示（variant_label「横向条形」）→ `T35_ranking_list_bars`
- C｜`ranking_list_podium` · 领奖台式：前三名突出，后几名缩小排列（variant_label「领奖台」）→ `T35_ranking_list_podium`

## 验收
按 00 自检；确认 3~5 项都正常渲染；倒计时时序正确；数值条比例正确。
