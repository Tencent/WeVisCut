# T32 · cause_effect_chain（类别3 逻辑与关系）　【bound】

> 先贴 `00_shared_preamble.md`。新增：**结构 cause_effect_chain（family=diagram）**。
> **Bound 结构**：内嵌 structure-script 实现多米诺式逐段点亮动画。
> 视觉层复用现有（推荐 `clean_saas` 或 `dark_tech`）。
> mother_id = `T32_cause_effect_chain`，duration 默认 7.0s。

## 结构层 generated/structures/cause_effect_chain/
因果链/多米诺图。A→B→C→D 的因果链条，每个节点用 emoji 图标 + 文字标签，连线带箭头，逐段点亮。

- DOM（`data-anim-role` / `data-slot`）：
  - 小标题：顶部（anim-role=`eyebrow`, slot=`eyebrow`）
  - 主标题：（anim-role=`title`, slot=`headline`）
  - 因果链容器：横向 flex 排列，每个节点包含：
    - emoji 图标：font-size 48~64px（anim-role=`decoration`）
    - 节点标签：font-size 24~32px（anim-role=`list-item`）
    - 节点之间的箭头连线（anim-role=`accent-line`，inline SVG 箭头）
  - 节点数据来自 slot `nodes`（JSON 数组），structure-script 动态渲染
- slot_schema：`eyebrow`≤10、`headline`≤16、`nodes`= JSON 数组，每项 {emoji, label}，3~5 项
- default：eyebrow="因果链"、headline="全球变暖的多米诺效应"、nodes=[{emoji:"🏭",label:"温室气体增加"},{emoji:"🌡️",label:"全球变暖"},{emoji:"🧊",label:"冰川融化"},{emoji:"🌊",label:"海平面上升"}]
- structure_card：use_for=[因果链, 多米诺效应, 连锁反应]，spotlight_level=medium，duration_range=[6.0,8.0]

## 结构绑定动画（structure-script）
1. 读取 `nodes` slot 数据（通过 `window.__hyperframes.getVariables()` 或 data-slot 解析）
2. 动态生成节点 DOM（emoji + label + 箭头连线）
3. 多米诺式逐段点亮：
   - 每个节点初始灰色/透明
   - 依次：箭头连线 scaleX 0→1（从左到右）→ 下一节点 scale 0.8→1 + opacity 0→1 + 颜色变亮
   - 每个节点间隔 0.8~1.2s
   - 节点点亮时配 `tick` 音效
4. 最后一个节点点亮后短暂 pulse/glow 强调
5. 完成后 removeAttribute("data-anim-role")

## 音效
- sfx_cues：`[{"at":0.5,"sfx":"tick","comment":"第一个节点"},{"at":1.5,"sfx":"tick","comment":"第二个节点"},{"at":2.5,"sfx":"tick","comment":"第三个节点"},{"at":3.5,"sfx":"sparkle","comment":"最终结果"}]`

## 多样性变体矩阵（3 个模版）
- A｜`cause_effect_chain` · 横向链条：节点从左到右排列（variant_label「横向链条」）→ mother_id `T32_cause_effect_chain`
- B｜`cause_effect_chain_vertical` · 纵向瀑布：节点从上到下瀑布式排列（variant_label「纵向瀑布」）→ `T32_cause_effect_chain_vertical`
- C｜`cause_effect_chain_zigzag` · Z 字形：节点 Z 字形排列，增加视觉丰富度（variant_label「Z字蜿蜒」）→ `T32_cause_effect_chain_zigzag`

## 验收
按 00 自检；确认 3~5 个节点不溢出；逐段点亮时序正确；数据驱动渲染正常。
