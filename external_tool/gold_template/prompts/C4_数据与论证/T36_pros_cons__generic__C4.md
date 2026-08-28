# T36 · pros_cons（类别4 数据与论证）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 pros_cons（family=argument）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `slide_in_hold`）。
> 视觉层复用现有（推荐 `clean_saas` 或 `warm_editorial`）。
> mother_id = `T36_pros_cons`，duration 默认 6.0s。

## 结构层 generated/structures/pros_cons/
利弊分析双栏。左 ✓ 列（优势/支持）+ 右 ✗ 列（劣势/反对），每侧 2~4 条 list-item。

- DOM（`data-anim-role` / `data-slot`）：
  - 小标签（anim-role=`eyebrow`, slot=`eyebrow`）
  - 主标题（anim-role=`title`, slot=`headline`）
  - 左栏（优势）：
    - 栏标题 "✓ 优势"（anim-role=`subtitle`, 无 slot）
    - 列表项 2~4 条（anim-role=`list-item`, slot=`pro1`/`pro2`/`pro3`/`pro4`），每条前带 ✓ 图标，背景 `rgba(34,197,94,0.1)`
  - 右栏（劣势）：
    - 栏标题 "✗ 劣势"（anim-role=`subtitle`, 无 slot）
    - 列表项 2~4 条（anim-role=`list-item`, slot=`con1`/`con2`/`con3`/`con4`），每条前带 ✗ 图标，背景 `rgba(239,68,68,0.1)`
  - `.stage-panel` 内 grid 两列布局
- slot_schema：`eyebrow`≤10、`headline`≤16、`pro1~pro4`≤20、`con1~con4`≤20
- default：eyebrow="利弊分析"、headline="核能发电"、pro1="零碳排放运行"、pro2="能量密度极高"、pro3="不受天气影响"、con1="核废料处理难"、con2="建设成本极高"、con3="事故风险虽小但后果严重"
- structure_card：use_for=[利弊分析, 优劣对比, 正反面]，spotlight_level=medium，duration_range=[5.0,7.0]

## 运动层
**不产出独立运动**。通用运动负责 list-item stagger 入场（左栏先，右栏后）。

## 音效
- sfx_cues：`[{"at":0.5,"sfx":"tick","comment":"优势逐条"},{"at":1.5,"sfx":"tick","comment":"劣势逐条"}]`

## 多样性变体矩阵（3 个模版）
- A｜`pros_cons` · 左右双栏：经典双栏（variant_label「左右双栏」）→ mother_id `T36_pros_cons`
- B｜`pros_cons_stack` · 上下堆叠：上方优势+下方劣势（variant_label「上下堆叠」）→ `T36_pros_cons_stack`
- C｜`pros_cons_cards` · 卡片交替：优劣交替排列为卡片（variant_label「卡片交替」）→ `T36_pros_cons_cards`

## 验收
按 00 自检；确认各栏 4 条文本不溢出；✓/✗ 颜色对比度；双栏间距合理。
