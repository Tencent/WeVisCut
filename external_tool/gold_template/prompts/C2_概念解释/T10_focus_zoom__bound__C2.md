# T10 · focus_zoom（类别2 概念与信息解释）　【diagram · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 focus_zoom（family=diagram，含内嵌动画脚本）**。
> **不产出独立运动**，「总览→放大某组件」的镜头动画内嵌在 `<script data-hf="structure-script">`。
> 编译时搭配通用叙事运动（推荐 `slide_in_hold`，仅负责 title）。
> 视觉层复用现有（推荐 `clean_saas` 或 `dark_tech`）。
> mother_id = `T10_focus_zoom`，duration 默认 6.5s。先读 `references/techniques.md`（图层/镜头）+ `references/motion-principles.md`。

## 结构层 generated/structures/focus_zoom/
从整体讲到细节：先展示总览图，再 zoom into 某个组件并旁注说明。
- DOM：
  - `title`：标题（anim-role=`title`, slot=`headline`）
  - `overview`：总览容器（anim-role=`overview`），内含若干小组件方块（CSS grid 排布的占位块）+ `overview_caption`（slot=`overview_caption`）
  - `zoom-frame`：聚焦框（anim-role=`zoom-frame`），定位在某个组件上，放大时该组件移入框内
  - `callout`：旁侧标注卡（anim-role=`callout`），含 `zoom_label`（slot）+ `zoom_desc`（slot）
  - `accent-line`（anim-role=`accent-line`）
  - 用包裹 div 做 scale/transform，不直接缩放文字尺寸
- slot_schema：`headline`≤16、`overview_caption`≤16、`zoom_label`≤10、`zoom_desc`≤28
- default：headline=「CPU 内部结构」、overview_caption=「处理器芯片全貌」、zoom_label=「算术逻辑单元」、zoom_desc=「负责所有数学运算和逻辑判断的核心部件」
- structure_card：use_for=[局部放大,细节聚焦,由总到分]，spotlight_level=mid，duration_range=[5.5,8.5]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init = function(tl, ctx) { ... }`：
- 阶段1：`overview` 整体 scale 1→1 渐显，组件方块 stagger 浮现，`overview_caption` 渐显
- 阶段2（约 50% 处）：对 `overview` 包裹层做 `scale 1→1.8` + `transform-origin` 指向目标组件（模拟 zoom-in），`zoom-frame` 描边浮现
- 阶段3：`callout`（含 zoom_label/zoom_desc）从侧边滑入定格
- 完成后对 overview/zoom-frame/callout 等 `removeAttribute("data-anim-role")`，**不处理 title**

### sfx_cues（可选）
`[{"at":1.6,"sfx":"whoosh","comment":"zoom-in 推进"}]`

## 运动层
**不产出独立运动**。编译时用通用运动负责 title。

## 视觉层
复用 `generated/visuals/clean_saas/`。

## 编译命令
```bash
python assembler/assemble.py --structure focus_zoom --motion slide_in_hold --visual clean_saas \
  --slots generated/templates/T10_focus_zoom/example_inputs.json --duration 6.5 \
  --out generated/templates/T10_focus_zoom/index.html
```

## 多样性变体矩阵（一个 prompt 产出 2 个模版，均 bound 各自内嵌脚本）
按 00「多样性变体矩阵」，产出 2 个空间布局变体（各独立 id，追加 `variant_of:"focus_zoom"` + `variant_label`）：
- A｜`focus_zoom` · 居中型：总览居中 → 放大目标组件（「居中放大」）→ `T10_focus_zoom`
- B｜`focus_zoom_corner` · 画中画型：角落保留总览缩略图 + 主区展示放大细节（「角落总览」）→ `T10_focus_zoom_corner`

## 验收
按 00 自检；确认 zoom 推进不出界、callout 不遮挡聚焦组件、render 下镜头确定性；viewer 自动跳过通用运动。
