# Gold Template — 知识科普视频「正交三层」母模板库

用 Gemini 3.1 Pro / Claude 批量生成 **12 个母模板**，生成时即按**结构 × 运动 × 视觉**三层正交拆开，
由一套编译器 `assembler/assemble.py` 任意重组成可 `hyperframes render` 的 1920×1080 spotlight 镜头。

> 配套总方案见 `../../Planning/260602_Stage3换HyperFrames方案.md`。本库就是其中的 **Stage 0：Gold Template Library**。

---

## 1. 为什么是「正交三层」而不是「整模板」

你给的三层就是天然正交的：

```
Layer 1 结构 Structure  → 决定画面结构（排版 / 槽位）        ←→ HyperFrames 的 DOM + data-slot
Layer 2 运动 Motion     → 决定动画方式（GSAP 时间线）         ←→ 只认 data-anim-role 的动画模块
Layer 3 视觉 Visual     → 决定字体/颜色/阴影/背景/圆角/装饰    ←→ :root 设计令牌 (tokens.css)
```

只要三层严格零耦合（铁律见 `CONTRACT.md`），就能 `10 结构 × 8 运动 × 6 视觉 = 480` 种组合自由拼。
这样 Qwen 后面学的不是写 HTML，而是选 `(structure_id, motion_id, visual_id) + 填槽`。

---

## 2. 关于「要不要先统计再决定模板」——回答你的疑问

之前建议「先统计 `selected_template_family / good_rate / repair_rate` 再定 8 个模板」，
**前提是资源稀缺、要排优先级**。你现在有 ¥10 万 Gemini 额度、要一次铺满 12 类，那：

- **直接铺 12 类完全合理**：这 12 个结构确实是知识科普高频结构，不必等统计。
- **统计降级为「事后覆盖率体检」**：12 类做完后，拿 30~50 篇真实文章跑 Stage1/2 → 标 `is_spotlight` →
  看真实 spotlight 落在哪些结构。只回答一个问题：**有没有高频结构没被这 12 类覆盖？** 有就补，没有就说明铺满了。
- 所以统计从「准入门槛」变成「验收清单」，不挡你现在批量生产。

---

## 3. 12 个母模板矩阵（覆盖 10 结构 / 8 运动 / 6 视觉）

| # | structure | motion | visual | 本 prompt 新增的子模块 |
|---|---|---|---|---|
| 01 | hook_title_intro | scale_pop_hold | dark_tech | 结构+运动+视觉 |
| 02 | quote_card | mask_reveal | warm_editorial | 结构+运动+视觉 |
| 03 | stat_highlight | number_countup | clean_saas | 结构+运动+视觉 |
| 04 | timeline_step | card_stack_sequence | minimal_academic | 结构+运动+视觉 |
| 05 | comparison_split | slide_in_hold | dark_tech | 结构+运动（复用视觉） |
| 06 | problem_solution | flash_transition | neon_cyber | 结构+运动+视觉 |
| 07 | product_feature | parallax_pan | clean_saas | 结构+运动（复用视觉） |
| 08 | social_overlay | typewriter_caption | documentary_caption | 结构+运动+视觉 |
| 09 | map_route | parallax_pan | documentary_caption | 结构（复用运动+视觉） |
| 10 | ending_cta | scale_pop_hold | warm_editorial | 结构（复用运动+视觉） |
| 11 | stat_highlight | slide_in_hold | minimal_academic | **0 新增（纯重组验证）** |
| 12 | hook_title_intro | mask_reveal | neon_cyber | **0 新增（纯重组验证）** |

- #01–#10 把 **10 结构 + 8 运动 + 6 视觉 = 24 个子模块**全部产齐。
- #11、#12 不再新造任何子模块，而是用 `assemble.py` 把已有三层**跨组合重组**，验证「100% 解耦」是否成立。
  若重组渲染出问题 → 说明某层违反了 `CONTRACT.md`，回去修那一层（这正是解耦的压力测试）。

---

## 4. 目录结构

```text
Gold_Template/
├── README.md                 你正在看
├── CONTRACT.md               ★三层解耦铁律（所有 prompt 必须遵守）
├── registry/
│   ├── structures.json       10 结构登记（id / 适用语义 / 用到的 anim-role / slots）
│   ├── motions.json          8 运动登记（id / 能量 / 适配 anim-role）
│   └── visuals.json          6 视觉登记（id / 真实配色 / 字体 / 风格关键词）
├── assembler/
│   └── assemble.py           ★一套代码：三层 + slots → 合法 HyperFrames index.html
├── prompts/
│   ├── 00_shared_preamble.md  所有母模板共用前置（调 skills + 守 CONTRACT）
│   └── 01..12_*.md            12 个母模板精细化生成 prompt
├── structures/<id>/          生成产物：structure.html + slot_schema.json + structure_card.json
├── motions/<id>/             生成产物：motion.js + motion_card.json
├── visuals/<id>/             生成产物：tokens.css + visual_card.json
└── templates/<mother_id>/    生成产物：index.html(成品) + example_inputs.json
```

---

## 5. 使用流程

```bash
# 1) 准备一个 hyperframes 工程目录（可复制你跑通的 wechat 案例的 package.json / meta.json）
#    把字体放到 fonts/，把 assemble.py 产出的 index.html 放工程根

# 2) 逐个把 prompts/01..10 丢给 Gemini/Claude（IDE 里带着 hyperframes skill）
#    要求它把三层文件写进 structures/ motions/ visuals/，并产 templates/<id>/index.html

# 3) 校验每个母模板
npx hyperframes lint && npx hyperframes validate && npx hyperframes inspect
npx hyperframes render

# 4) 跑 #11 #12 的重组验证
python assembler/assemble.py \
  --structure stat_highlight --motion slide_in_hold --visual minimal_academic \
  --slots templates/11_stat_academic/example_inputs.json \
  --duration 6 --out templates/11_stat_academic/index.html
#   → 再 lint/inspect/render；过了就证明三层正交可重组

# 可视化
cd /mnt/shang2cephfs/mm-base-vision-ascend/chenyumu/Code/WeDirector/Gold_Template && python3 -m http.server 7862
```

下游 Stage3 的固定编译器直接调 `assemble.py`（或其等价实现）：Qwen 只产
`(structure_id, motion_id, visual_id, slots, duration)`，渲染交给这套代码。

---

## 6. 一句话

**视觉只活在 `tokens.css`、运动只认 `data-anim-role`、结构只管 DOM+槽位；
三层在生成时就拆开，`assemble.py` 负责任意重组——这就是「一套代码、100% 解耦」。**
