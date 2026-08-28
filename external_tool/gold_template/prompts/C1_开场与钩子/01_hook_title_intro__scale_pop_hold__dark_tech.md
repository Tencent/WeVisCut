# 01 · hook_title_intro × scale_pop_hold × dark_tech

> 先贴 `00_shared_preamble.md`。本 prompt 新增：**结构 hook_title_intro + 运动 scale_pop_hold + 视觉 dark_tech（三者都新建）**。
> mother_id = `01_hook_dark`，duration 默认 6.0s。

## 结构层 generated/structures/hook_title_intro/
开头钩子/全片标题镜头。垂直居中三段式：小标签 → 巨大主标题 → 副标题；底部一条 accent-line。
- DOM（`data-anim-role` / `data-slot`）：
  - `eyebrow`：上方胶囊小标签（anim-role=`eyebrow`, slot=`eyebrow`）
  - `title`：主标题，font-size 110~140px，允许主标题里一个关键词用 `.accent`（颜色 `var(--accent)`）（anim-role=`title`, slot=`headline`）
  - `subhead`：副标题一行（anim-role=`subtitle`, slot=`subhead`）
  - 一条 `accent-line`（anim-role=`accent-line`，无 slot），渐变细线
- slot_schema（max_chars）：`eyebrow`≤14、`headline`≤18、`subhead`≤20
- default：eyebrow=「5分钟科普」、headline=「为什么冰比水轻？」、subhead=「一个反常识的物理现象改变了整个地球」
- structure_card：use_for=[开头钩子,科普开场,知识引入]，spotlight_level=high，duration_range=[4.5,7.0]

## 运动层 generated/motions/scale_pop_hold/
- `eyebrow`：y+30、opacity0→1，power3.out，0.15s
- `title`：scale .82→1、opacity0→1，back.out(1.4)，0.30s（弹出感）
- `subtitle`：y+24 渐显，expo.out，0.62s
- `accent-line`：scaleX 0→1（transform-origin 左），power2.out，0.5s
- 之后保持，不退场。对不存在的角色跳过。

## 视觉层 generated/visuals/dark_tech/
用 `registry/visuals.json` 的 dark_tech palette 写满 `:root` 令牌全集。
`.bg-stage`：深色径向双光晕（青 `--accent` + 金 `--accent-2` 低透明）+ 80px 网格（`--grid-color`）+ 椭圆遮罩暗角。

## 验收
按 00 自检；额外确认主标题 18 字仍单屏不溢出、网格不抢标题。各层可独立测试（单独 lint/render 结构、运动、视觉层目录）。
