# 02 · quote_card × mask_reveal × warm_editorial

> 先贴 `00_shared_preamble.md`。新增：**结构 quote_card + 运动 mask_reveal + 视觉 warm_editorial**。
> mother_id = `02_quote_warm`，duration 默认 5.5s。读 `references/css-patterns.md`（擦出/遮罩）。

## 结构层 generated/structures/quote_card/
金句/引述镜头，杂志编辑感。左侧大引号装饰，居中偏左排一段金句，下方署名与来源。
- DOM：
  - 装饰大引号「“」（纯装饰，`data-layout-ignore`，绝对定位允许）
  - `title`：金句正文，font-display 衬线，56~76px，`max-width` 自然换行（anim-role=`title`, slot=`quote`）
  - `accent-line`：金句下细线（anim-role=`accent-line`）
  - `eyebrow`：署名（anim-role=`eyebrow`, slot=`speaker`）
  - `badge`：来源角标（anim-role=`badge`, slot=`source`）
- slot_schema：`quote`≤42、`speaker`≤16、`source`≤20
- default：quote=「任何足够先进的技术，都与魔法无法区分」、speaker=「亚瑟 · 克拉克」、source=「《未来的轮廓》1962」
- structure_card：use_for=[名人名言,经典引述,思想名句]，spotlight_level=mid，duration_range=[4.0,6.5]

## 运动层 generated/motions/mask_reveal/
用 `clip-path: inset()` 做擦出（确定性、可 seek）：
- `title`：clip-path 从 `inset(0 100% 0 0)` → `inset(0 0 0 0)`，power4.out，0.7s，0.2s 起
- `accent-line`：scaleX 0→1，expo.out，0.6s
- `subtitle`/`badge`：y+20 渐显，power2.out，stagger 0.12s
- 不退场；空角色跳过。

## 视觉层 generated/visuals/warm_editorial/
warm_editorial palette 写满令牌（衬线 display）。`.bg-stage`：米白纸面 + 极淡噪点 + 边缘发丝线，**无强光晕、无阴影**（warm_editorial shadow-glow=none）。

## 验收
按 00 自检；确认衬线中文渲染正常、金句 42 字两行不溢出。各层可独立测试（单独 lint/render 结构、运动、视觉层目录）。
