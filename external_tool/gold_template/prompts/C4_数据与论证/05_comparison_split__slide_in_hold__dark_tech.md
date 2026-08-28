# 05 · comparison_split × slide_in_hold × dark_tech（复用 dark_tech 视觉）

> 先贴 `00_shared_preamble.md`。新增：**结构 comparison_split + 运动 slide_in_hold（两者新建）**。
> **视觉 dark_tech 已由 #01 生成，直接复用 `generated/visuals/dark_tech/`，不要重做。**
> mother_id = `05_compare_dark`，duration 默认 6.5s。

## 结构层 generated/structures/comparison_split/
左右对比镜头（误区 vs 事实 / A vs B / 前 vs 后）。顶部标题，下面左右两张等宽对比卡，中间一条竖直 accent-line 分隔，可放「VS」装饰。
- DOM：
  - `title`：标题（anim-role=`title`, slot=`headline`）
  - 左卡 anim-role=`card`：标签 slot=`left_label`、正文 slot=`left_text`（左卡用 `--danger`/中性色系暗示「误区/旧」）
  - 右卡 anim-role=`card`：标签 slot=`right_label`、正文 slot=`right_text`（右卡用 `--accent`/`--success` 暗示「事实/新」）
  - 中间竖 `accent-line` + 「VS」装饰（装饰用 `data-layout-ignore`）
  - 颜色差异只能用令牌（左 danger / 右 success），**不写死 hex**
- slot_schema：`headline`≤18；`left_label`/`right_label`≤8；`left_text`/`right_text`≤30
- default：headline=「关于感冒的误区」、left_label=「误区」、left_text=「淋雨会导致感冒」、right_label=「事实」、right_text=「病毒感染才是真正原因」
- structure_card：use_for=[正误对比,A vs B,二元对比]，spotlight_level=high，duration_range=[5.0,8.0]

## 运动层 generated/motions/slide_in_hold/
最稳通用入场：
- `title`：y+30 渐显 power3.out 0.15s
- `card`：两张分别从左/右滑入——可用 index 奇偶决定 `x:-60`/`x:60`，`gsap.from` opacity0，expo.out，0.35s 起 stagger 0.12s
- `accent-line`：scaleY 0→1（origin 中）power2.out 0.5s
- 通用：对 `eyebrow/subtitle/stat/list-item` 若存在也给轻微 y 渐显（保证此 motion 能服务其他结构）
- 不退场；空角色跳过。

## 视觉层
复用 `generated/visuals/dark_tech/tokens.css`。

## 验收
按 00 自检；确认左右卡正文 30 字不溢出、左右色彩对立靠令牌实现、slide_in_hold 套到 #01 hook 结构上也能跑（重组兼容）。各层可独立测试（单独 lint/render 结构、运动、视觉层目录）。
