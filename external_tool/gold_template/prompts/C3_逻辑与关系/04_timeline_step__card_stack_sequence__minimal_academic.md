# 04 · timeline_step × card_stack_sequence × minimal_academic

> 先贴 `00_shared_preamble.md`。新增：**结构 timeline_step + 运动 card_stack_sequence + 视觉 minimal_academic**。
> mother_id = `04_timeline_academic`，duration 默认 8.0s。

## 结构层 generated/structures/timeline_step/
三阶段时间轴/步骤流程。顶部标题，下面一条水平轴 accent-line，轴上挂 3 张步骤卡（横向等距）。
- DOM：
  - `title`：标题（anim-role=`title`, slot=`headline`）
  - 水平 `accent-line`（轴线，anim-role=`accent-line`）
  - 3 张卡，每张 anim-role=`card`，内含序号(01/02/03)、`list-item` 小标题、说明：
    - card1: slot=`step1_title`/`step1_desc`，card2: `step2_title`/`step2_desc`，card3: `step3_title`/`step3_desc`
  - 用 grid 3 列等宽，节点与轴线对齐
- slot_schema：`headline`≤18；每个 `stepN_title`≤8、`stepN_desc`≤22
- default：headline=「宇宙简史」、step1_title=「138亿年前」、step1_desc=「大爆炸」、step2_title=「46亿年前」、step2_desc=「太阳系形成」、step3_title=「38亿年前」、step3_desc=「生命出现」
- structure_card：use_for=[时间线,历史发展,演进过程]，spotlight_level=mid，duration_range=[6.0,9.0]

## 运动层 generated/motions/card_stack_sequence/
- `title`：y+30 渐显 power3.out 0.15s
- `accent-line`：scaleX 0→1（origin 左）expo.out 0.7s 0.3s 起
- `card`：依次接力出现——`gsap.from(cards,{y:40,opacity:0,scale:.96,duration:.5,ease:"back.out(1.3)",stagger:0.22})`，跟随轴线延展节奏
- `list-item`：卡内小标题轻微 y 渐显（可并入 card 或单独 stagger 0.1s）
- 不退场；空角色跳过。

## 视觉层 generated/visuals/minimal_academic/
minimal_academic palette 写满令牌（衬线 display、`--shadow-*: none`、圆角 4px）。`.bg-stage`：纯净米白 + 唯一一条发丝分隔线，**扁平无阴影**，卡片靠 `--line` 描边区分而非阴影。

## 验收
按 00 自检；确认 3 卡横向不溢出、stagger 节奏与轴线协调、学术扁平观感（无 glow）。各层可独立测试（单独 lint/render 结构、运动、视觉层目录）。
