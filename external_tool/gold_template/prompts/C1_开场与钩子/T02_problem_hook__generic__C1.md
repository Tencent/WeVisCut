# T02 · problem_hook（类别1 开场与观点引入）　【narrative · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 problem_hook（family=narrative）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `scale_pop_hold` 或 `slide_in_hold`），痛点标签靠 `list-item` 的 stagger 入场。
> 视觉层复用现有（推荐 `dark_tech` 或 `neon_cyber`），已存在则直接复用。
> mother_id = `T02_problem_hook`，duration 默认 5.5s。

## 结构层 generated/structures/problem_hook/
开场抛出核心问题，下方并列 2~3 个痛点标签。垂直居中：小标签 → 巨大问句 → 痛点 pill 行 → accent-line。
- DOM（`data-anim-role` / `data-slot`）：
  - `eyebrow`：上方胶囊小标签（anim-role=`eyebrow`, slot=`eyebrow`）
  - 主问句：font-size 80~110px，可含一个 `.accent` 关键词（anim-role=`title`, slot=`question_text`）
  - 痛点 pill 行：2~3 个 pill（anim-role=`list-item`，slot=`tag1`/`tag2`/`tag3`），每个 pill 带圆点/边框装饰，横向 flex 居中、可换行
  - `accent-line`：底部渐变细线（anim-role=`accent-line`，无 slot）
- slot_schema（max_chars）：`eyebrow`≤14、`question_text`≤24、`tag1/2/3`≤10
- default 要写满（非空）：eyebrow=「思维陷阱」、question_text=「为什么你总觉得飞机比汽车危险？」、tag1=「可得性偏差」、tag2=「概率错觉」、tag3=「恐惧放大」
- structure_card：use_for=[思维误区,认知偏差,反直觉问题]，spotlight_level=high，duration_range=[4.5,7.0]

## 运动层
**不产出独立运动**。通用运动负责 eyebrow/title 入场、list-item stagger（痛点逐个蹦出）、accent-line 展开。

## 视觉层
复用现有 visual（如 `generated/visuals/dark_tech/`），不重造。

## 编译命令
```bash
python assembler/assemble.py --structure problem_hook --motion scale_pop_hold --visual dark_tech \
  --slots generated/templates/T02_problem_hook/example_inputs.json --duration 5.5 \
  --out generated/templates/T02_problem_hook/index.html
```

## 音效与趣味 motif
- **sfx_cues（写进 structure_card.json）**：`[{"at":0.6,"sfx":"tick","comment":"痛点逐个出现"},{"at":1.4,"sfx":"pop","comment":"末个痛点强调"}]`
- **趣味 motif（可选）**：问号/感叹号小图标在问句旁轻弹；痛点 pill 出现时轻微抖一下。若做 bespoke 描边/抖动动画 → 改为内嵌 `structure-script`（结构即变 bound，自动跳过通用运动）。

## 多样性变体矩阵（一个 prompt 产出 3 个模版）
按 00「多样性变体矩阵」，本概念产出 3 个空间布局变体（各独立 `structure_id` + `mother_id`，逐个先做存在性检测再生成，并在 `registry/structures.json` 追加 `variant_of:"problem_hook"` + `variant_label`）：
- A｜`problem_hook` · 聚焦型：居中问句 + 下方 pill 行（variant_label「居中聚焦」）→ mother_id `T02_problem_hook`
- B｜`problem_hook_scatter` · 散布型：痛点卡片用 `position:absolute` 散布画面四角错落、问句居中（variant_label「散布痛点」）→ `T02_problem_hook_scatter`
- C｜`problem_hook_split` · 分栏型：左大问句 + 右痛点纵列（variant_label「左右分栏」）→ `T02_problem_hook_split`

## 验收
按 00 自检；额外确认：问句 24 字仍单屏不溢出、3 个 pill 不挤压换行错乱；搭配非原配运动仍完整可见。
