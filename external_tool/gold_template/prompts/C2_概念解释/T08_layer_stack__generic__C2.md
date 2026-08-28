# T08 · layer_stack（类别2 概念与信息解释）　【diagram · 通用入场运动】

> 先贴 `00_shared_preamble.md`。新增：**结构 layer_stack（family=diagram）**。
> **不产出独立运动**：搭配通用入场运动（推荐 `card_stack_sequence`），分层靠 `card` stagger 自下而上入场。
> 视觉层复用现有（推荐 `dark_tech` 或 `clean_saas`）。
> mother_id = `T08_layer_stack`，duration 默认 6.0s。

## 结构层 generated/structures/layer_stack/
分层架构（上中下三层堆叠，如「应用层 / 模型层 / 数据层」）。上 `title` → 居中三层横条堆叠 → accent-line。
- DOM：
  - `title`：标题（anim-role=`title`, slot=`headline`）
  - 三层横条 `card`（anim-role=`card`），纵向堆叠（`flex-direction:column`），自上而下宽度可微缩（梯形堆叠感）：
    - 每层内：层名（slot=`layer1`/`layer2`/`layer3`）+ 可选小标注
    - 顶层用 `var(--accent)`，下层渐弱（用 `--ink-3`/边框区分），靠令牌不写死色
  - `accent-line`（anim-role=`accent-line`）
- slot_schema：`headline`≤16；`layer1/2/3`≤16
- default：headline=「互联网协议的四层结构」、layer1=「应用层」、layer2=「传输层」、layer3=「网络层」、layer4=「链路层」
- structure_card：use_for=[层次结构,分层模型,堆叠概念]，spotlight_level=mid，duration_range=[5.0,8.0]

## 运动层
**不产出独立运动**。通用运动负责 title + card stagger（建议靠运动 stagger 顺序呈现自下而上堆叠感）+ accent-line。

## 视觉层
复用 `generated/visuals/dark_tech/`。

## 编译命令
```bash
python assembler/assemble.py --structure layer_stack --motion card_stack_sequence --visual dark_tech \
  --slots generated/templates/T08_layer_stack/example_inputs.json --duration 6.0 \
  --out generated/templates/T08_layer_stack/index.html
```

## 音效与趣味 motif
- **sfx_cues（写进 structure_card.json）**：`[{"at":0.4,"sfx":"whoosh","comment":"层叠入"},{"at":1.1,"sfx":"tick","comment":"逐层落位"}]`
- **趣味 motif（可选）**：各层像积木自下而上「咔」地叠上/卡入；顶层到位轻高亮。bespoke 卡入 → 内嵌 `structure-script`（变 bound）。

## 多样性变体矩阵（一个 prompt 产出 3 个模版）
按 00「多样性变体矩阵」，产出 3 个空间布局变体（各独立 id，追加 `variant_of:"layer_stack"` + `variant_label`）：
- A｜`layer_stack` · 正面型：正面横条上下堆叠（「正面堆叠」）→ `T08_layer_stack`
- B｜`layer_stack_iso` · 等距型：等距/斜视分层（skew/rotate 透视感）（「等距侧视」）→ `T08_layer_stack_iso`
- C｜`layer_stack_exploded` · 分解型：各层拉开 + 引线标注（「分解展开」）→ `T08_layer_stack_exploded`

## 验收
按 00 自检；确认三层横条居中堆叠不溢出、层间留白均匀、颜色层级取自令牌。
