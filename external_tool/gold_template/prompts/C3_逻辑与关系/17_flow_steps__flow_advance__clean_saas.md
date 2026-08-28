# 17 · flow_steps × clean_saas　【diagram · 动画内嵌结构】

> 先贴 `00_shared_preamble.md`。新增：**结构 flow_steps（family=diagram，含内嵌动画脚本）+ 视觉 clean_saas（复用）**。
> **不产出独立运动层**。编译时搭配通用叙事运动（推荐 `card_stack_sequence`）。
> mother_id = `17_flow_saas`，duration 默认 7.5s。读 `references/css-patterns.md` + `references/techniques.md`。

## 数据契约
- slot `nodes` JSON：`[{"id":"n1","label":"观察"},{"id":"n2","label":"假设"},{"id":"n3","label":"实验"},{"id":"n4","label":"验证"}]`
- slot `edges` JSON：`[{"from":"n1","to":"n2"},{"from":"n2","to":"n3"},{"from":"n3","to":"n4"}]`
- 结构层渲染横向流程占位；内嵌脚本读 nodes/edges 驱动出现。3~5 个节点最佳。

## 结构层 generated/structures/flow_steps/
> 结构层必须内嵌默认数据的静态占位图形。

流程/步骤镜头：`title` → 横向流程（node 卡片 `--surface`/`--radius-card`，edge 用 `--line-strong`，marker 箭头 `--accent`）→ `accent-line`。
- anim-role：`title` / `node`（每个）/ `node-label`（每个）/ `edge`（每条）/ `marker`（每个箭头）/ `accent-line`
- slot_schema：`headline`≤18、`nodes`(json)、`edges`(json)
- default：headline=「科学方法的四个步骤」
- structure_card：use_for=[流程步骤,方法论,科学方法]，spotlight_level=mid，duration_range=[6.0,9.0]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init`：
- 读 nodes/edges → 清空占位 → 创建 node/edge DOM
- `node`：依次弹入（back.out(1.2)，stagger 0.25）
- `edge`：scaleX 从起点向终点生长
- `marker`：edge 到达后弹现
- `node-label`：随 node 渐显

### sfx_cues（写在 structure_card.json）
`[{"at":0.3,"sfx":"pop"},{"at":0.7,"sfx":"whoosh"}]`

## 运动层
**不产出独立运动**。编译时使用通用叙事运动（如 `card_stack_sequence`）。

## 编译命令
```bash
python assembler/assemble.py --structure flow_steps --motion card_stack_sequence --visual clean_saas \
  --slots generated/templates/17_flow_saas/example_inputs.json --duration 7.5 \
  --out generated/templates/17_flow_saas/index.html
```

## 验收
按 00 自检；确认 3/4/5 节点都不溢出、edge 与 node 对齐、箭头方向正确。
