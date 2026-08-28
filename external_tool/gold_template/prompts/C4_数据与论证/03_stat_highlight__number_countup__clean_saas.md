# 03 · stat_highlight × clean_saas　【含内嵌 countup 动画】

> 先贴 `00_shared_preamble.md`。新增：**结构 stat_highlight（含内嵌 countup 脚本）+ 视觉 clean_saas**。
> **不产出独立运动层**，countup 逻辑内嵌在结构的 `<script data-hf="structure-script">` 中。
> 编译时搭配通用入场运动（推荐 `scale_pop_hold`）。
> mother_id = `03_stat_saas`，duration 默认 5.5s。读 `data-in-motion.md`。

## 结构层 generated/structures/stat_highlight/
单个关键数字镜头。上小标签 → 居中超大数字+单位 → 下一句说明 → accent-line。
- DOM：
  - `eyebrow`：上方标签
  - 数字行：`stat`（超大数字 180~240px，`font-variant-numeric: tabular-nums`）+ `unit`（后缀如 % 倍）
  - `subtitle`：一句说明
  - `accent-line`
- slot_schema：`eyebrow`≤14、`stat_value`≤6、`unit`≤4、`caption_text`≤24
- default：eyebrow=「惊人数字」、stat_value=「137」、unit=「亿年」、caption_text=「这是宇宙从大爆炸到现在的年龄」
- structure_card：use_for=[惊人数字,单数据高亮,科普数据]，spotlight_level=high，duration_range=[4.0,6.5]

### 内嵌动画脚本（`<script data-hf="structure-script">`）
定义 `window.__hf_structure_init`：
- 读 `[data-anim-role="stat"]` 的 textContent 解析末值（支持小数/千分位）
- 建代理对象 `{v:0}`，用 `tl.to(obj,{v:final, onUpdate:()=>el.textContent=fmt(obj.v)})` 做 0→N 滚动
- 完成后 `stat` 元素 removeAttribute("data-anim-role") 避免通用运动重复处理

## 运动层
**不产出独立运动**。编译时使用通用入场运动（如 `scale_pop_hold`），负责 eyebrow/unit/subtitle/accent-line 入场。

## 视觉层 generated/visuals/clean_saas/
clean_saas palette 写满令牌。已存在则复用。

## 编译命令
```bash
python assembler/assemble.py --structure stat_highlight --motion scale_pop_hold --visual clean_saas \
  --slots generated/templates/03_stat_saas/example_inputs.json --duration 5.5 \
  --out generated/templates/03_stat_saas/index.html
```

## 验收
按 00 自检；确认 countup 在 render 下确定（同帧数值固定）、数字不溢出。搭配不同通用运动（如 slide_in_hold）验证兼容。
