# RUNBOOK — 怎么一个个把 12 个母模板跑出来

> 环境：Cursor IDE + Claude/Gemini，skill 在 `Skills/hyperframes/`，渲染工程根 = 本 `Gold_Template/`（已放 package.json / hyperframes.json / meta.json / fonts）。

## 0. 一次性准备（已做好）
- 本目录已是一个 hyperframes 工程：`fonts/` 含 Noto Sans SC（400/500/700/900）+ Noto Serif SC Bold；CLI 钉在 `hyperframes@0.6.64`。
- **字体自动接线**：`assemble.py` 会扫描 `fonts/` 自动生成 `@font-face`（family/weight/format 自动推断）注入每个 index.html。
  CJK 字体不在 HyperFrames 内置列表（内置只有拉丁字体），所以必须靠 `fonts/` 提供 + 本机制注入。
- **衬线字体说明**：`warm_editorial`/`minimal_academic` 只在大标题用 Noto Serif SC（weight 700），Bold.otf 已够。
  若想要更轻的衬线字重（400/500），再把 `NotoSerifSC-Regular.otf` 丢进 `fonts/` 即可（会被自动接线）。

## 1. 运行顺序（必须按序，后面的复用前面的层）
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10   （造齐 24 个子模块）
11 → 12                                            （纯重组验证，不造新模块）
```
05/07 复用视觉、09/10 复用运动+视觉，所以别跳着跑。

## 2. 每一步怎么做（循环）
对第 N 步（N = 01..12）：

1. 在 Cursor 聊天里**新开一轮**，把下面【运行提示词】里的 `NN` 换成当前编号，整段发给 Agent。
2. Agent 会读 skill + 契约 + 该步 prompt，生成三层文件（写进 `structures/ motions/ visuals/`），
   再用 `assemble.py` 拼出 `index.html`，然后自检。
3. 你只需看它最后的 checklist：lint/validate/inspect/render 是否全绿、三层是否零耦合。
4. 绿了就进入下一步；红了让它按 `fix_hint` 改完再继续。

## 3. 【运行提示词】（把 NN 改成 01~12，逐步复制这一整段发给 Agent）

```
请严格按下面流程生成 Gold Template 的第 NN 个母模板，全程用中文回复，代码标识符用英文。

第一步 读取（务必先读，不要凭记忆）：
1. Skills/hyperframes/SKILL.md（HyperFrames 权威用法）+ 其中 prompt 里点名的 references/*.md
2. Code/WeDirector/Gold_Template/CONTRACT.md（三层解耦铁律，逐条遵守）
3. Code/WeDirector/Gold_Template/prompts/00_shared_preamble.md（共用前置）
4. Code/WeDirector/Gold_Template/prompts/NN_*.md（本步的具体规格）
5. Code/WeDirector/Gold_Template/registry/visuals.json（取本步 visual 的真实配色令牌）

第二步 生成（严格按 CONTRACT §7 落盘；已存在的层【复用，别重造】）：
- 新结构 → structures/<structure_id>/{structure.html, slot_schema.json, structure_card.json}
- 新运动 → motions/<motion_id>/{motion.js, motion_card.json}
- 新视觉 → visuals/<visual_id>/{tokens.css, visual_card.json}
- 示例   → templates/<mother_id>/example_inputs.json（真实感中文，覆盖短/中/偏长三组）
红线：结构层零硬编码颜色字体(全 var(--token))且无 script；运动层只用 [data-anim-role] 选择器、入场+保持、确定性；
视觉层只定义 CONTRACT §1 令牌全集(含 --stage-bg=0.5透明 / --radius-stage / --stage-blur)+ .stage/.bg-stage/.caption-bar，
不要写 .stage-panel（那是 assembler 的）。内容只渲染在居中半透明圆角面板内。

第三步 编译 + 验收（在 Gold_Template/ 目录下执行）：
1) python assembler/assemble.py --structure <s> --motion <m> --visual <v> \
     --slots templates/<mother_id>/example_inputs.json --duration <d> --out index.html
2) npm run check        # = hyperframes lint && validate && inspect
3) npm run render       # 出 mp4
4) 把验收通过的 index.html 复制一份到 templates/<mother_id>/index.html 存档
5) 至少换 3 组 slots 重跑 assemble.py，确认中文不溢出（必要时用 window.__hyperframes.fitTextFontSize）

最后给我一段 checklist：lint/validate/inspect/render 结果、三层零耦合自查、是否触发修复及修了哪层。
如果是 NN=11 或 12（重组验证）：不要造新子模块，直接用 assemble.py 跨层重组已有三层，渲染验证；
若渲染异常，定位是哪一层违反 CONTRACT 并回那一层修，不要在本步打补丁。
```

## 4. 渲染产物去哪了
- 临时渲染目标固定是工程根 `index.html`（每步会被覆盖，属正常）。
- 每步验收通过后，成品归档在 `templates/<mother_id>/index.html`，mp4 在 hyperframes 默认输出目录（`renders/` 或 CLI 提示路径）。

## 5. 跑完后的总验收
- `structures/` 10 个、`motions/` 8 个、`visuals/` 6 个，数量对齐。
- 随机抽几个**没在 12 矩阵里**的三元组（如 `quote_card × number_countup × neon_cyber`）用 assemble.py 验证能拼能渲。
- 回 `../../Planning/260602_Stage3换HyperFrames方案.md`：拿 30~50 篇真实文章跑 Stage1/2 标 is_spotlight，统计真实 spotlight 结构分布，确认 10 结构是否覆盖；缺高频结构再补。
```
