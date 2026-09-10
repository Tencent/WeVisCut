"""Prompt helpers for future 0907 edit-planning model calls."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """你是 newvis0907 的编辑规划器，只能做“定位后规划”，不能声明已经成功执行。

硬性约束：
1. 只能从输入里的 `allowed_target_ids` 选择目标，绝不能发明 shot_id、subclip_id 或章节 ID。
2. 只能从输入里的 `supported_ops` 选择操作，绝不能输出未知 op。
3. 只输出一个 JSON 对象，不要 markdown，不要解释性前后文。
4. `reply` 只能描述“计划如何修改”或“为何需要澄清”，不能声称“已经成功”“已完成渲染”“修改已生效”。
5. 如果目标不唯一或没有命中，输出空 `ops` 并在 `reply` 里要求澄清。

输出格式：
{
  "reply": "中文说明",
  "ops": [
    {"op": "...", "...": "..."}
  ]
}
"""


def build_planner_payload(
    *,
    message: str,
    locator_result: dict[str, Any],
    allowed_target_ids: list[str],
    supported_ops: list[str],
) -> str:
    """Build the JSON payload for a future planning-model call."""
    payload = {
        "message": str(message or ""),
        "locator_result": locator_result,
        "allowed_target_ids": [str(value) for value in allowed_target_ids],
        "supported_ops": [str(value) for value in supported_ops],
    }
    return json.dumps(payload, ensure_ascii=False)
