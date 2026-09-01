# external_tool/configs — optional decoration hints

`studio_config.yaml` points two optional paths here:

| Config key | Expected file |
|---|---|
| `paths.gold_template_policies_file` | `gold_template_policies.yaml` |
| `paths.instructions_file` | `instructions.yaml` |

Neither file ships with the repository, and **nothing breaks without them**.
They are prompt fragments appended when the pipeline decorates a spotlight
shot: the first describes how to choose among Gold Template structures, the
second carries house style rules for the generated slot text.

`pipeline/tools/decorate_shot.py` guards both reads (`_load_policy_hint` and
`_load_instructions_hint` check `Path.exists()` inside a `try`), so a missing
file yields an empty hint and decoration proceeds on the model's own judgement
plus the structure registry. The observable effect is that template choice and
slot wording get less steering, not that a shot fails to render.

## Supplying your own

Drop either file in this directory using the name above; the config already
resolves to `${REPO_ROOT}/external_tool/configs/`, so no edit is needed. Both
are read as plain text, not parsed as structured YAML — the pipeline truncates
them (2000 and 1500 characters respectively) before splicing them into the
prompt, so keep them short and declarative. To store them elsewhere, override
the two `paths.*` keys.
