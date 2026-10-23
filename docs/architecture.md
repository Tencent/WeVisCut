# Architecture

How WeVisCut runs a trained long-horizon policy against a real video-editing
environment, and why it is built the way it is.

The short version: a 9B policy model emits one tool call per step; a runtime owns
the editing state and rebuilds the model's entire input from that state on every
step; middleware may reject a call but never replaces it. Everything else follows
from those three sentences.

---

## Contents

- [Layers](#layers)
- [The alignment problem](#the-alignment-problem)
- [Component design](#component-design)
  - [The chat model](#the-chat-model-aicut_agentchat_modelpy)
  - [The tool layer](#the-tool-layer-aicut_agenttoolspy)
  - [The runtime](#the-runtime-aicut_agentruntimepy)
  - [The guardrail middleware](#the-guardrail-middleware-aicut_agentmiddlewarepy)
  - [Agent assembly](#agent-assembly-aicut_agentagentpy)
- [The tool environment](#the-tool-environment-pipeline)
- [Service layer](#service-layer-studio)
- [What is enforced by tests](#what-is-enforced-by-tests)

---

## Layers

```
static/          browser UI — timeline, manual editing, chat editing
   │ REST + WebSocket
studio/          FastAPI app, run registry, project store, editor, chat editing
   │
aicut_agent/     deepagents / LangGraph integration
   │   chat_model.py   LangChain ⇄ training-format conversion
   │   tools.py        the 12 tools as StructuredTools
   │   runtime.py      per-run state authority + event bridge
   │   middleware.py   guardrails
   │   agent.py        create_deep_agent assembly
   │
server/          inference engine — model loading, state serialisation,
   │             tool schemas, output parsing, layer-split rendering
pipeline/        the actual environment: 12 tool implementations
   │   providers/      pluggable footage / BGM back ends
external_tool/   renderer, retrieval tool, template library, binaries
```

Two of these boundaries matter more than the rest.

`server/` is where the training contract lives. The chat template, the state JSON
builder, and the tool schemas here are the same code paths the training-data
generator used. Nothing above this layer is allowed to reformat what the model
sees.

`pipeline/` is a plain Python environment with no notion of an agent. Each tool
is a function that takes arguments and mutates the world (writes audio, downloads
footage, renders a shot). It can be driven by the agent, by the editor when a
user changes one shot, or by a test.

---

## The alignment problem

A policy trained on a fixed input format degrades when deployed against a
slightly different one. Two of this system's central design choices exist only to
close that gap.

### The runtime is the state authority

A conventional ReAct harness accumulates messages: system, user, assistant, tool,
assistant, tool, and so on. The context grows, and the model sees a shape it
never saw in training.

This policy was trained on **single-turn** inputs. Each training sample is
`[system, user]`, where the user message is one JSON document containing
`recent_history`, `last_observation`, `global_state`, and `agent_contract`. The
history is *serialised into the state*, not spread across turns.

So on every step `AICutChatModel` discards the harness's message list and rebuilds
that JSON from the runtime's authoritative state. The model's input at step 7 has
exactly the same shape as at step 1, and exactly the shape it trained on. The
training data itself declares this: `agent_contract.runtime_is_state_authority`
is `true`.

The consequence to keep in mind when extending the system: **the `ToolMessage`
content the harness stores is for bookkeeping and the UI only.** If you want the
model to see something, it has to reach the runtime state.

### Guardrails observe, they do not correct

`AICutGuardrailMiddleware` may do exactly three things:

1. reject a call whose preconditions are unmet, feeding the rejection back as an
   observation;
2. count repetitions and failures, and stop a run whose budget is exhausted;
3. end the graph when `Tool_StitchVideo` has produced a final video.

It never rewrites arguments and never substitutes a different tool. An illegal
call becomes an observation, and the model recovers on its own — which it was
trained to do.

Every server-side "helpful" override is therefore a config switch under
`agent.autonomy.*`, and every one of them defaults to off. A policy trained to
recover should be measured on whether it recovers, so the honest default is to
let it try. Turn them on to make a demo more robust, but know that you are then
measuring the harness, not the model.

---

## Component design

### The chat model (`aicut_agent/chat_model.py`)

`AICutChatModel` subclasses `BaseChatModel` and does one job: translate between
LangChain's interface and the training format.

**`bind_tools(tools)`** records the bound tool names and returns `self`. It
deliberately does *not* let LangChain generate the JSON schema — the schema sent
to the model comes from `model_loader._make_agent_tools()`, built from
`tool_schema.LIGHT_TOOL_SCHEMAS` plus `REQUIRED_TOOL_ARGS`, with
`additionalProperties: false`. That is the training generator's code path.

**`_generate(messages, ...)`** in agent mode:

1. takes the system prompt from the runtime — the original training prompt, not
   anything deepagents composed, so no profile or middleware can contaminate it;
2. rebuilds the user payload with
   `orchestrator.build_agentic_state_json(input_script, state)`;
3. calls `loader.predict_orchestrator(system, user_json, stage="agent_loop", allowed_tools=...)`
   with `stop_words=["</tool_call>"]`;
4. on a parse failure or an unknown tool name, writes the error into
   `last_observation` and re-predicts **inside the model**, up to
   `max_consecutive_parse_failures` times, so LangGraph only ever receives a
   well-formed tool call;
5. converts the result to `AIMessage(content=<think> summary, tool_calls=[...])`
   and attaches latency, token counts and raw output in `response_metadata` for
   the UI's terminal panel.

In **plain-text mode** — no runtime, no bound tools — it passes `(system, user)`
straight through to `loader.predict` and returns text. Conversational editing
uses this path.

### The tool layer (`aicut_agent/tools.py`)

The 12 tools become LangChain `StructuredTool`s:

- `name` is the training tool name;
- `description` is `TOOL_CARDS[name]` plus that tool's entry in
  `TOOL_PRECONDITIONS`;
- `args_schema` is built dynamically from `LIGHT_TOOL_SCHEMAS[name]` and
  `REQUIRED_TOOL_ARGS[name]`.

The body awaits `runtime.dispatch(name, args)`, which routes to
`orchestrator._dispatch_tool`, then applies the state update, appends to
`agent_history`, sets `last_observation`, and broadcasts progress events. The
observation JSON is returned as `ToolMessage.content`.

`Tool_StitchVideo` is the only terminal tool. When it sets
`state["final_video_path"]`, it returns
`Command(goto=END, update={"messages": [ToolMessage(...)]})`.

### The runtime (`aicut_agent/runtime.py`)

One `AICutRuntime` per run. It holds:

- the `AutonomousOrchestrator` instance, reused wholesale for state management,
  `_dispatch_tool`, `build_agentic_state_json`, `_record_agent_history`,
  `_make_agent_observation`, preconditions, and preview generation;
- `state`, `agent_history`, `last_observation`;
- `run_id`, the broadcast callback that translates internal events into the
  WebSocket vocabulary, and the progress callback;
- guardrail counters: parse failures, identical calls, identical failures,
  stage-4 repair budget, total tool calls;
- pause and cancel flags.

The chat model and the tools receive the same runtime by constructor injection.
Each run compiles its own agent graph; graph compilation is negligible next to
model inference, and constructor injection avoids depending on context variables
for correctness.

### The guardrail middleware (`aicut_agent/middleware.py`)

`AICutGuardrailMiddleware(AgentMiddleware)` implements `wrap_tool_call` /
`awrap_tool_call`:

- **Preconditions.** `orchestrator._agent_precondition_missing(name, state)` —
  if unmet, `handler` is never called; the middleware records history and returns
  a blocked observation.
- **Repetition.** Identical arguments (`max_identical_tool_calls`) and
  identical failure causes (`max_identical_tool_failures`) are counted.
- **Stage-4 budget.** Repair and judge attempts (`max_stage4_repairs`,
  `max_stage4_attempts`, judge stall detection) are capped, because
  review-and-repair is where a long-horizon agent can spin.
- On reaching a limit: either refuse the call with a `budget_exhausted`
  observation, or return `Command(goto=END)`.

`before_model` checks the pause and cancel flags each step. The step ceiling maps
onto LangGraph's `recursion_limit`.

### Agent assembly (`aicut_agent/agent.py`)

`build_aicut_agent(runtime, ...)` calls `create_deep_agent` with the chat model,
the 12 tools, the training system prompt, and the guardrail middleware — and then
switches off everything deepagents would otherwise add:

- `HarnessProfile(excluded_tools=[ls, read_file, write_file, edit_file, glob, grep, execute, task])`
  removes the built-in tools;
- `GeneralPurposeSubagentProfile(enabled=False)` removes the general-purpose
  subagent;
- `SummarizationMiddleware` and anything else that would call the model or edit
  the prompt is excluded.

The model ends up seeing the 12 tools it was trained on and nothing else. This is
asserted by `tests/test_aicut_agent.py`, because a harness upgrade silently
adding a tool would be an easy regression to miss.

This is also why `deepagents` is pinned to an exact upstream tag in
`requirements.txt` rather than a range. The action space the policy sees is a
function of the harness version, so the harness is part of the training contract,
not an interchangeable dependency.

---

## The tool environment (`pipeline/`)

The 12 tools split into three roles, which is also how the training system prompt
routes them:

| Role | Tools | Needs the LLM |
|---|---|---|
| Generator | `Tool_SegmentArticle`, `Tool_PlanShots`, `Tool_DecorateShot` | yes |
| Critic | `Tool_TextQuickJudge`, `Tool_VLMRenderJudge`, `Tool_SelfReflect` | yes |
| External | `Tool_GenerateAudio`, `Tool_RetrieveVisual`, `Tool_TimelineCompose`, `Tool_MusicSearch`, `Tool_ModalProbe`, `Tool_StitchVideo` | no |

Generator and critic tools call an LLM through `pipeline/llm.py`, which points at
the studio's own `/v1/chat/completions` endpoint by default — the same policy
weights serve both the agent loop and the tools' internal reasoning, in one
process, with no second GPU.

External tools talk to real services: TTS, the retrieval back end, the
HyperFrames renderer, ffmpeg. They are the ones that fail in deployment, which is
why `Tool_ModalProbe` exists and why every one of them degrades rather than
raising.

Ordering is not free-form. `Tool_GenerateAudio` establishes each shot's true
duration, so footage retrieval and template filling downstream depend on it;
`TOOL_PRECONDITIONS` encodes those dependencies and the middleware enforces them.

### Retrieval as an extension point

`pipeline/providers/` is a small registry, separate from the tools that use it.
`Tool_RetrieveVisual` and `Tool_MusicSearch` ask the registry for a named
provider and normalise whatever comes back. A provider that raises or returns
nothing counts as "no candidates", and the next back end in the cascade runs.

That isolation is deliberate: connecting the system to your own media library is
the one integration every deployment needs, and it should not require touching
tool code. See [`providers.md`](providers.md).

---

## Service layer (`studio/`)

| Module | Responsibility |
|---|---|
| `app.py` | FastAPI app, endpoints, WebSocket, static files, `/v1` endpoint |
| `config.py` | Load, interpolate and validate `studio_config.yaml`; project it into pipeline config; `--check` / `--summary` |
| `model_client.py` | Background weight loading, status reporting |
| `orchestrator.py` | State management, tool dispatch, state-JSON construction, previews |
| `runs.py` | Run registry, lifecycle, event history |
| `tracks.py` | Project state → the 8 UI tracks |
| `editor.py` | Manual edit operations and selective re-render |
| `project.py` | Project store, versioning, rollback |
| `chat_agent.py`, `chat_prompts.py` | Conversational editing |

Two things are worth knowing.

**Selective re-render.** An edit marks only the affected shots dirty.
`editor.py` re-runs `Tool_TimelineCompose` for those, then re-stitches. Changing
one line of narration in a 30-shot video re-renders one shot.

**Conversational editing does not use the agent loop.** `ChatEditAgent` makes a
single-turn call that returns a JSON envelope of proposed operations, validates
them against the project, and asks for confirmation on destructive ones. It has
its own prompts and its own memory compaction. It is a different problem from
long-horizon generation and it gets a different, simpler mechanism.

---

## What is enforced by tests

The interesting tests are the ones that guard the alignment properties, since
those are what break silently:

| Property | Test |
|---|---|
| The model sees exactly the 12 trained tools | `test_aicut_agent.py` |
| LangChain ⇄ training format round-trips | `test_aicut_agent.py` |
| Tool schemas match `tool_schema.py`, `verify_consistency()` passes | `test_aicut_agent.py` |
| Preconditions block; budgets refuse; a successful stitch ends the graph | `test_aicut_agent.py` |
| Plain-text conversation still works | `test_aicut_agent.py` |
| `<think>` / `<tool_call>` parsing, including malformed output | `test_thinking_parse.py` |
| Provider registration, normalisation, failure isolation | `test_providers.py` |
| Provider routing and cascade order | `test_retrieval_provider_switch.py` |
| Config, tracks, editor ops, chat parsing, guardrails | `test_studio_offline.py` |
| REST + WebSocket contract the frontend relies on | `test_frontend.js` |

All of them run offline with a mock loader and no GPU. Anything that needs real
weights is an end-to-end check, not a unit test:

```bash
bash run_studio.sh --check    # every resolved path and the agent import closure
bash run_studio.sh            # then generate one video and exercise the UI
```
