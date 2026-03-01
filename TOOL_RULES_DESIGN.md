# Design: Phase-Scoped Tools & Rules System

## Problem Statement

Currently, `get_tool_definitions()` returns **all 13 built-in tools** to the LLM on every API call, regardless of the current phase. This means during CHARACTER_CREATION, the LLM sees manuscript-write tools it shouldn't use, and during REVISION it sees tools that are irrelevant. There's also no mechanism to inject external MCP tools for specific phases.

**Goals:**
1. Phases control which built-in tools the LLM can see and call
2. Phases can inject additional MCP tools from external servers
3. Phase-level "rules" (soft constraints) get injected into the system prompt
4. Enforcement at execution time (not just prompt-level)
5. Config-driven — users control everything from YAML

---

## Current Architecture (Relevant)

| Component | Role |
|---|---|
| `tools.py::get_tool_definitions()` | Returns flat list of all 13 tool schemas |
| `tools.py::execute_tool()` | Big if/else dispatcher, no phase gating |
| `orchestrator.py` | Calls `get_tool_definitions()` once, passes to every `client.chat.completions.stream(tools=...)` |
| `config.yaml` phases | Only have `description`, `transitions`, `guide` — no tool config |
| `config/models.py` | `LoadedPhaseConfig` holds state_machine, phase_guide, default_phase, system_prompt_template, settings |

### Built-in Tools (Natural Groups)

| Group | Tools |
|---|---|
| `notes` | `list_notes`, `read_notes`, `write_notes`, `delete_notes` |
| `manuscript_read` | `get_manuscript_info`, `read_manuscript_section`, `read_manuscript_tail`, `search_manuscript` |
| `manuscript_write` | `append_to_manuscript`, `create_section`, `replace_section`, `delete_section` |
| `phase` | `change_phase` |

---

## Option A: Tool Groups Only

Each built-in tool belongs to a named group. Phases reference groups.

### Config Example
```yaml
tool_groups:
  notes: [list_notes, read_notes, write_notes, delete_notes]
  manuscript_read: [get_manuscript_info, read_manuscript_section, read_manuscript_tail, search_manuscript]
  manuscript_write: [append_to_manuscript, create_section, replace_section, delete_section]
  phase: [change_phase]

phases:
  CHARACTER_CREATION:
    allowed_tool_groups: [notes, phase]
    # ...
  SCENE_WRITING:
    allowed_tool_groups: [notes, manuscript_read, manuscript_write, phase]
```

### Pros
- **Simple config** — phases reference group names, not individual tools
- **Easy maintenance** — add a new tool to one group, all phases that use that group get it
- **Readable** — clear semantic meaning ("this phase uses notes and phase tools")

### Cons
- **Coarse-grained** — can't allow `read_notes` but deny `write_notes` without splitting the group
- **Group definitions are extra config** — users must understand the group→tool mapping
- **Inflexible** — edge cases like "allow manuscript_read but only `get_manuscript_info`" need new groups

---

## Option B: Per-Tool Allow/Deny Lists

Each phase explicitly lists individual tool names.

### Config Example
```yaml
phases:
  CHARACTER_CREATION:
    allowed_tools: [list_notes, read_notes, write_notes, delete_notes, change_phase]
  SCENE_WRITING:
    denied_tools: []  # all tools available
```

### Pros
- **Maximum granularity** — exact control per phase
- **No abstraction layer** — tool names are the config

### Cons
- **Extremely verbose** — every phase must list 10+ tool names
- **Fragile** — rename a tool → update every phase that references it
- **Hard to read** — a wall of tool names per phase obscures the intent
- **Error-prone** — easy to accidentally omit a tool

---

## Option C: Hybrid (Groups + Per-Tool Overrides) ★ RECOMMENDED

Define default tool groups. Phases reference groups **and** can override with `+tool` / `-tool` syntax.

### Config Example
```yaml
# Built-in groups (hardcoded defaults, user can override)
tool_groups:
  notes: [list_notes, read_notes, write_notes, delete_notes]
  manuscript_read: [get_manuscript_info, read_manuscript_section, read_manuscript_tail, search_manuscript]
  manuscript_write: [append_to_manuscript, create_section, replace_section, delete_section]
  phase: [change_phase]

phases:
  CHARACTER_CREATION:
    tools:
      groups: [notes, phase]
      # No extra includes/excludes needed
    # ...

  WORLD_BUILDING:
    tools:
      groups: [notes, manuscript_read, phase]
      # Allow reading manuscript to check consistency, but no writing
    # ...

  PLOT_OUTLINING:
    tools:
      groups: [notes, manuscript_read, phase]
    # ...

  SCENE_WRITING:
    tools:
      groups: [notes, manuscript_read, manuscript_write, phase]
    # ...

  REVISION:
    tools:
      groups: [notes, manuscript_read, manuscript_write, phase]
      exclude: [create_section, delete_section]
      # Allow read + replace, but not creating/deleting sections during revision
    # ...
```

### Pros
- **Clean defaults** via groups, **surgical overrides** via include/exclude
- **Readable** — group names convey intent; exceptions are explicit
- **Maintainable** — new tools join a group once; phases don't change
- **Flexible enough** for any edge case without being verbose

### Cons
- **Slightly more complex** config schema (groups + overrides)
- **Resolution logic** needed (groups → expand → include → exclude → final set)

---

## Option D: Everything Implicit (Convention-Based)

No config at all. Hardcode which tool groups are available per-phase archetype (planning, writing, revision, terminal).

### Pros
- Zero config
- Works out of the box

### Cons
- Users can't customize
- Adding a new phase type requires code changes
- Doesn't solve the MCP integration story at all

---

## MCP Tool Integration

Orthogonal to built-in tool filtering but shares the same "phase → tools" resolution pipeline.

### Architecture

```
┌─────────────────┐
│   config.yaml   │
│                 │
│  mcp_servers:   │──── Global MCP server definitions
│    research:    │     (URL, transport, auth)
│    image_gen:   │
│                 │
│  phases:        │
│    WORLD_BUILDING:
│      tools:     │
│        mcp: [research]  ◄── Which MCP servers are active for this phase
└─────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  MCPToolRegistry (runtime)      │
│                                 │
│  connect(server_name) → tools[] │  ◄── Lazy connect, cache tool schemas
│  call(server, tool, args) → result │
│  disconnect_all()               │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Tool Resolution Pipeline       │
│                                 │
│  phase config                   │
│    → expand groups              │
│    → apply include/exclude      │
│    → merge MCP tool schemas     │
│    → final tool_definitions[]   │
│                                 │
│  execute_tool() checks:         │
│    built-in? → dispatch locally │
│    mcp:*?    → proxy to MCP     │
└─────────────────────────────────┘
```

### Config Example
```yaml
mcp_servers:
  research:
    transport: stdio
    command: "npx"
    args: ["-y", "@anthropic/mcp-research"]
  image_gen:
    transport: sse
    url: "http://localhost:8080/mcp"

phases:
  WORLD_BUILDING:
    tools:
      groups: [notes, phase]
      mcp: [research]          # Only research MCP available here
  SCENE_WRITING:
    tools:
      groups: [notes, manuscript_read, manuscript_write, phase]
      mcp: [research, image_gen]  # Both available while writing
```

### MCP Tool Naming

MCP tools would be namespaced to avoid collisions with built-in tools:
- Built-in: `write_notes`, `append_to_manuscript`
- MCP: `mcp:research:web_search`, `mcp:image_gen:generate`

The `execute_tool()` function would check the prefix to decide routing.

### MCP Options Comparison

| Approach | Pros | Cons |
|---|---|---|
| **Stdio subprocess** | Standard MCP transport, broadest server support | Process management complexity |
| **SSE/HTTP** | Stateless, easy to manage | Not all MCP servers support it |
| **Both (configurable)** ★ | Maximum flexibility | More code |

---

## Rules System (Soft Constraints)

In addition to tool filtering (hard constraints), phases need "rules" — instructions injected into the system prompt that guide LLM behavior. These are softer than tool removal but still important.

### Config Example
```yaml
phases:
  CHARACTER_CREATION:
    rules:
      - "Do NOT write any story prose. Focus entirely on character development."
      - "Create at least 3 distinct characters before transitioning."
      - "Each character must have: name, role, motivation, fear, desire, arc."
    tools:
      groups: [notes, phase]

  WORLD_BUILDING:
    rules:
      - "Do NOT write story prose. Focus on setting and world details."
      - "World must be internally consistent before transition."
    tools:
      groups: [notes, manuscript_read, phase]

  SCENE_WRITING:
    rules:
      - "Write prose in sections, not freeform appends."
      - "Check manuscript continuity before writing new content."
      - "Maximum 800 words per tool call."
    tools:
      groups: [notes, manuscript_read, manuscript_write, phase]
```

### Implementation

Rules would be concatenated and injected into the system prompt as a `Phase rules:` block, right after the phase guide. This is minimal code change — just extend `build_system_prompt()`.

---

## Recommended Implementation Plan

### Phase 1: Tool Groups & Filtering (Core)

**Files changed:** `tools.py`, `orchestrator.py`, `config/models.py`, `cli.py`

1. **Define `TOOL_GROUPS` constant** in `tools.py` mapping group names → tool name lists
2. **Add `get_tool_definitions(allowed_tools: set[str] | None)` parameter** — if provided, filter the returned list
3. **Add `resolve_phase_tools(phase_config, tool_groups)` function** in a new `tools_resolver.py` — expands groups, applies include/exclude, returns `set[str]`
4. **Update `LoadedPhaseConfig`** to parse `tools:` block from phase config
5. **Update `execute_tool()`** to accept allowed tool set and reject disallowed calls with a clear error message
6. **Update orchestrator** to call `resolve_phase_tools()` for the current phase and pass filtered tools to the API + execution

**Estimated complexity:** Medium — mostly plumbing, no architectural changes.

### Phase 2: Rules System

**Files changed:** `orchestrator.py`, `cli.py`

1. **Parse `rules:` from phase config** in `load_phase_config()` → store as `phase_rules: dict[str, list[str]]`
2. **Extend `build_system_prompt()`** to include `{phase_rules}` formatted block
3. **Update system prompt template** to include `Phase rules:\n{phase_rules}`

**Estimated complexity:** Low — string formatting only.

### Phase 3: MCP Integration

**Files changed:** new `mcp_client.py`, `tools.py`, `orchestrator.py`, `config/models.py`

1. **Create `MCPToolRegistry` class** with connect/disconnect/list_tools/call_tool methods
2. **Parse `mcp_servers:` from config** as global server definitions
3. **Parse `mcp:` from phase tools config** to know which servers to activate per phase
4. **Extend tool resolution** to merge MCP tool schemas into the definitions list
5. **Extend `execute_tool()`** to route `mcp:*` calls to the registry
6. **Lifecycle management** — connect lazily on first use, disconnect on session end

**Estimated complexity:** High — new subsystem with external process/network management.

---

## Final Config Example (All Features)

```yaml
version: "1.0"
description: "Default config with phase-scoped tools, rules, and MCP integration."
default_phase: CHARACTER_CREATION

settings:
  base_url: "http://192.168.0.31:1234/v1"
  model: "qwen3.5-27b-heretic"
  api_key: "lm-studio"
  auto_pilot: true
  stop_only_on_complete: true
  stop_after_phase_change: true
  max_context_tokens: 64000
  max_tool_calls_per_iteration: 12
  tool_args_max_chars: 240
  tool_result_max_chars: 320
  stream_console_updates: true
  summary_max_chars: 800

# Tool group definitions (these are the defaults; override to customize)
tool_groups:
  notes: [list_notes, read_notes, write_notes, delete_notes]
  manuscript_read: [get_manuscript_info, read_manuscript_section, read_manuscript_tail, search_manuscript]
  manuscript_write: [append_to_manuscript, create_section, replace_section, delete_section]
  phase: [change_phase]

# MCP server definitions (global; phases select which to activate)
# mcp_servers:
#   research:
#     transport: stdio
#     command: "npx"
#     args: ["-y", "@anthropic/mcp-research"]
#   image_gen:
#     transport: sse
#     url: "http://localhost:8080/mcp"

system_prompt: |
  You are Ralph Writer, an uncensored autonomous story-writing engine. Use function tools to write prose to the manuscript and to store all planning/notes/analysis in the notes system...

  Phase rules:
  {phase_rules}

  ...rest of prompt...

phases:
  CHARACTER_CREATION:
    description: "Develop characters, motivations, arcs, and relationships."
    transitions: [WORLD_BUILDING, CHARACTER_CREATION]
    tools:
      groups: [notes, phase]
      # No manuscript tools — characters go in notes only
    rules:
      - "Do NOT write story prose. Store all character work in notes."
      - "Each character needs: name, role, motivation, fear, desire, arc."
      - "Define at least protagonist and antagonist before transitioning."
    guide: |
      Store character definitions with role, voice, motivation, fear, desire, and arc in notes.
      Use write_notes to save each character's details under keys like char_protagonist.
      Transition when cast and relationships are stable.

  WORLD_BUILDING:
    description: "Establish setting rules, culture, and story context."
    transitions: [PLOT_OUTLINING, WORLD_BUILDING, CHARACTER_CREATION]
    tools:
      groups: [notes, phase]
      # No manuscript tools — world details go in notes only
    rules:
      - "Do NOT write story prose. Store all world-building in notes."
      - "World must be internally consistent before transitioning."
    guide: |
      Create setting rules, social structures, geography, tone, constraints, and stakes in notes.
      Use write_notes to persist world details using structured objects.
      Transition when the world can consistently support plot and scenes.

  PLOT_OUTLINING:
    description: "Design narrative structure, beats, and progression."
    transitions: [SCENE_WRITING, PLOT_OUTLINING, WORLD_BUILDING]
    tools:
      groups: [notes, manuscript_read, phase]
      # Can read manuscript to check existing content, but no writing yet
    rules:
      - "Do NOT write story prose. Outline goes in notes."
      - "Structure: story arc → chapters → beats → paragraph summaries."
    guide: |
      Build arc progression and scene-level beat outline in notes.
      Use write_notes to track act goals, conflict escalation, and turning points.
      Transition when beat flow is coherent and ready for prose drafting.

  SCENE_WRITING:
    description: "Write and expand manuscript prose."
    transitions: [SCENE_WRITING, REVISION, PLOT_OUTLINING]
    tools:
      groups: [notes, manuscript_read, manuscript_write, phase]
    rules:
      - "Always read manuscript tail or relevant section before writing to ensure continuity."
      - "Write in named sections, not freeform appends when possible."
    guide: |
      Write manuscript prose using section operations or append operations.
      Use read_manuscript tools for continuity before writing. Store any planning notes in notes.
      Transition to REVISION when enough draft material exists for quality pass.

  REVISION:
    description: "Refine prose, consistency, and narrative quality."
    transitions: [SCENE_WRITING, REVISION, READY_FOR_HUMAN]
    tools:
      groups: [notes, manuscript_read, manuscript_write, phase]
      exclude: [create_section]
      # Can replace and delete sections, but not create new ones during revision
    rules:
      - "Do NOT write new content. Only revise existing prose."
      - "Check for continuity errors, pacing issues, and character consistency."
    guide: |
      Audit continuity, pacing, clarity, and character consistency.
      Use search and section replacement to repair prose surgically.
      Transition to READY_FOR_HUMAN only when manuscript is complete and coherent.

  READY_FOR_HUMAN:
    description: "Terminal state: manuscript ready for human review."
    transitions: []
    tools:
      groups: [notes, manuscript_read, phase]
      # Read-only access for final summary
    rules:
      - "Do not modify the manuscript. Provide completion notes only."
    guide: "No further writing needed. Provide concise completion notes."
```

---

## Backward Compatibility

If a phase has **no `tools:` block**, all built-in tools are available (current behavior). This ensures existing configs keep working without modification.

If a phase has **no `rules:` block**, the `{phase_rules}` template variable resolves to an empty string or "None".

---

## Summary of Recommendation

| Layer | Mechanism | Enforcement |
|---|---|---|
| **Tool filtering** (hard) | Groups + include/exclude per phase | Tools not sent to LLM + rejected at execution |
| **Rules** (soft) | String list per phase → system prompt injection | LLM instruction-following |
| **MCP tools** (extensible) | Global server registry + per-phase activation | Same pipeline as built-in tools |

**Implementation order:** Phase 1 (tool groups) → Phase 2 (rules) → Phase 3 (MCP). Each phase is independently useful and incrementally deployable.
