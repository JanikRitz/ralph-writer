import json
import os
import sys
import time
import tiktoken
from datetime import datetime
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

# --- CONFIG ---
CONFIG = {
    "base_url": "http://192.168.0.31:1234/v1",
    "model": "qwen/qwen3.5-35b-a3b",
    "auto_pilot": True,
    "stop_only_on_complete": True,  # Only stop when phase is READY_FOR_HUMAN with success, else rerun
}

# --- STATE MACHINE ---
STATE_MACHINE = {
    "CHARACTER_CREATION": {
        "description": "Develop characters, their backgrounds, motivations, and arcs",
        "transitions": ["WORLD_BUILDING", "CHARACTER_CREATION"]
    },
    "WORLD_BUILDING": {
        "description": "Create the setting, rules, cultures, and world context",
        "transitions": ["PLOT_OUTLINING", "WORLD_BUILDING", "CHARACTER_CREATION"]
    },
    "PLOT_OUTLINING": {
        "description": "Structure the story beats, acts, and narrative arc",
        "transitions": ["SCENE_WRITING", "PLOT_OUTLINING", "WORLD_BUILDING"]
    },
    "SCENE_WRITING": {
        "description": "Write actual manuscript prose scene by scene",
        "transitions": ["SCENE_WRITING", "REVISION", "PLOT_OUTLINING"]
    },
    "REVISION": {
        "description": "Review and refine existing content, check consistency",
        "transitions": ["SCENE_WRITING", "REVISION", "READY_FOR_HUMAN"]
    },
    "READY_FOR_HUMAN": {
        "description": "Story complete and polished - ready for human review",
        "transitions": []  # Terminal state
    }
}

# Phase-specific guidance — only the current phase's text is included in the prompt
PHASE_GUIDE = {
    "CHARACTER_CREATION": (
        "Develop characters with names, roles, motivations, arcs, and relationships.\n"
        "Store each character under its own state key (e.g., write_state key='char_red').\n"
        "When characters are well-defined, transition to WORLD_BUILDING."
    ),
    "WORLD_BUILDING": (
        "Create the setting, rules, cultures, and world context.\n"
        "Store world data in organized keys (e.g., 'world_locations', 'world_rules').\n"
        "When the world is fleshed out, transition to PLOT_OUTLINING."
    ),
    "PLOT_OUTLINING": (
        "Structure the story: acts, beats, chapter outlines, pacing.\n"
        "Store plot structure in state (e.g., 'plot_beats', 'chapter_outline').\n"
        "When the outline is solid, transition to SCENE_WRITING."
    ),
    "SCENE_WRITING": (
        "Write prose for the manuscript. Use read_manuscript_tail() to get context for continuing.\n"
        "Use append_to_manuscript() for new content or create_section() for organized chapters.\n"
        "Write one scene/chapter per iteration. When all scenes are written, transition to REVISION."
    ),
    "REVISION": (
        "Review and polish the manuscript. Use search_manuscript() to find issues.\n"
        "Use replace_section() to fix content. Check consistency against your state.\n"
        "When fully polished, transition to READY_FOR_HUMAN."
    ),
    "READY_FOR_HUMAN": "The story is complete. No further action needed.",
}

console = Console()
client = OpenAI(base_url=CONFIG["base_url"], api_key="lm-studio")
encoding = tiktoken.get_encoding("cl100k_base")

def get_token_count(text):
    return len(encoding.encode(text))

# --- PROJECT MANAGEMENT ---
def list_projects():
    if not os.path.exists("projects"):
        os.makedirs("projects")
        return []
    return [d for d in os.listdir("projects") if os.path.isdir(os.path.join("projects", d))]

def setup_project(name):
    base_dir = f"projects/{name}"
    os.makedirs(base_dir, exist_ok=True)
    state_path = f"{base_dir}/state.json"
    stats_path = f"{base_dir}/stats.json"
    manuscript_path = f"{base_dir}/manuscript.md"
    
    is_new = not os.path.exists(state_path)
    
    if is_new:
        console.print(Panel(f"🌟 [bold green]Creating New Project: {name}[/bold green]"))
        seed = Prompt.ask("[bold cyan]Enter the initial story seed/prompt[/bold cyan]")
        
        initial_state = {
            "phase": "CHARACTER_CREATION",
            "manuscript_file": manuscript_path,
            "user_feedback": seed,  # The seed becomes the first feedback
            "previous_summary": "",  # LLM's notes-to-self between iterations
            "ai_state": {}  # Free-form state managed by the LLM
        }
        with open(state_path, "w") as f:
            json.dump(initial_state, f, indent=4)
            
        with open(stats_path, "w") as f:
            json.dump({"loops": [], "total_input_tokens": 0, "total_output_tokens": 0, "total_time_seconds": 0}, f)
            
    return state_path, stats_path, manuscript_path

# --- MANUSCRIPT HELPERS ---
def _read_manuscript(manuscript_path):
    """Read manuscript content, return empty string if doesn't exist."""
    if not os.path.exists(manuscript_path):
        return ""
    with open(manuscript_path, "r", encoding="utf-8") as f:
        return f.read()

def _write_manuscript(manuscript_path, content):
    """Write manuscript content."""
    with open(manuscript_path, "w", encoding="utf-8") as f:
        f.write(content.strip())

def _describe_value(v):
    """Compact type+size description of a value for list_state_keys."""
    if isinstance(v, dict):
        return f"object ({len(v)} keys)"
    elif isinstance(v, list):
        return f"list ({len(v)} items)"
    elif isinstance(v, str):
        words = len(v.split())
        return f"string ({words} words)" if words > 10 else f"string ({len(v)} chars)"
    elif isinstance(v, bool):
        return str(v)
    elif isinstance(v, (int, float)):
        return str(v)
    else:
        return type(v).__name__

def get_manuscript_info(manuscript_path):
    """Get manuscript table-of-contents: sections, word counts, line counts.
    Returns NO prose content — use read_manuscript_section or read_manuscript_tail for that."""
    content = _read_manuscript(manuscript_path)
    if not content:
        return {"exists": False, "word_count": 0, "line_count": 0, "sections": []}
    
    lines = content.split("\n")
    sections = []
    section_stack = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("<!-- SECTION:") and not stripped.startswith("<!-- END"):
            name = stripped.replace("<!-- SECTION:", "").replace("-->", "").strip()
            section_stack.append({"name": name, "start": i + 1, "words": 0})
        elif stripped.startswith("<!-- END SECTION:"):
            if section_stack:
                section = section_stack.pop()
                section["end"] = i + 1
                sections.append(section)
        elif section_stack:
            section_stack[-1]["words"] += len(line.split())
    
    return {
        "exists": True,
        "word_count": len(content.split()),
        "line_count": len(lines),
        "sections": [
            {"name": s["name"], "lines": f"{s['start']}-{s['end']}", "words": s["words"]}
            for s in sections
        ]
    }

def read_manuscript_section_content(manuscript_path, section_name):
    """Read content of a specific named section."""
    content = _read_manuscript(manuscript_path)
    start_marker = f"<!-- SECTION: {section_name} -->"
    end_marker = f"<!-- END SECTION: {section_name} -->"
    
    if start_marker not in content:
        return None
    
    start_idx = content.find(start_marker) + len(start_marker)
    end_idx = content.find(end_marker)
    if end_idx == -1:
        return None
    
    return content[start_idx:end_idx].strip()

def read_manuscript_tail(manuscript_path, word_count=500):
    """Read the last N words of the manuscript for continuation context."""
    content = _read_manuscript(manuscript_path)
    if not content:
        return ""
    words = content.split()
    if len(words) <= word_count:
        return content
    return "...\n" + " ".join(words[-word_count:])

def search_in_manuscript(manuscript_path, query, context_lines=2, max_results=10):
    """Search manuscript for text, return matches with surrounding context."""
    content = _read_manuscript(manuscript_path)
    if not content:
        return []
    
    lines = content.split("\n")
    matches = []
    query_lower = query.lower()
    
    for i, line in enumerate(lines):
        if query_lower in line.lower():
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            context = "\n".join(lines[start:end])
            matches.append({"line": i + 1, "context": context})
            if len(matches) >= max_results:
                break
    
    return matches

# --- FUNCTION TOOLS DEFINITION ---
# Each tool does exactly ONE thing with the simplest possible parameters.
# No compound array-of-operations tools. No overloaded read tools.
def get_function_tools():
    """Define atomic function tools for the LLM."""
    return [
        # ── STATE: list keys ──
        {
            "type": "function",
            "function": {
                "name": "list_state_keys",
                "description": "List all keys stored in your persistent state. Returns key names with value types and sizes. Call this first to see what data you've previously saved.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        # ── STATE: read one key ──
        {
            "type": "function",
            "function": {
                "name": "read_state",
                "description": "Read the value stored under a key. Use list_state_keys() first to see available keys.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Key name to read"
                        }
                    },
                    "required": ["key"]
                }
            }
        },
        # ── STATE: write one key ──
        {
            "type": "function",
            "function": {
                "name": "write_state",
                "description": "Store data under a key (overwrites if exists). Use descriptive key names like 'char_alice', 'world_locations', 'plot_act2_beats'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Key name (use snake_case, descriptive)"
                        },
                        "data": {
                            "description": "Any JSON-serializable data to store (object, array, string, number, etc.)"
                        }
                    },
                    "required": ["key", "data"]
                }
            }
        },
        # ── STATE: delete key ──
        {
            "type": "function",
            "function": {
                "name": "delete_state",
                "description": "Delete a key from state storage. Use to clean up obsolete data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Key to delete"
                        }
                    },
                    "required": ["key"]
                }
            }
        },
        # ── MANUSCRIPT: info/outline ──
        {
            "type": "function",
            "function": {
                "name": "get_manuscript_info",
                "description": "Get the manuscript table of contents: section names, line ranges, word counts per section, and total stats. Returns NO prose — for actual text, use read_manuscript_section or read_manuscript_tail.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        # ── MANUSCRIPT: read section ──
        {
            "type": "function",
            "function": {
                "name": "read_manuscript_section",
                "description": "Read the prose content of one named section. Use get_manuscript_info() first to see available section names.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section_name": {
                            "type": "string",
                            "description": "Exact section name from get_manuscript_info()"
                        }
                    },
                    "required": ["section_name"]
                }
            }
        },
        # ── MANUSCRIPT: read tail ──
        {
            "type": "function",
            "function": {
                "name": "read_manuscript_tail",
                "description": "Read the last N words of the manuscript. Use this before writing to get continuation context. Default 500 words.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word_count": {
                            "type": "integer",
                            "description": "Number of words from the end (default: 500)",
                            "default": 500
                        }
                    },
                    "required": []
                }
            }
        },
        # ── MANUSCRIPT: search ──
        {
            "type": "function",
            "function": {
                "name": "search_manuscript",
                "description": "Search the manuscript for a text string (case-insensitive). Returns matching lines with surrounding context. Useful during revision to find inconsistencies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text to search for"
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context around each match (default: 2)",
                            "default": 2
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        # ── MANUSCRIPT: append ──
        {
            "type": "function",
            "function": {
                "name": "append_to_manuscript",
                "description": "Append new prose to the end of the manuscript.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Prose text to append"
                        }
                    },
                    "required": ["content"]
                }
            }
        },
        # ── MANUSCRIPT: create section ──
        {
            "type": "function",
            "function": {
                "name": "create_section",
                "description": "Create a new named section at the end of the manuscript. Wraps content in section markers so it can be replaced or deleted later.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Section name identifier"
                        },
                        "content": {
                            "type": "string",
                            "description": "Section content"
                        }
                    },
                    "required": ["name", "content"]
                }
            }
        },
        # ── MANUSCRIPT: replace section ──
        {
            "type": "function",
            "function": {
                "name": "replace_section",
                "description": "Replace the content of an existing named section, keeping the section markers. Use get_manuscript_info() to see available sections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Section name to replace"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content for the section"
                        }
                    },
                    "required": ["name", "content"]
                }
            }
        },
        # ── MANUSCRIPT: delete section ──
        {
            "type": "function",
            "function": {
                "name": "delete_section",
                "description": "Remove a named section and all its content from the manuscript.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Section name to delete"
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        # ── PHASE: change ──
        {
            "type": "function",
            "function": {
                "name": "change_phase",
                "description": "Transition to a different writing phase. Only transition when you have completed meaningful work in the current phase.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_phase": {
                            "type": "string",
                            "enum": list(STATE_MACHINE.keys()),
                            "description": "Target phase"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for transition"
                        }
                    },
                    "required": ["new_phase", "reason"]
                }
            }
        }
    ]

def execute_function(function_name, arguments, state, manuscript_path):
    """Execute a tool call. Returns minimal data to save tokens."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return {"error": "Invalid JSON arguments"}
    
    ai_state = state.setdefault("ai_state", {})
    
    # ── STATE TOOLS ──
    if function_name == "list_state_keys":
        if not ai_state:
            return {"keys": []}
        return {k: _describe_value(v) for k, v in ai_state.items()}
    
    elif function_name == "read_state":
        key = args.get("key", "")
        if key not in ai_state:
            return {"error": f"'{key}' not found", "available": list(ai_state.keys())}
        return ai_state[key]
    
    elif function_name == "write_state":
        try:
            key = args.get("key", "")
            data = args.get("data")
            if not key:
                return {"error": "Missing 'key' parameter"}
            ai_state[key] = data
            return "ok"
        except Exception as e:
            return {"error": f"Failed to write state: {e}"}
    
    elif function_name == "delete_state":
        key = args.get("key", "")
        if key in ai_state:
            del ai_state[key]
            return "ok"
        return {"error": f"'{key}' not found"}
    
    # ── MANUSCRIPT READ TOOLS ──
    elif function_name == "get_manuscript_info":
        return get_manuscript_info(manuscript_path)
    
    elif function_name == "read_manuscript_section":
        try:
            section_name = args.get("section_name", "")
            if not section_name:
                return {"error": "Missing 'section_name' parameter"}
            content = read_manuscript_section_content(manuscript_path, section_name)
            if content is None:
                info = get_manuscript_info(manuscript_path)
                return {"error": f"Section '{section_name}' not found", "available": [s["name"] for s in info["sections"]]}
            return content
        except Exception as e:
            return {"error": f"Failed to read section: {e}"}
    
    elif function_name == "read_manuscript_tail":
        wc = args.get("word_count", 500)
        return read_manuscript_tail(manuscript_path, wc)
    
    elif function_name == "search_manuscript":
        query = args.get("query", "")
        ctx = args.get("context_lines", 2)
        matches = search_in_manuscript(manuscript_path, query, ctx)
        if not matches:
            return {"matches": [], "note": "No results found"}
        return {"matches": matches}
    
    # ── MANUSCRIPT WRITE TOOLS ──
    elif function_name == "append_to_manuscript":
        try:
            content = _read_manuscript(manuscript_path)
            new_text = args.get("content", "")
            if not isinstance(new_text, str):
                return {"error": "'content' must be a string"}
            content = (content + "\n\n" + new_text).strip()
            _write_manuscript(manuscript_path, content)
            return f"appended {len(new_text.split())} words (total: {len(content.split())})"
        except Exception as e:
            return {"error": f"Failed to append: {e}"}
    
    elif function_name == "create_section":
        try:
            content = _read_manuscript(manuscript_path)
            name = args.get("name", "")
            section_text = args.get("content", "")
            if not name:
                return {"error": "Missing 'name' parameter"}
            if not isinstance(section_text, str):
                return {"error": "'content' must be a string"}
            block = f"<!-- SECTION: {name} -->\n{section_text}\n<!-- END SECTION: {name} -->"
            content = (content + "\n\n" + block).strip()
            _write_manuscript(manuscript_path, content)
            return f"created section '{name}' ({len(section_text.split())} words)"
        except Exception as e:
            return {"error": f"Failed to create section: {e}"}
    
    elif function_name == "replace_section":
        try:
            content = _read_manuscript(manuscript_path)
            name = args.get("name", "")
            new_text = args.get("content", "")
            if not name:
                return {"error": "Missing 'name' parameter"}
            if not isinstance(new_text, str):
                return {"error": "'content' must be a string"}
            start_marker = f"<!-- SECTION: {name} -->"
            end_marker = f"<!-- END SECTION: {name} -->"
            
            if start_marker in content and end_marker in content:
                start_idx = content.find(start_marker)
                end_idx = content.find(end_marker) + len(end_marker)
                content = content[:start_idx] + start_marker + "\n" + new_text + "\n" + end_marker + content[end_idx:]
                _write_manuscript(manuscript_path, content)
                return f"replaced section '{name}' ({len(new_text.split())} words)"
            else:
                info = get_manuscript_info(manuscript_path)
                return {"error": f"Section '{name}' not found", "available": [s["name"] for s in info["sections"]]}
        except Exception as e:
            return {"error": f"Failed to replace section: {e}"}
    
    elif function_name == "delete_section":
        try:
            content = _read_manuscript(manuscript_path)
            name = args.get("name", "")
            if not name:
                return {"error": "Missing 'name' parameter"}
            start_marker = f"<!-- SECTION: {name} -->"
            end_marker = f"<!-- END SECTION: {name} -->"
            
            if start_marker in content and end_marker in content:
                start_idx = content.find(start_marker)
                end_idx = content.find(end_marker) + len(end_marker)
                content = content[:start_idx] + content[end_idx:]
                _write_manuscript(manuscript_path, content)
                return f"deleted section '{name}'"
            else:
                return {"error": f"Section '{name}' not found"}
        except Exception as e:
            return {"error": f"Failed to delete section: {e}"}
    
    # ── PHASE ──
    elif function_name == "change_phase":
        try:
            new_phase = args.get("new_phase", "")
            reason = args.get("reason", "")
            if not new_phase:
                return {"error": "Missing 'new_phase' parameter"}
            valid = STATE_MACHINE.get(state["phase"], {}).get("transitions", [])
            if new_phase not in valid:
                return {"error": f"Cannot go from {state['phase']} to {new_phase}", "valid": valid}
            old = state["phase"]
            state["phase"] = new_phase
            return f"phase: {old} → {new_phase}"
        except Exception as e:
            return {"error": f"Failed to change phase: {e}"}
    
    else:
        return {"error": f"Unknown function: {function_name}"}

# --- SYSTEM PROMPT BUILDER ---
def build_system_prompt(state, manuscript_path):
    """Build a focused system prompt using only what the LLM needs this iteration.
    
    Instead of dumping the full state machine, all phases, and generic workflow tips,
    we include: current phase + its guidance, manuscript stats at a glance, state key
    names (not values), and the previous iteration summary for continuity.
    """
    phase = state["phase"]
    info = STATE_MACHINE.get(phase, {})
    transitions = info.get("transitions", [])
    
    # Compact manuscript stats
    ms_info = get_manuscript_info(manuscript_path)
    if ms_info["exists"]:
        section_names = ", ".join(s["name"] for s in ms_info["sections"]) or "none"
        ms_status = f"{ms_info['word_count']} words, {ms_info['line_count']} lines, sections: [{section_names}]"
    else:
        ms_status = "empty"
    
    # State key names only (not values)
    ai_keys = ", ".join(state.get("ai_state", {}).keys()) or "none"
    
    prompt = (
        f"You are a story-writing engine.\n\n"
        f"PHASE: {phase} — {info.get('description', '')}\n"
        f"Transitions: {', '.join(transitions) if transitions else 'terminal (story complete)'}\n"
        f"Manuscript: {ms_status}\n"
        f"State keys: [{ai_keys}]\n\n"
        f"GUIDE:\n{PHASE_GUIDE.get(phase, '')}\n\n"
    )
    
    # Previous iteration context — gives the LLM memory between loops
    prev = state.get("previous_summary", "")
    if prev:
        prompt += f"YOUR NOTES FROM LAST ITERATION:\n{prev}\n\n"
    
    prompt += (
        "IMPORTANT: End your response with a brief note-to-self summarizing what you "
        "accomplished and what to do next. This note will be shown to you at the start "
        "of the next iteration as memory."
    )
    
    return prompt

# --- ENGINE ---
def run_iteration(state, project_name, manuscript_path):
    start_time = time.time()
    phase = state["phase"]
    
    # Get state machine info for display
    current_phase_info = STATE_MACHINE.get(phase, {})
    phase_description = current_phase_info.get("description", "Unknown phase")
    available_transitions = current_phase_info.get("transitions", [])
    
    # Build focused system prompt
    system_prompt = build_system_prompt(state, manuscript_path)
    user_msg = state.get("user_feedback") or "Continue with your creative process."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]
    
    # Get manuscript info for display
    manuscript_info = get_manuscript_info(manuscript_path)
    
    # Build phase visualization
    transitions_text = " → ".join(available_transitions) if available_transitions else "✓ Terminal"
    manuscript_summary = f"{manuscript_info['word_count']} words" if manuscript_info['exists'] else "Empty"
    
    console.print(Panel(
        f"[bold blue]Project:[/bold blue] {project_name}\n"
        f"[bold magenta]Phase:[/bold magenta] {phase}\n"
        f"[dim]{phase_description}[/dim]\n"
        f"[bold cyan]Next Steps:[/bold cyan] {transitions_text}\n"
        f"[bold yellow]Manuscript:[/bold yellow] {manuscript_summary}",
        title="🚀 Story Engine Status"
    ))
    
    input_tokens = 0
    output_tokens = 0
    full_response = ""
    tool_calls_made = []
    max_iterations = 15  # Allow more iterations since tools are granular now
    iteration = 0
    
    try:
        while iteration < max_iterations:
            iteration += 1
            
            # Make API call with tools
            response = client.chat.completions.create(
                model=CONFIG["model"],
                messages=messages,
                tools=get_function_tools(),
                tool_choice="auto"
            )
            
            # Safely parse response with fallback for malformed responses
            try:
                message = response.choices[0].message
            except (IndexError, AttributeError, KeyError) as e:
                raise RuntimeError(f"Invalid response structure: {e}")
            
            input_tokens += response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else get_token_count(str(messages))
            output_tokens += response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else get_token_count(str(getattr(message, 'content', None) or ""))
            
            # Display assistant's response if any
            message_content = getattr(message, 'content', None)
            if message_content:
                console.print(f"\n[bold yellow]LLM:[/bold yellow] {message_content}")
                full_response += message_content + "\n"
            
            # Check if there are tool calls (safely access the attribute)
            tool_calls = getattr(message, 'tool_calls', None)
            if not tool_calls:
                # No more tool calls, we're done
                break
            
            # Add assistant message to conversation
            messages.append(message)
            
            # Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                
                console.print(f"\n[bold cyan]🔧 Tool Call:[/bold cyan] {function_name}")
                console.print(f"[dim]Arguments: {arguments[:200]}{'...' if len(arguments) > 200 else ''}[/dim]")
                
                # Execute the function
                result = execute_function(function_name, arguments, state, manuscript_path)
                result_json = json.dumps(result)
                
                # Show truncated result in console
                display_result = result_json[:300] + "..." if len(result_json) > 300 else result_json
                console.print(f"[bold green]✓ Result:[/bold green] {display_result}")
                
                tool_calls_made.append({
                    "function": function_name,
                    "arguments": arguments,
                    "result_size": len(result_json)
                })
                
                # Add tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_json
                })
        
        if iteration >= max_iterations:
            console.print(f"[yellow]Warning: Reached maximum tool call iterations ({max_iterations})[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]LM Studio Error:[/bold red] {e}")
        duration = time.time() - start_time
        return None, input_tokens, output_tokens, duration, []

    # Capture the LLM's last text response as the iteration summary for next loop
    if full_response.strip():
        summary = full_response.strip()
        # Truncate to avoid bloating the system prompt on next iteration
        if len(summary) > 800:
            summary = summary[:800] + "..."
        state["previous_summary"] = summary

    duration = time.time() - start_time
    return full_response, input_tokens, output_tokens, duration, tool_calls_made

def update_logs(stats_path, loop_data):
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    stats["loops"].append(loop_data)
    stats["total_input_tokens"] += loop_data["in_tokens"]
    stats["total_output_tokens"] += loop_data["out_tokens"]
    stats["total_time_seconds"] = stats.get("total_time_seconds", 0) + loop_data.get("duration_seconds", 0)
    
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    return stats

def show_stats(stats):
    table = Table(title="📊 Ralph Engine Statistics")
    table.add_column("Loop #", style="cyan")
    table.add_column("Phase", style="magenta")
    table.add_column("Result", style="green")
    table.add_column("Tokens (In/Out)", style="yellow")
    table.add_column("Time", style="blue")

    for i, loop in enumerate(stats["loops"][-5:]):
        duration = loop.get("duration_seconds", 0)
        time_str = f"{duration:.1f}s"
        table.add_row(
            str((max(len(stats["loops"]) - 5, 0) + i + 1)), 
            loop["phase"], 
            loop["status"], 
            f"{loop['in_tokens']}/{loop['out_tokens']}",
            time_str
        )
    
    total_tokens = stats['total_input_tokens'] + stats['total_output_tokens']
    total_time = stats.get('total_time_seconds', 0)
    total_time_str = f"{total_time/60:.1f}m" if total_time >= 60 else f"{total_time:.1f}s"
    
    console.print(table)
    console.print(f"[dim]Total Token Burn: {total_tokens} | Total Time: {total_time_str}[/dim]\n")

# --- MAIN ---
if __name__ == "__main__":
    console.clear()
    console.print(Panel("[bold green]WELCOME TO THE RALPH LOOP STORY ENGINE[/bold green]\nFully Local, Fully Autonomous."))

    # Project Selection Logic
    projects = list_projects()
    if projects:
        console.print("[bold yellow]Existing Projects:[/bold yellow]")
        for i, p in enumerate(projects):
            console.print(f" {i+1}. {p}")
        console.print(" n. [Create New Project]")
        
        choice = Prompt.ask("Choose a project", choices=[str(i+1) for i in range(len(projects))] + ["n"])
        if choice == "n":
            project_name = Prompt.ask("Enter new project name").strip().replace(" ", "_")
        else:
            project_name = projects[int(choice)-1]
    else:
        project_name = Prompt.ask("No projects found. Enter new project name").strip().replace(" ", "_")

    state_p, stats_p, manus_p = setup_project(project_name)
    
    with open(state_p, "r") as f: 
        state = json.load(f)
    
    # Migrate older projects: ensure previous_summary field exists
    if "previous_summary" not in state:
        state["previous_summary"] = ""
    
    # Validate and fix phase if needed
    if state["phase"] not in STATE_MACHINE:
        console.print(f"[yellow]Warning: Unknown phase '{state['phase']}'. Resetting to CHARACTER_CREATION.[/yellow]")
        state["phase"] = "CHARACTER_CREATION"
    
    # Check if already in terminal state
    if state["phase"] == "READY_FOR_HUMAN":
        console.print("[bold green]✨ This project is marked as complete! ✨[/bold green]")
        console.print(f"[dim]Manuscript: {manus_p}[/dim]")
        if Confirm.ask("Do you want to continue working on it anyway?"):
            state["phase"] = "REVISION"
            console.print("[yellow]Moving back to REVISION phase...[/yellow]")
        else:
            sys.exit(0)

    try:
        while True:
            raw_text, in_t, out_t, duration, tool_calls = run_iteration(state, project_name, manus_p)
            
            if raw_text is None:
                # Log failed attempt with tokens that were used
                current_stats = update_logs(stats_p, {
                    "timestamp": str(datetime.now()),
                    "phase": state["phase"],
                    "status": "Failed: Connection Error",
                    "in_tokens": in_t,
                    "out_tokens": out_t,
                    "duration_seconds": duration
                })
                show_stats(current_stats)
                console.print("[yellow]⚠ Connection error. Retrying in 3 seconds...[/yellow]")
                time.sleep(3)  # Wait before retrying
                continue

            status = "Success"
            
            # Check if we've reached terminal state
            if state["phase"] == "READY_FOR_HUMAN":
                console.print("\n[bold green]✨ Story is ready for human review! ✨[/bold green]")
                console.print(f"[dim]Manuscript saved to: {manus_p}[/dim]")
                status = "Completed"
            
            # Display tool call summary
            if tool_calls:
                console.print(f"\n[dim]🔧 Made {len(tool_calls)} tool call(s) this iteration[/dim]")
            
            state["user_feedback"] = ""  # Clear after successful loop

            # Log Stats
            current_stats = update_logs(stats_p, {
                "timestamp": str(datetime.now()),
                "phase": state["phase"],
                "status": status,
                "in_tokens": in_t,
                "out_tokens": out_t,
                "duration_seconds": duration
            })
            
            with open(state_p, "w") as f:
                json.dump(state, f, indent=4)
            
            show_stats(current_stats)
            
            # Break if story is complete
            if state["phase"] == "READY_FOR_HUMAN":
                break

            # Control Flow
            if not CONFIG["auto_pilot"]:
                feedback = console.input("\n[yellow]Paused. Feedback (or Enter to loop): [/yellow]")
                if feedback.lower() == 'exit':
                    if CONFIG["stop_only_on_complete"]:
                        console.print("[yellow]Cannot exit - stop_only_on_complete mode requires reaching READY_FOR_HUMAN. Use Ctrl+C to force stop.[/yellow]")
                    else:
                        break
                if feedback: state["user_feedback"] = feedback
            else:
                # 2-second window to hit Ctrl+C or prepare to type an interruption
                time.sleep(2) 

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopping Engine... State saved.[/bold red]")
