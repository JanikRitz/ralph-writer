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
            "ai_state": {}  # Free-form state managed by the LLM
        }
        with open(state_path, "w") as f:
            json.dump(initial_state, f, indent=4)
            
        with open(stats_path, "w") as f:
            json.dump({"loops": [], "total_input_tokens": 0, "total_output_tokens": 0, "total_time_seconds": 0}, f)
            
    return state_path, stats_path, manuscript_path

# --- PARSING ---
def extract_last_json(text):
    brace_count, end_index = 0, -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '}':
            if brace_count == 0: end_index = i + 1
            brace_count += 1
        elif text[i] == '{':
            brace_count -= 1
            if brace_count == 0 and end_index != -1:
                return text[i:end_index]
    return None

def deep_merge(base, updates):
    """Recursively merge updates into base dictionary."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

# --- MANUSCRIPT MANAGEMENT ---
def get_manuscript_summary(manuscript_path, max_chars=3000):
    """Get a summary of the manuscript for LLM context."""
    if not os.path.exists(manuscript_path):
        return {
            "exists": False,
            "word_count": 0,
            "sections": [],
            "preview": "",
            "full_content": ""
        }
    
    with open(manuscript_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    words = content.split()
    word_count = len(words)
    
    # Extract sections (looking for markdown headers or section markers)
    sections = []
    lines = content.split("\n")
    current_section = None
    
    for i, line in enumerate(lines):
        if line.startswith("#") or line.startswith("<!-- SECTION:"):
            if line.startswith("<!-- SECTION:"):
                section_name = line.replace("<!-- SECTION:", "").replace("-->", "").strip()
            else:
                section_name = line.strip("# ").strip()
            sections.append({"line": i + 1, "name": section_name})
    
    # Get preview (last N characters)
    preview = content[-max_chars:] if len(content) > max_chars else content
    if len(content) > max_chars:
        preview = "..." + preview
    
    return {
        "exists": True,
        "word_count": word_count,
        "char_count": len(content),
        "sections": sections,
        "preview": preview,
        "full_content": content
    }

def apply_manuscript_operations(manuscript_path, operations):
    """Apply manuscript operations: append, replace, insert, delete."""
    # Read current content
    if os.path.exists(manuscript_path):
        with open(manuscript_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = ""
    
    for op in operations:
        op_type = op.get("type")
        
        if op_type == "append":
            # Add to end
            text = op.get("content", "")
            content += "\n\n" + text
            
        elif op_type == "replace_all":
            # Replace entire manuscript
            content = op.get("content", "")
            
        elif op_type == "replace_section":
            # Replace content between section markers
            section_name = op.get("section")
            new_content = op.get("content", "")
            
            start_marker = f"<!-- SECTION: {section_name} -->"
            end_marker = f"<!-- END SECTION: {section_name} -->"
            
            if start_marker in content and end_marker in content:
                start_idx = content.find(start_marker)
                end_idx = content.find(end_marker) + len(end_marker)
                content = content[:start_idx] + start_marker + "\n" + new_content + "\n" + end_marker + content[end_idx:]
            else:
                # Section doesn't exist, append it
                content += f"\n\n{start_marker}\n{new_content}\n{end_marker}"
                
        elif op_type == "insert_section":
            # Insert a new section with markers
            section_name = op.get("section")
            new_content = op.get("content", "")
            position = op.get("position", "end")  # "start", "end", or after another section
            
            section_block = f"<!-- SECTION: {section_name} -->\n{new_content}\n<!-- END SECTION: {section_name} -->"
            
            if position == "start":
                content = section_block + "\n\n" + content
            else:
                content += "\n\n" + section_block
                
        elif op_type == "delete_section":
            # Remove a section
            section_name = op.get("section")
            start_marker = f"<!-- SECTION: {section_name} -->"
            end_marker = f"<!-- END SECTION: {section_name} -->"
            
            if start_marker in content and end_marker in content:
                start_idx = content.find(start_marker)
                end_idx = content.find(end_marker) + len(end_marker)
                content = content[:start_idx] + content[end_idx:]
    
    # Write back
    with open(manuscript_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    
    return content

# --- ENGINE ---
def run_iteration(state, project_name, manuscript_path):
    start_time = time.time()
    phase = state["phase"]
    ai_state = state.get("ai_state", {})
    
    # Get state machine info
    current_phase_info = STATE_MACHINE.get(phase, {})
    phase_description = current_phase_info.get("description", "Unknown phase")
    available_transitions = current_phase_info.get("transitions", [])
    
    # Get manuscript summary
    manuscript_info = get_manuscript_summary(manuscript_path)
    
    # Build manuscript status text
    if manuscript_info["exists"]:
        sections_text = ", ".join([s["name"] for s in manuscript_info["sections"]]) if manuscript_info["sections"] else "No sections marked"
        manuscript_status = (
            f"Word count: {manuscript_info['word_count']}\n"
            f"Sections: {sections_text}\n"
            f"Recent content:\n{manuscript_info['preview']}"
        )
    else:
        manuscript_status = "Empty - no content yet"
    
    # Build state machine visualization
    state_machine_text = "\n".join([
        f"  - {p}: {STATE_MACHINE[p]['description']}" 
        for p in STATE_MACHINE.keys()
    ])
    
    system_prompt = (
        f"You are a collaborative story-writing engine.\n\n"
        f"STATE MACHINE:\n{state_machine_text}\n\n"
        f"CURRENT PHASE: {phase}\n"
        f"Description: {phase_description}\n"
        f"Available transitions: {', '.join(available_transitions) if available_transitions else 'None (terminal state)'}\n\n"
        f"MANUSCRIPT STATUS:\n{manuscript_status}\n\n"
        f"Your current state:\n{json.dumps(ai_state, indent=2)}\n\n"
        "INSTRUCTIONS:\n"
        "- Work on the current phase. You decide when you've done enough for this iteration.\n"
        "- You can stay in the same phase for multiple iterations if needed.\n"
        "- Only transition to a new phase when you feel the current phase is complete or you need to work on something else.\n"
        "- The 'READY_FOR_HUMAN' phase is terminal - only go there when the story is truly complete and polished.\n\n"
        "MANUSCRIPT OPERATIONS:\n"
        "Use 'manuscript_ops' (an array of operations) to modify the manuscript:\n"
        "  - {\"type\": \"append\", \"content\": \"text\"} - Add text to the end\n"
        "  - {\"type\": \"insert_section\", \"section\": \"name\", \"content\": \"text\"} - Create a new named section\n"
        "  - {\"type\": \"replace_section\", \"section\": \"name\", \"content\": \"text\"} - Update an existing section\n"
        "  - {\"type\": \"delete_section\", \"section\": \"name\"} - Remove a section\n"
        "  - {\"type\": \"replace_all\", \"content\": \"text\"} - Replace entire manuscript (use carefully!)\n\n"
        "RESPONSE FORMAT:\n"
        "Think through your decisions, then end your response with a JSON block containing ONLY the fields you want to update.\n"
        "You control your own state structure - add any fields you need for:\n"
        "  - Character profiles, relationships, arcs\n"
        "  - World building notes, rules, locations\n"
        "  - Plot beats, timelines, story structure\n"
        "  - Research notes, ideas, reminders\n"
        "  - Progress tracking, todos, open questions\n"
        "Your state updates are merged with existing state, so you only need to include what's changing.\n\n"
        "JSON EXAMPLES:\n\n"
        "Example 1 - Adding characters and writing a scene:\n"
        "{\n"
        "  \"characters\": {\n"
        "    \"alice\": {\"age\": 28, \"role\": \"protagonist\", \"motivation\": \"find her sister\"},\n"
        "    \"marcus\": {\"age\": 45, \"role\": \"mentor\", \"secret\": \"knows the truth\"}\n"
        "  },\n"
        "  \"manuscript_ops\": [\n"
        "    {\"type\": \"insert_section\", \"section\": \"chapter_1\", \"content\": \"Chapter 1\\n\\nAlice...\"}\n"
        "  ],\n"
        "  \"phase_progress\": \"Created main characters and started chapter 1\"\n"
        "}\n\n"
        "Example 2 - Tracking plot structure:\n"
        "{\n"
        "  \"plot_beats\": [\n"
        "    {\"act\": 1, \"beat\": \"inciting_incident\", \"description\": \"Alice receives mysterious letter\"},\n"
        "    {\"act\": 2, \"beat\": \"midpoint\", \"description\": \"Discovers Marcus is her father\"}\n"
        "  ],\n"
        "  \"timeline\": {\"day_1\": \"Letter arrives\", \"day_3\": \"Meets Marcus\"},\n"
        "  \"open_questions\": [\"How did the letter get there?\", \"What is Marcus hiding?\"]\n"
        "}\n\n"
        "Example 3 - Revising a section:\n"
        "{\n"
        "  \"manuscript_ops\": [\n"
        "    {\"type\": \"replace_section\", \"section\": \"chapter_1\", \"content\": \"Chapter 1 (Revised)\\n\\nAlice...\"}\n"
        "  ],\n"
        "  \"revision_notes\": \"Strengthened Alice's voice, added more tension\",\n"
        "  \"phase\": \"SCENE_WRITING\"\n"
        "}\n\n"
        "Example 4 - Phase transition:\n"
        "{\n"
        "  \"phase\": \"PLOT_OUTLINING\",\n"
        "  \"phase_progress\": \"Characters complete. Moving to plot structure.\"\n"
        "}\n"
    )
    user_msg = state.get("user_feedback") or "Continue with your creative process."
    
    input_tokens = get_token_count(system_prompt + user_msg)
    full_response = ""
    
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
    console.print(Panel(f"[bold cyan]System Prompt:[/bold cyan]\n{system_prompt}\n\n[bold cyan]User Message:[/bold cyan]\n{user_msg}", title="📝 LLM Input", border_style="cyan"))
    console.print("\n[bold yellow]LLM Response:[/bold yellow]")
    
    try:
        stream = client.chat.completions.create(
            model=CONFIG["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                print(content, end="", flush=True)
    except Exception as e:
        console.print(f"[bold red]LM Studio Error:[/bold red] {e}")
        duration = time.time() - start_time
        # Still track input tokens even on failure
        output_tokens = get_token_count(full_response) if full_response else 0
        return None, input_tokens, output_tokens, duration

    output_tokens = get_token_count(full_response)
    duration = time.time() - start_time
    return full_response, input_tokens, output_tokens, duration

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
            str(len(stats["loops"]) - 5 + i + 1), 
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
            raw_text, in_t, out_t, duration = run_iteration(state, project_name, manus_p)
            
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
                console.input("[red]Connection failed. Check LM Studio and press Enter to retry...[/red]")
                continue

            json_str = extract_last_json(raw_text)
            status = "Success"
            
            try:
                if not json_str: raise ValueError("No JSON block found.")
                data = json.loads(json_str)
                
                # Handle manuscript operations
                if data.get("manuscript_ops"):
                    operations = data["manuscript_ops"]
                    if not isinstance(operations, list):
                        operations = [operations]
                    apply_manuscript_operations(manus_p, operations)
                    console.print(f"\n[dim]✍️  Applied {len(operations)} manuscript operation(s)[/dim]")
                
                # Update phase (optional - only when AI wants to change it)
                if "phase" in data:
                    new_phase = data["phase"]
                    current_phase_info = STATE_MACHINE.get(state["phase"], {})
                    valid_transitions = current_phase_info.get("transitions", [])
                    
                    # Validate transition
                    if new_phase not in valid_transitions:
                        raise ValueError(
                            f"Invalid phase transition: {state['phase']} -> {new_phase}. "
                            f"Valid transitions: {', '.join(valid_transitions)}"
                        )
                    
                    old_phase = state["phase"]
                    state["phase"] = new_phase
                    
                    # Show transition notification
                    if old_phase != new_phase:
                        console.print(f"\n[bold green]📍 Phase Transition: {old_phase} → {new_phase}[/bold green]")
                    
                    # Check if we've reached terminal state
                    if new_phase == "READY_FOR_HUMAN":
                        console.print("\n[bold green]✨ Story is ready for human review! ✨[/bold green]")
                        console.print(f"[dim]Manuscript saved to: {manus_p}[/dim]")
                        break
                
                # Merge all updates (except manuscript_ops) into AI state
                # This preserves existing state and only updates changed fields
                updates = {k: v for k, v in data.items() if k not in ["manuscript_ops", "phase"]}
                state["ai_state"] = deep_merge(state.get("ai_state", {}), updates)
                state["user_feedback"] = ""  # Clear after successful loop
                
            except Exception as e:
                status = f"Failed: {str(e)}"
                state["user_feedback"] = ""  # Don't get stuck in a loop of errors without fresh input

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

            # Control Flow
            if "Failed" in status:
                console.print(f"[bold red]⚠️ ERROR:[/bold red] {status}")
                if CONFIG["stop_only_on_complete"]:
                    console.print("[yellow]Auto-retrying due to stop_only_on_complete mode...[/yellow]")
                    time.sleep(2)  # Brief pause before retry
                else:
                    state["user_feedback"] = console.input("[yellow]The AI failed to update the state. Help it out with a hint: [/yellow]")
            elif not CONFIG["auto_pilot"]:
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
