# Ultralight Code Assistant -- User Guide

This guide walks you through everything from installation to daily use.

---

## Table of Contents

1. [Installation](#1-installation)
2. [First Launch](#2-first-launch)
3. [The Interface](#3-the-interface)
4. [Writing Code](#4-writing-code)
5. [Multi-Turn Conversations](#5-multi-turn-conversations)
6. [Code Review and Debugging](#6-code-review-and-debugging)
7. [Project Context](#7-project-context)
8. [Switching Models](#8-switching-models)
9. [Terminal Mode](#9-terminal-mode)
10. [API Mode](#10-api-mode)
11. [Configuration](#11-configuration)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Installation

### Windows -- Zero Install (recommended)

No Python installation needed. This downloads a portable Python and handles everything.

1. Open PowerShell and run:
```powershell
git clone https://github.com/densanon-devs/ultralight-coder.git
```
2. Open the `ultralight-coder` folder in File Explorer
3. **Double-click `start.bat`**

That's it. First run will:
- Download portable Python 3.12 (~25MB)
- Install pip and all dependencies
- Download the default AI model (469MB)
- Start the app

**First run takes 5-10 minutes** (downloading). Every run after that starts in ~15 seconds.

### Linux / macOS -- One-Line Install

```bash
curl -fsSL https://raw.githubusercontent.com/densanon-devs/ultralight-coder/master/install.sh | bash
```

With GPU acceleration:
```bash
curl -fsSL https://raw.githubusercontent.com/densanon-devs/ultralight-coder/master/install.sh | bash -s -- --gpu
```

### Manual Install (all platforms)

Requires Python 3.10-3.12 (3.13+ will work but installs slower).

```bash
git clone https://github.com/densanon-devs/ultralight-coder.git
cd ultralight-coder

# Install llama-cpp-python with pre-built wheels (fast, no C++ compiler needed)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Install everything else
pip install -r requirements.txt

# Download the AI model (469MB)
python download_model.py

# Start the app
python launch.py
```

**Important:** Always install llama-cpp-python with the `--extra-index-url` flag. Without it, pip tries to compile from C++ source which takes 20+ minutes and requires a C++ compiler.

For NVIDIA GPU acceleration, use `/whl/cu121` instead of `/whl/cpu`:
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### First-Time Internet Requirements

The first launch downloads a sentence-transformer model (~90MB) from HuggingFace. After that, the app works **fully offline**. If you get a connection error on first launch:

```bash
# Windows (start.bat users)
set TRANSFORMERS_OFFLINE=0
set HF_HUB_OFFLINE=0
python\python.exe server.py

# Linux/macOS / manual install
TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 python server.py
```

### What gets installed

| Component | Size | Purpose |
|---|---|---|
| Python packages | ~1.5 GB | llama-cpp-python, sentence-transformers, FastAPI, FAISS, etc. |
| Sentence-transformer model | ~90 MB | Cached locally (first run only) |
| Coding model (Qwen 0.5B) | 469 MB | The AI model that generates code |

Total disk usage: ~2 GB after first setup.

---

## 2. First Launch

### Windows (start.bat)

Double-click `start.bat`. It opens the app in a native desktop window. If the desktop window doesn't appear, it falls back to your browser at `http://localhost:8000`.

### Windows (manual / PowerShell)

```powershell
cd ultralight-coder
python\python.exe server.py
```
Then open `http://localhost:8000` in your browser.

### Linux / macOS

```bash
cd ultralight-coder
python launch.py
```

**What happens:**
1. Python version is checked
2. GPU is detected (if available)
3. Missing dependencies are offered for installation
4. You pick a model (or it uses the default)
5. A native desktop window opens with the UI

If pywebview isn't available, it falls back to opening in your browser at `http://localhost:8000`.

**Other launch modes:**
```bash
python launch.py --browser    # Force browser instead of desktop window
python launch.py --cli        # Terminal mode (no UI)
python launch.py --port 9000  # Use a different port
```

**First-run note:** The first launch takes longer (~30 seconds) because the sentence-transformer embedding model downloads and caches. Every launch after that is faster (~10 seconds).

---

## 3. The Interface

The UI has four main areas:

### Header Bar
- **Ultralight Code Assistant** -- app title
- **Green dot** -- shows the model is loaded and ready (yellow = still loading)
- **Model name** -- displays which model is active
- **New Chat** -- clears the conversation and starts fresh
- **Model dropdown** -- switch between downloaded models

### Project Bar
- **Path input** -- type or paste a directory path
- **Index button** -- indexes that directory so the model can reference your code
- **Status text** -- shows how many files/chunks are indexed

### Chat Area
- Messages appear as bubbles -- your prompts on the right (blue), responses on the left (dark)
- Code blocks have a **copy** button that appears on hover
- Each response shows timing info at the bottom

### Input Area
- **{ } button** -- toggles the code attachment panel
- **Text input** -- type your prompt here
- **Send button** -- or press Enter
- Shift+Enter inserts a new line without sending

---

## 4. Writing Code

Just describe what you want in plain language. The assistant supports 12 languages.

**Examples:**

```
Write a Python function that finds all duplicates in a list
```

```
Create a Go HTTP server with middleware for logging and auth
```

```
Write a SQL query that finds the top customers by total spend
```

```
Build a TypeScript generic class for an observable value
```

```
Write a Rust function that reads a file and returns a Result
```

```
Create a bash script that backs up a directory with a timestamp
```

**Tips for best results:**
- Name the language explicitly ("Write a **Go** function...")
- Be specific about what you want ("...with error handling and retry logic")
- Mention frameworks if relevant ("...using Express" or "...using FastAPI")
- The model works best with focused, single-task prompts

---

## 5. Multi-Turn Conversations

The assistant remembers what you said earlier in the conversation. You can build on previous responses:

```
You:  Write a Python class for a user account with name and email
Bot:  [generates User class]

You:  Add password hashing to it
Bot:  [updates the class with bcrypt hashing]

You:  Now write tests for it
Bot:  [generates pytest tests for the User class]
```

The conversation history is maintained server-side. Click **New Chat** to start fresh.

**How much history is kept:** The last 30 turns (configurable in `config.yaml` under `memory.short_term.max_turns`). Older turns are compressed into a summary automatically.

---

## 6. Code Review and Debugging

Click the **{ }** button next to the input to open the code attachment panel.

### Review existing code

1. Click **{ }** to open the code panel
2. Paste your code into the green textarea
3. Type a prompt like "Review this code for bugs" or "How can I improve this?"
4. Click Send

### Fix broken code

1. Click **{ }**
2. Paste the broken code
3. Type "Fix the bug in this code" or describe the specific issue
4. Click Send

The code panel closes automatically after you send. The assistant sees both your code and your description.

**Example prompts for code review:**
- "Review this for security issues"
- "Is there a memory leak in this code?"
- "Refactor this to be more readable"
- "Why does this throw a TypeError?"
- "Add error handling to this"

---

## 7. Project Context

This is the most powerful feature. When you index a project directory, the assistant can reference your actual codebase when generating code.

### How to index

1. Type your project path in the **Project** bar at the top
   - Example: `/home/user/my-webapp` or `C:\Users\me\projects\api`
2. Click **Index**
3. Wait a few seconds (you'll see "X chunks from Y files" when done)

### What gets indexed

- All source files matching supported extensions (.py, .js, .ts, .go, .rs, .java, .rb, etc.)
- Config files (.yaml, .json, .toml)
- Documentation (.md, .txt)

### What gets skipped

- `node_modules/`, `__pycache__/`, `.git/`, `venv/` and other common ignore patterns
- Files larger than 500KB
- Binary files

### How it works

After indexing, every prompt you send automatically searches the index for relevant code snippets. These snippets are injected into the model's context, so it can:

- Match your naming conventions
- Use your existing imports and frameworks
- Reference your actual function/class names
- Follow your project's patterns

### Example

After indexing a FastAPI project:

```
You:  Add a new endpoint for user registration
Bot:  [generates code that matches your existing route patterns, uses your User model, follows your error handling style]
```

### Re-indexing

If your code changes significantly, click **Index** again to refresh. The old index is replaced.

### Clearing

To remove the project index, send a POST to `/project/clear` or restart the server.

---

## 8. Switching Models

### Downloading additional models

```bash
python download_model.py --model coder-0.5b    # 469MB  -- fast, good for most tasks
python download_model.py --model coder-1.5b    # 1.1GB  -- balanced
python download_model.py --model coder-3b      # 2.0GB  -- best quality
python download_model.py --model llama          # 770MB  -- good general purpose
python download_model.py --model deepseek-1.3b  # 834MB  -- efficient
python download_model.py --model phi-3.5-mini   # 2.3GB  -- strong performance
```

See all options: `python download_model.py --help`

### Hot-swapping in the UI

Use the **model dropdown** in the top-right corner of the header. Select a different model and it will:

1. Show "Switching model..." in the status area
2. Unload the current model from memory
3. Load the new model
4. Auto-adjust augmentor mode (rerank1 for small models, rerank for large)
5. Show the new model name when ready

**No restart needed.** All downloaded `.gguf` files in the `models/` directory appear in the dropdown.

### Which model to pick

| Your situation | Recommended model |
|---|---|
| Low RAM (< 4GB) | Qwen 0.5B (469MB) |
| General use | Qwen 0.5B (469MB) or Llama 1B (770MB) |
| Need maximum quality | Qwen 3B (2.0GB) |
| Have a GPU | Any model -- GPU offload makes them all fast |
| Complex refactoring | Qwen 1.5B+ (0.5B struggles with multi-step changes) |

### Tips for small models (0.5B)

The 0.5B model handles single-task prompts excellently but struggles with compound instructions. If a response misses what you wanted:

- **Be explicit about the language**: "Write a **Go** function..." not just "write a function..."
- **One change at a time**: Instead of "change the type and swap the checks and keep the old logic", break it into steps
- **Repeat context**: "In the CheckCondition function from before, change X to Y"
- **Upgrade to 1.5B** for complex refactoring — use the dropdown to switch without restarting

---

## 9. Terminal Mode

For SSH sessions, headless servers, or if you prefer the terminal:

```bash
python main.py
```

This gives you an interactive REPL:

```
  Ultralight Code Assistant v0.1.0
  Model: qwen2.5-coder-0.5b-instruct-q4_k_m
  Type /help for commands, /quit to exit

you> Write a function to check if a number is prime
[generates code with rich markdown formatting]

you> Now make it handle negative numbers
[generates updated version]
```

### Terminal commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/modules` | List available modules |
| `/memory` | Show memory status |
| `/remember X` | Store a fact in long-term memory |
| `/explain X` | Explain how a prompt would be routed |
| `/status` | System status |
| `/perf` | Last response performance stats |
| `/quit` | Exit |

---

## 10. API Mode

The REST API lets you integrate the assistant into other tools, scripts, or editors.

### Start the server

```bash
python server.py              # Default: localhost:8000
python server.py --port 9000  # Custom port
```

### Generate code

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to reverse a linked list"}'
```

### Stream code (token by token)

```bash
curl -N -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a sorting algorithm in Go"}'
```

### Send code for review

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Fix the bug in this code",
    "code": "def add(a, b):\n    return a - b"
  }'
```

### Full API docs

Open `http://localhost:8000/docs` in your browser for interactive Swagger documentation with all endpoints.

---

## 11. Configuration

Edit `config.yaml` in the project root. The most useful settings:

### Model settings

```yaml
base_model:
  path: models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf  # Which model to load
  context_length: 4096    # Token context window (higher = more context, more RAM)
  gpu_layers: 99          # Layers to offload to GPU (99 = all, 0 = CPU only)
  threads: 8              # CPU threads (set to your core count)
  temperature: 0.2        # 0.0 = deterministic, 1.0 = creative
  max_tokens: 512         # Max tokens in each response
```

### Prompt settings

```yaml
fusion:
  mode: lean              # lean = minimal overhead, structured = more context sections
  max_prompt_tokens: 2048 # Total token budget for the prompt
```

### Memory settings

```yaml
memory:
  short_term:
    max_turns: 30         # How many conversation turns to keep
    max_tokens: 2048      # Token budget for conversation history
```

### Project context settings

```yaml
project_context:
  enabled: true
  top_k: 5                    # Number of code snippets to retrieve per query
  similarity_threshold: 0.35  # Minimum relevance score (0-1)
  max_chunk_lines: 40         # Max lines per indexed chunk
  max_file_size_kb: 500       # Skip files larger than this
```

---

## 12. Troubleshooting

### "No models found"

Download a model first:
```bash
python download_model.py
```

### Server won't start / port in use

Another instance might be running. Kill it or use a different port:
```bash
python launch.py --port 9000
```

### Slow generation

- **No GPU:** Set `gpu_layers: 0` in config.yaml and increase `threads` to match your CPU cores
- **With GPU:** Make sure `gpu_layers: 99` is set. Check that llama-cpp-python was installed with CUDA support:
  ```bash
  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --force-reinstall
  ```

### "sentence-transformers" download on every startup

The embedding model caches in `~/.cache/huggingface`. If this directory is missing or cleared, it re-downloads (~90MB). After the first download, the app works fully offline.

### Wrong language in output

Be explicit about the language in your prompt. Instead of "write a function to sort a list", say "write a **Go** function to sort a slice". The augmentor system uses language keywords to route to the right examples.

### Desktop window doesn't open

pywebview might not be installed:
```bash
pip install pywebview
```

Or use browser mode:
```bash
python launch.py --browser
```

### Out of memory

- Switch to a smaller model (Qwen 0.5B is 469MB)
- Reduce `context_length` in config.yaml (try 2048)
- Set `gpu_layers: 0` to keep the model on CPU (slower but uses less VRAM)

### Model generates gibberish

- Lower the temperature in config.yaml (try `temperature: 0.1`)
- Make sure you're using a Q4_K_M quantized model (the default)
- Try a different model -- some models work better than others for code

---

## Getting Help

- GitHub Issues: https://github.com/densanon-devs/ultralight-coder/issues
- API Docs: http://localhost:8000/docs (when server is running)
- Commands: Type `/help` in terminal mode
