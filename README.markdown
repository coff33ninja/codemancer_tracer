# Codemancer Tracer

🧙 **Codemancer Tracer** is a Python-powered code analysis tool that visualizes your scripts, maps imports and function calls, and optionally uses AI to uncover dead code, redundancy, and performance bottlenecks.

---

## 🚀 Features

- 📁 Analyze all Python files in a folder
- 🔍 Extract function signatures, imports, and call graphs
- 🌐 Generate interactive visual maps with [Pyvis](https://pyvis.readthedocs.io/)
- 🧠 LLM-powered code auditing via:
  - OpenAI (`gpt-4o` or other configured model, if `OPENAI_API_KEY` is set)
  - Ollama with models like `mistral`, `llama3`, `phi3`, or `deepseek-coder`
- 📄 Outputs Markdown and JSON summaries + AI insights
- 💾 Caches AI summaries in `.llm_cache/` to skip recomputing for unchanged code
- 🛠️ Automatically installs Python dependencies from `requirements.txt`
- 🚀 Auto-starts Ollama daemon if not running
- 🔐 Supports `.env` file for API key configuration

---

## 🛠️ Requirements

The script automatically installs required Python packages from `requirements.txt` on first run:

- `prettytable`
- `networkx`
- `pyvis`
- `openai`
- `requests`
- `tenacity`
- `python-dotenv`

Create a `.env` file in the project directory to configure API keys (optional):

```plaintext
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
```

If using **Ollama**:

```bash
# Install Ollama (script auto-starts the daemon if not running):
curl https://ollama.ai/install.sh | sh

# Optionally pull supported models manually (auto-pulls if needed):
ollama pull mistral
ollama pull llama3
ollama pull phi3
ollama pull deepseek-coder
```

---

## 📦 Usage

Run the script from the `codemancer_tracer_package` directory:

```bash
python -m codemancer_tracer_package.main /path/to/python/project [options]
```

### Options:

- `--use-openai`: Use OpenAI for LLM analysis (requires `OPENAI_API_KEY`)
- `--use-ollama`: Use Ollama for LLM analysis
- `--model <model_name>`: Specify LLM model (e.g., `gpt-4o`, `mistral`, `llama3`, `phi3`, `deepseek-coder`)
- `--json`: Output summaries in JSON format (`output/function_summary.json`, `output/ai_summary.json`)

### Output:

- `output/function_summary.md` — Parsed functions and imports (Markdown)
- `output/function_summary.json` — Parsed functions and imports (JSON, if `--json` is used)
- `output/ai_summary.md` — AI-enhanced code analysis (Markdown)
- `output/ai_summary.json` — AI-enhanced code analysis (JSON, if `--json` is used)
- `function_map.html` — Interactive visual graph (open in browser)
- `.llm_cache/` — Cached AI summaries

---

## ⚡ Dependency Installation (with or without venv)

Before running any analysis, install all dependencies. The script supports both CPU and GPU (CUDA) environments and can auto-install the correct PyTorch wheels for your GPU.

### With venv (recommended on Windows):

```powershell
# Create and activate a virtual environment (if not already)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies for Nvidia 10 series (CUDA 12.1):
.\.venv\Scripts\python.exe -m codemancer_tracer_package.main --install-deps --gpu-10-series --path E:\SCRIPTS\voice_assistant

# For Nvidia 20/30/40 series (CUDA 12.2):
.\.venv\Scripts\python.exe -m codemancer_tracer_package.main --install-deps --gpu-20plus-series --path E:\SCRIPTS\voice_assistant

# For CPU-only (no GPU):
.\.venv\Scripts\python.exe -m codemancer_tracer_package.main --install-deps --path E:\SCRIPTS\voice_assistant
```

### Without venv (system Python):

```powershell
# Install dependencies for Nvidia 10 series (CUDA 12.1):
python -m codemancer_tracer_package.main --install-deps --gpu-10-series --path E:\SCRIPTS\voice_assistant

# For Nvidia 20/30/40 series (CUDA 12.2):
python -m codemancer_tracer_package.main --install-deps --gpu-20plus-series --path E:\SCRIPTS\voice_assistant

# For CPU-only (no GPU):
python -m codemancer_tracer_package.main --install-deps --path E:\SCRIPTS\voice_assistant
```

---

## 🧠 LLM Behavior

The script selects the LLM backend based on:

1. Command-line flags (`--use-openai` or `--use-ollama`)
2. If no flags, uses OpenAI if `OPENAI_API_KEY` is set, else falls back to Ollama
3. If both are unavailable, skips AI analysis with a warning

Set environment variables in `.env` or system environment:
- `OPENAI_API_KEY`: Required for OpenAI
- `OPENAI_MODEL`: OpenAI model (defaults to `gpt-4o`)

Supported Ollama models: `mistral`, `llama3`, `phi3`, `deepseek-coder`

---

## 📍 Example

```bash
python -m codemancer_tracer_package.main ./my_flask_app --use-ollama --model llama3 --json
```

Opens `function_map.html`:

```
main.route_home() --> render_template
api.get_data() --> requests.get
```

Generates JSON files if `--json` is used:

```json
{
  "module_name": {
    "functions": ["function_name(arg1, arg2)"],
    "imports": ["module1", "module2"],
    "calls": ["function_call"]
  }
}
```

---

## 📦 PyPI Installation

Soon available via:

```bash
pip install codemancer-tracer
```

Then run globally:

```bash
codemancer-trace ./your_codebase [options]
```

---

## 💡 Future Ideas

- Integration with VS Code or TUI interface
- GraphML or Mermaid.js export for CI/CD pipelines
- Auto-detect and analyze popular frameworks (Flask, Django, FastAPI)
- Support for xAI's Grok API (when available)

---

## ✨ Credits

Crafted by wizards who see code as spells.

---

## 📜 License

MIT — use it, share it, fork it, enchant it.

---

## GPU Support (NVIDIA GTX 1070, Python 3.11)

This project supports GPU acceleration for transformer models using PyTorch and CUDA.

### Requirements
- NVIDIA GTX 1070 (CUDA Compute Capability 6.1)
- CUDA Toolkit 12.x (or latest supported by your driver)
- Python 3.11
- PyTorch with CUDA support (see below)

### Installation
To install PyTorch with CUDA 12.x for Python 3.11:

```sh
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

See the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page for the latest instructions and compatibility.

### Verifying GPU Support
The codebase includes a helper (`ensure_gpu_ready`) in `transformer_client.py` to log CUDA device info and verify GPU availability.

If CUDA is not available, computation will fall back to CPU automatically.