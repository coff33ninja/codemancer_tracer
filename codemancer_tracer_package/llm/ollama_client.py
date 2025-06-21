import subprocess
import time

SUPPORTED_OLLAMA_MODELS = ["mistral", "llama3", "phi3", "deepseek-coder"]

def start_ollama_daemon():
    """Starts the Ollama daemon if not running."""
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        print("[*] Ollama daemon is running.")
        return True
    except subprocess.CalledProcessError:
        print("[*] Starting Ollama daemon...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
            print("[*] Ollama daemon started successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to start Ollama daemon: {e}")
            return False
    except FileNotFoundError:
        print("[!] Ollama command not found. Please ensure Ollama is installed and in your PATH.")
        return False

def summarize_with_ollama(md_summary: str, model: str) -> str:
    """Sends codebase summary to Ollama for analysis."""
    print(f"[*] Using Ollama with {model}...")
    try:
        if not start_ollama_daemon():
            return "[!] Ollama daemon could not be started or is not installed."

        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if model not in result.stdout:
            print(f"[*] Pulling {model} model for Ollama...")
            subprocess.run(["ollama", "pull", model], check=True, capture_output=True, text=True)

        prompt = (
            "You are an expert Python auditor.\n"
            "Review this code summary and highlight dead code, redundancy, and performance issues:\n\n"
            f"{md_summary}\n\n"
            "Return your analysis in Markdown format with clear sections for issues and recommendations."
        )

        ollama_result = subprocess.run(
            ["ollama", "run", model], # Removed shell=True
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        return ollama_result.stdout
    except FileNotFoundError:
        return "[!] Ollama not found. Please install Ollama or set OPENAI_API_KEY."
    except subprocess.CalledProcessError as e:
        error_output = e.stderr or e.stdout or f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        return f"[!] Failed to run or pull model {model} with Ollama: {error_output.strip()}"
