import argparse
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from .utils.dependency_manager import install_dependencies
from .utils.cache_manager import get_codebase_hash, load_cached_summary, save_cached_summary
from .core.parser import find_py_files
from .core.grapher import build_call_graph, create_pyvis_graph
from .llm.openai_client import summarize_with_openai
from .llm.ollama_client import summarize_with_ollama, SUPPORTED_OLLAMA_MODELS

def run_analysis():
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    parser = argparse.ArgumentParser(description="Codemancer Tracer: Python script analyzer with LLM auditing.")
    parser.add_argument("path", help="Path to Python project directory")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for LLM analysis")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama for LLM analysis")
    parser.add_argument("--model", default="mistral" if not OPENAI_KEY else DEFAULT_OPENAI_MODEL,
                        help="LLM model to use (e.g., gpt-4o, mistral, llama3, phi3, deepseek-coder)")
    parser.add_argument("--view-map-only", action="store_true", help="Only generate the function map HTML, skip AI analysis.")
    parser.add_argument("--json", action="store_true", help="Output summaries in JSON format")
    args = parser.parse_args()

    install_dependencies()

    folder = Path(args.path)
    if not folder.exists() or not folder.is_dir():
        print(f"Invalid folder: {folder}")
        return

    use_openai_llm = args.use_openai
    use_ollama_llm = args.use_ollama

    if not use_openai_llm and not use_ollama_llm:
        if OPENAI_KEY:
            use_openai_llm = True
        else:
            use_ollama_llm = True

    if use_ollama_llm and args.model not in SUPPORTED_OLLAMA_MODELS:
        print(f"[!] Model {args.model} not supported by Ollama. Supported models: {', '.join(SUPPORTED_OLLAMA_MODELS)}")
        return

    Path("output").mkdir(exist_ok=True)

    files = find_py_files(folder)
    print(f"[+] Found {len(files)} Python files.")
    G, md_output, func_map, json_output = build_call_graph(files)

    with open("output/function_summary.md", "w", encoding="utf-8") as f:
        f.write("# Function and Import Summary\n\n" + md_output)
    if args.json:
        with open("output/function_summary.json", "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)
        print("[+] Function summary saved as output/function_summary.json")

    # Generate the interactive graph before potentially slow AI analysis
    create_pyvis_graph(G, func_map)

    if args.view_map_only:
        print("[+] Skipping AI analysis as requested (--view-map-only).")
        return # Exit the function after generating the map

    else:
        ai_summary = "[!] AI analysis skipped: No valid LLM backend configured or issue occurred."
        ai_summary_json = {"summary": ai_summary}

        codebase_hash = get_codebase_hash(files)
        cached_summary = load_cached_summary(codebase_hash)

        if cached_summary:
            print("[*] Using cached AI summary.")
            ai_summary = cached_summary
            ai_summary_json = {"summary": cached_summary}
        else:
            if use_openai_llm and OPENAI_KEY:
                try:
                    ai_summary = summarize_with_openai(md_output, args.model, OPENAI_KEY)
                    save_cached_summary(codebase_hash, ai_summary)
                except Exception:
                    print("[*] Falling back to Ollama due to OpenAI failure...")
                    ai_summary = summarize_with_ollama(md_output, args.model)
                    save_cached_summary(codebase_hash, ai_summary)
            elif use_ollama_llm:
                ai_summary = summarize_with_ollama(md_output, args.model)
                save_cached_summary(codebase_hash, ai_summary)
            ai_summary_json = {"summary": ai_summary}

        with open("output/ai_summary.md", "w", encoding="utf-8") as f:
            f.write("# AI-Powered Code Analysis\n\n" + ai_summary)
        if args.json:
            with open("output/ai_summary.json", "w", encoding="utf-8") as f:
                json.dump(ai_summary_json, f, indent=2)
            print("[+] AI summary saved as output/ai_summary.json")

if __name__ == "__main__":
    run_analysis()
