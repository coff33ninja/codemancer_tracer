import argparse
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from .utils.dependency_manager import install_dependencies

def run_analysis():
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # --- New: Handle --install-deps as a standalone command ---
    # Create a minimal parser to check for --install-deps
    # This parser will only look for --install-deps and ignore others for now
    install_parser = argparse.ArgumentParser(add_help=False)
    install_parser.add_argument("--install-deps", action="store_true", 
                                help="Install all required Python dependencies and exit. Must be the first argument.")
    install_parser.add_argument("--gpu", action="store_true", help="Install GPU-enabled torch/torchvision/torchaudio (CUDA) if available.")
    install_parser.add_argument("--gpu-10-series", action="store_true", help="Force install for Nvidia 10 series (GTX, CUDA 12.1)")
    install_parser.add_argument("--gpu-20plus-series", action="store_true", help="Force install for Nvidia 20/30/40 series (RTX, CUDA 12.2)")
    
    # Parse all arguments to see if --install-deps and related flags are present
    install_args, _ = install_parser.parse_known_args(sys.argv[1:]) # Parse all args, not just the first

    if install_args.install_deps:
        print("[*] --install-deps flag detected as the primary command. Installing dependencies...")
        gpu_series = None
        gpu_flag = install_args.gpu
        if getattr(install_args, 'gpu_10_series', False):
            gpu_series = '10'
            gpu_flag = True
        elif getattr(install_args, 'gpu_20plus_series', False):
            gpu_series = '20+'
            gpu_flag = True
        install_dependencies(gpu=gpu_flag, gpu_series=gpu_series)
        print("[*] Dependencies installed. Exiting.")
        sys.exit(0) # Exit immediately after installing dependencies

    # All other imports must come here, before any code that uses them
    from .utils.cache_manager import get_codebase_hash, load_cached_summary, save_cached_summary
    from .core.parser import find_py_files
    from .core.grapher import build_call_graph, create_pyvis_graph
    from .llm.openai_client import summarize_with_openai
    from .llm.transformer_client import process_with_transformer
    from .llm.ollama_client import summarize_with_ollama, SUPPORTED_OLLAMA_MODELS

    parser = argparse.ArgumentParser(description="Codemancer Tracer: Python script analyzer with LLM auditing.")
    parser.add_argument("path", help="Path to Python project directory")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for LLM analysis")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama for LLM analysis")
    parser.add_argument("--use-transformer", action="store_true", help="Use a local HuggingFace transformer model for LLM analysis") # NEW ARG
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate for transformer text generation models. Only applicable with --use-transformer and --transformer-task text-generation.") # NEW ARG
    parser.add_argument("--top-k", type=int, default=50, help="Controls the number of highest probability vocabulary tokens to keep for top-k sampling. Only applicable with --use-transformer and --transformer-task text-generation.") # NEW ARG
    parser.add_argument("--top-p", type=float, default=0.95, help="Controls the cumulative probability for nucleus (top-p) sampling. Only applicable with --use-transformer and --transformer-task text-generation.") # NEW ARG
    parser.add_argument("--temperature", type=float, default=0.7, help="Controls the randomness of transformer text generation models (0.0-1.0). Only applicable with --use-transformer and --transformer-task text-generation.") # NEW ARG
    parser.add_argument("--transformer-prompt", type=str, default=None, help="Custom prompt template for transformer text generation. Use {codebase_summary} as a placeholder.") # NEW ARG
    parser.add_argument("--transformer-task", default="summarization", help="Specify the HuggingFace transformer task (e.g., 'summarization', 'text-generation'). Only applicable with --use-transformer.") # NEW ARG
    parser.add_argument("--model", default="mistral" if not OPENAI_KEY else DEFAULT_OPENAI_MODEL,
                        help="LLM model to use (e.g., gpt-4o, mistral, llama3, phi3, deepseek-coder)")
    parser.add_argument("--view-map-only", action="store_true", help="Only generate the function map HTML, skip AI analysis.")
    parser.add_argument("--json", action="store_true", help="Output summaries in JSON format")
    args = parser.parse_args()

    # --- Start of LLM Selection and Argument Validation Refinement ---
    # Change default for --model to None, and handle defaults based on chosen LLM
    # This line is effectively changed by the new logic below, but if it were a direct change:
    # parser.add_argument("--model", default=None, ...)
    # The current default is kept for now, but its effect is overridden by the new logic.
    folder = Path(args.path)
    if not folder.exists():
        print(f"[!] Error: The specified path '{folder}' does not exist.")
        return
    if not folder.is_dir():
        print(f"[!] Error: The specified path '{folder}' is not a directory.")
        return

    # Determine which LLM to use
    selected_llms_flags = [args.use_openai, args.use_ollama, args.use_transformer]
    num_selected_llms = sum(selected_llms_flags)

    if num_selected_llms > 1:
        print("[!] Please select only one LLM backend (--use-openai, --use-ollama, or --use-transformer).")
        sys.exit(1)

    use_openai_llm = False
    use_ollama_llm = False
    use_transformer_llm = False # NEW VAR

    if args.use_openai:
        use_openai_llm = True
    elif args.use_ollama:
        use_ollama_llm = True
    elif args.use_transformer: # NEW ASSIGNMENT
        use_transformer_llm = True
    else:
        # No LLM explicitly selected, try to infer default
        if OPENAI_KEY:
            use_openai_llm = True
        else:
            use_ollama_llm = True

    # Handle model defaults and requirements based on selected LLM
    if use_openai_llm:
        if not args.model:
            args.model = DEFAULT_OPENAI_MODEL
        if not OPENAI_KEY:
            print("[!] OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            sys.exit(1)
    elif use_ollama_llm:
        if not args.model:
            args.model = "mistral" # Default Ollama model
        if args.model not in SUPPORTED_OLLAMA_MODELS:
            print(f"[!] Model {args.model} not supported by Ollama. Supported models: {', '.join(SUPPORTED_OLLAMA_MODELS)}")
            sys.exit(1)
    elif use_transformer_llm: # NEW TRANSFORMER MODEL CHECK
        if not args.model:
            print("[!] Please specify a HuggingFace transformer model using --model (e.g., 'sshleifer/distilbart-cnn-12-6').")
            sys.exit(1)
    # --- End of LLM Selection and Argument Validation Refinement ---

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
            if use_openai_llm: # OPENAI_KEY check is now done earlier
                try:
                    ai_summary = summarize_with_openai(md_output, args.model, OPENAI_KEY) # type: ignore
                    save_cached_summary(codebase_hash, ai_summary)
                except Exception:
                    print("[*] Falling back to Ollama due to OpenAI failure...")
                    ai_summary = summarize_with_ollama(md_output, args.model)
                    save_cached_summary(codebase_hash, ai_summary)
            elif use_ollama_llm: # Existing Ollama logic
                ai_summary = summarize_with_ollama(md_output, args.model)
                save_cached_summary(codebase_hash, ai_summary)
            elif use_transformer_llm: # NEW TRANSFORMER LOGIC
                print(f"[*] Using HuggingFace transformer model: {args.model} for task: {args.transformer_task}...")
                try:
                    ai_summary = process_with_transformer(md_output, args.model, task=args.transformer_task, custom_prompt_template=args.transformer_prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p) # Pass task and custom prompt
                    save_cached_summary(codebase_hash, ai_summary)
                except Exception as e:
                    print(f"[!] Error using transformer model {args.model}: {e}")
                    ai_summary = f"[!] AI analysis failed with transformer model {args.model}: {e}"

            ai_summary_json = {"summary": ai_summary}

        with open("output/ai_summary.md", "w", encoding="utf-8") as f:
            f.write("# AI-Powered Code Analysis\n\n" + ai_summary)
        if args.json:
            with open("output/ai_summary.json", "w", encoding="utf-8") as f:
                json.dump(ai_summary_json, f, indent=2)
            print("[+] AI summary saved as output/ai_summary.json")

if __name__ == "__main__":
    run_analysis()
