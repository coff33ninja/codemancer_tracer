import logging
from typing import Optional # New import for Optional

logger = logging.getLogger(__name__)

def process_with_transformer(md_output: str, model_name: str, task: str = "summarization", custom_prompt_template: Optional[str] = None, **kwargs) -> str:
    """
    Processes the provided markdown output using a HuggingFace transformer model
    for a specified task (e.g., summarization, text-generation).
    """
    try:
        # Lazy imports for heavy dependencies (torch and transformers)
        try:
            from transformers.pipelines import pipeline
            from transformers.trainer_utils import set_seed
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Missing required dependencies for HuggingFace Transformers. "
                f"Please install them using 'python -m codemancer_tracer_package.main --install-deps'. "
                f"Original error: {e}"
            ) from e

        # Check for GPU availability and set the device accordingly.
        if torch.cuda.is_available():
            device = 0  # Use the first available GPU
            logger.info("CUDA is available. Using GPU for transformer pipeline.")
        else:
            device = -1  # Use CPU
            logger.info("CUDA not available. Using CPU for transformer pipeline.")

        set_seed(42) # For reproducibility, especially important for text generation
        pipe = pipeline(task, model=model_name, device=device)

        if task == "summarization":
            tokenizer = pipe.tokenizer
            if tokenizer is None: # Added check for tokenizer availability
                raise ValueError(f"Model '{model_name}' does not have a tokenizer for the '{task}' task.")

            # Get the model's actual maximum input length from its configuration
            # For encoder-decoder models like BART, this is typically `max_position_embeddings`.
            # Fallback to a common default if not found.
            model_max_input_tokens = getattr(pipe.model.config, 'max_position_embeddings', 1024)
            # A practical cap to prevent excessively large chunks if model_max_length is huge
            # Summarization models typically have max_length around 512-1024, but some can be larger.
            # Capping at 4096 tokens is a reasonable balance for most summarization tasks.
            practical_max_length = min(model_max_input_tokens, 4096) # Cap at 4096 tokens for summarization

            # Define overlap in tokens for contextual continuity
            overlap_tokens = 128
            if practical_max_length <= overlap_tokens: # Prevent issues if practical_max_length is very small
                overlap_tokens = practical_max_length // 2

            summaries = []

            # Tokenize the input with truncation and return overflowing tokens for robust chunking.
            # This is the recommended way to handle long inputs for models in HuggingFace.
            # `padding="max_length"` ensures consistent input shape for the pipeline.
            tokenized_inputs = tokenizer(
                md_output,
                max_length=practical_max_length,
                truncation=True,
                return_overflowing_tokens=True,
                stride=overlap_tokens,
                padding="max_length", # Pad to max_length for consistent input
                return_tensors="pt"
            )

            # The `input_ids` from `tokenized_inputs` will now contain all chunks.
            # Each chunk is already tokenized and truncated/padded to `practical_max_length`.
            chunks_input_ids = tokenized_inputs.input_ids
            
            # If there's more than one chunk, it means the original input was long and needed splitting.
            if len(chunks_input_ids) > 1:
                logger.warning(f"Input text required chunking. Original token length was estimated to be > {practical_max_length}. Processing {len(chunks_input_ids)} chunks.")

            for i, chunk_ids in enumerate(chunks_input_ids):
                # Decode the chunk back to string for the pipeline
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
                logger.info(f"Summarizing token chunk starting at index {i} (length: {len(chunk_ids)} tokens)...")
                result = pipe(
                    chunk_text,
                    max_length=max(50, len(chunk_ids) // 4),
                    min_length=max(20, len(chunk_ids) // 8),
                    do_sample=False,
                    **kwargs
                )
                # Safely extract summary text
                summary_text = None
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    item = result[0]
                    if isinstance(item, dict):
                        summary_text = item.get('summary_text', '')
                    else:
                        summary_text = str(item)
                elif result is not None:
                    summary_text = str(result)
                else:
                    summary_text = ''
                summaries.append(summary_text)

            # Combine chunk summaries
            combined_summary = "\n\n".join(summaries)
            # If the combined summary is still too long, summarize the summary of summaries
            combined_summary_tokens = tokenizer(
                combined_summary,
                truncation=True,
                max_length=practical_max_length,
                return_tensors="pt"
            ).input_ids[0]

            if len(combined_summary_tokens) > practical_max_length:
                logger.info("Summaries of chunks are still too long. Summarizing the combined summary.")
                result = pipe(
                    combined_summary,
                    max_length=max(256, len(combined_summary_tokens) // 4),
                    min_length=max(100, len(combined_summary_tokens) // 8),
                    do_sample=False,
                    **kwargs
                )
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    item = result[0]
                    if isinstance(item, dict):
                        final_summary = item.get('summary_text', '')
                    else:
                        final_summary = str(item)
                elif result is not None:
                    final_summary = str(result)
                else:
                    final_summary = ''
            else:
                final_summary = combined_summary

            return final_summary.strip()

        elif task == "text-generation":
            # For text generation, we typically provide a single prompt.
            # The `md_output` is the codebase summary. We want to generate an analysis from it.
            # Use custom prompt if provided, otherwise fall back to default.
            if custom_prompt_template:
                prompt_template = custom_prompt_template
            else:
                prompt_template = (
                    "Based on the following Python codebase summary, provide a detailed "
                    "analysis focusing on architecture, potential improvements, and key functionalities:\n\n"
                    "{codebase_summary}"
                )
            full_prompt = prompt_template.format(codebase_summary=md_output)

            # Default generation parameters, can be overridden by kwargs
            gen_kwargs = {
                "max_new_tokens": kwargs.get('max_new_tokens', 512), # Max tokens to generate
                "do_sample": kwargs.get('do_sample', True), # Allow sampling for more creative generation
                "temperature": kwargs.get('temperature', 0.7), # Control randomness
                "top_k": kwargs.get('top_k', 50), # Top-k sampling
                "top_p": kwargs.get('top_p', 0.95), # Nucleus sampling
                "repetition_penalty": kwargs.get('repetition_penalty', 1.2),
                **kwargs # Allow overriding
            }

            logger.info(f"Generating text with model {model_name} (task: {task}, max_new_tokens: {gen_kwargs['max_new_tokens']})...")
            result = pipe(full_prompt, **gen_kwargs)

            generated_text = None
            if isinstance(result, (list, tuple)) and len(result) > 0:
                item = result[0]
                if isinstance(item, dict):
                    generated_text = item.get('generated_text', '')
                else:
                    generated_text = str(item)
            elif result is not None:
                generated_text = str(result)
            else:
                generated_text = ''

            # The generated text usually includes the prompt. Remove it.
            if generated_text.startswith(full_prompt):
                generated_text = generated_text[len(full_prompt):].strip()

            return generated_text.strip()

        else:
            # Generic handling for other tasks. This might need more specific logic
            # depending on the task's expected input/output format.
            logger.info(f"Running generic transformer pipeline for task: {task} with model: {model_name}...")
            result = pipe(md_output, **kwargs)

            if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], dict):
                # Try common keys for generated text, or fall back to string representation
                generated_text = result[0].get('generated_text') or \
                                 result[0].get('summary_text') or \
                                 result[0].get('label') or \
                                 str(result[0])
            else:
                generated_text = str(result)

            return generated_text.strip()

    except Exception as e:
        logger.error(f"Failed to process with transformer model {model_name} for task {task}: {e}")
        raise
