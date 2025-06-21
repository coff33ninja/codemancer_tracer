import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
)
def summarize_with_openai(md_summary: str, model: str, api_key: str) -> str:
    """Sends codebase summary to OpenAI for analysis."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        print(f"[*] Using OpenAI {model} for summary...")
        prompt = (
            "You are a senior Python code auditor with expertise in performance optimization and code quality. "
            "Analyze the following codebase summary and provide a detailed report. Focus on:\n"
            "- Identifying potential dead code (unused functions or imports)\n"
            "- Detecting redundant code patterns\n"
            "- Highlighting performance bottlenecks\n"
            "- Suggesting improvements for code structure and maintainability\n"
            "- Providing specific, actionable recommendations\n\n"
            "Codebase summary:\n"
            f"{md_summary}\n\n"
            "Return your analysis in a well-structured Markdown format with clear sections."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a senior Python code auditor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[!] OpenAI API error: {e}")
        raise