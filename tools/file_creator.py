import os
import sys
import json
import requests
# import logging # Заменяем на print в stderr
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# logger = logging.getLogger(__name__)

# Constants
API_TIMEOUT_SECONDS = 120
MAX_OUTPUT_TOKENS_REWRITE = 8192

# --- Helper function to clean Gemini response ---
def clean_gemini_response(content: str, verbose: bool = True) -> str:
    """Removes markdown code fences if they wrap the entire response."""
    if content and content.startswith("```") and content.endswith("```"):
        lines = content.splitlines()
        if len(lines) > 1:
            first_line_content = lines[0][3:].strip()
            if (
                first_line_content
                and " " not in first_line_content
                and len(first_line_content) < 15
            ):
                cleaned = "\n".join(lines[1:-1])
            else:
                cleaned = "\n".join(lines[:-1])
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]

            if cleaned.strip():
                if verbose:
                    print("FILE_CREATOR: Removed Markdown code fences from rewrite.", file=sys.stderr)
                return cleaned
    return content

# --- Gemini API Call for Content Rewrite ---
def call_gemini_for_content_rewrite(
    content_to_rewrite: str, api_key: str, model_name: str, api_base_url: str, verbose: bool = True
) -> Optional[str]:
    """Calls the specified Gemini model to clean/rewrite content."""
    if verbose:
        print(f"FILE_CREATOR: Calling Gemini model '{model_name}' for content rewrite.", file=sys.stderr)
    if not api_key:
        if verbose:
            print("FILE_CREATOR: Error - GEMINI_API_KEY was not provided.", file=sys.stderr)
        return None
    if not model_name:
        if verbose:
            print("FILE_CREATOR: Error - Gemini model name was not provided.", file=sys.stderr)
        return None
    if not api_base_url:
         if verbose:
            print("FILE_CREATOR: Error - Gemini api_base_url was not provided.", file=sys.stderr)
         return None

    rewrite_system_prompt = (
        "You are an expert code editor/formatter. You will receive text content intended for a new file. "
        "Your task is to review this content, correct any potential JSON escaping issues or markdown artifacts, "
        "ensure it's well-formatted, and return ONLY the final, clean content suitable for direct saving to a file. "
        "Do not add any explanations, comments, or markdown code fences around the final content."
    )
    combined_user_prompt = f"{rewrite_system_prompt}\n\nPlease review and clean the following content for a new file:\n\n```\n{content_to_rewrite}\n```"

    headers = {"Content-Type": "application/json"}
    gemini_url = f"{api_base_url}/{model_name}:generateContent?key={api_key}"

    request_data = {
        "contents": [{"role": "user", "parts": [{"text": combined_user_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": MAX_OUTPUT_TOKENS_REWRITE,
        },
        # "system_instruction": ... # Не используется здесь, т.к. промпт в user content
    }

    if verbose:
        print(f"FILE_CREATOR: Sending rewrite request to {model_name}...", file=sys.stderr)
    try:
        response = requests.post(
            gemini_url, headers=headers, json=request_data, timeout=API_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        data = response.json()
        # if verbose: print(f"FILE_CREATOR DEBUG: Rewrite Response: {json.dumps(data)[:200]}...", file=sys.stderr)

        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if verbose:
                print(f"FILE_CREATOR: Rewrite finished with reason: {finish_reason}", file=sys.stderr)

            if (
                "content" in candidate
                and "parts" in candidate["content"]
                and candidate["content"]["parts"]
            ):
                rewritten_content = candidate["content"]["parts"][0]["text"].strip()
                if rewritten_content:
                    if verbose:
                        print("FILE_CREATOR: Content successfully rewritten.", file=sys.stderr)
                    return clean_gemini_response(rewritten_content, verbose=verbose)
                else:
                    if verbose:
                        print("FILE_CREATOR: Warning - Gemini rewrite response was empty.", file=sys.stderr)
                    return content_to_rewrite
            elif finish_reason != "STOP":
                if verbose:
                    print(f"FILE_CREATOR: Error - Gemini rewrite finished with reason '{finish_reason}' but no content.", file=sys.stderr)
                return None
            else:
                if verbose:
                    print("FILE_CREATOR: Warning - Gemini rewrite API stopped but no content found (empty parts).", file=sys.stderr)
                return None
        else:
            if verbose:
                print("FILE_CREATOR: Warning - No candidates found in Gemini rewrite response.", file=sys.stderr)
            return None

    except requests.exceptions.HTTPError as e:
        if verbose:
            print(f"FILE_CREATOR: Error - HTTP error calling rewrite API ({model_name}): {e}", file=sys.stderr)
            if e.response is not None:
                print(f"FILE_CREATOR: Response text: {e.response.text}", file=sys.stderr)
        return None
    except Exception as e:
        if verbose:
            print(f"FILE_CREATOR: Error - Unexpected error calling rewrite API ({model_name}): {e}", file=sys.stderr)
        return None

# --- Main Tool Backend Function ---
def handle_create_file(
    target_file: str,
    content: str,
    project_root: Path, # Должен быть CWD из gemini.py
    api_key: Optional[str] = None,
    rewrite_model_name: Optional[str] = None, # Имя модели для рерайта
    api_base_url: Optional[str] = None, # URL для рерайта
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Handles the create_file tool request.
    Optionally rewrites content using Gemini and saves the file relative to project_root.

    Returns:
        Tuple (success: bool, message: str)
    """
    if not target_file or content is None:
        msg = f"Error: Invalid arguments for create_file. Missing 'target_file' or 'content'. Target: '{target_file}', Content Provided: {content is not None}"
        if verbose:
            print(f"FILE_CREATOR: {msg}", file=sys.stderr)
        return False, msg

    if verbose:
        print(f"FILE_CREATOR: Handling request. Target='{target_file}', Root='{project_root}', Content Len={len(content)}", file=sys.stderr)

    # --- Content Rewrite Step ---
    final_content = content
    if api_key and rewrite_model_name and api_base_url:
        if verbose:
            print(f"FILE_CREATOR: Rewriting content ({len(content)} bytes) using {rewrite_model_name}...", file=sys.stderr)
        rewritten = call_gemini_for_content_rewrite(content, api_key, rewrite_model_name, api_base_url, verbose=verbose)
        if rewritten is None:
            error_msg = f"Error: Failed to rewrite content for '{target_file}'. File not created/modified."
            return False, error_msg
        final_content = rewritten
    else:
        if verbose:
            print("FILE_CREATOR: Skipping content rewrite (API key, model name, or base URL missing).", file=sys.stderr)
    # --- End Content Rewrite ---

    # --- Path Resolution and Validation ---
    target_path = Path(target_file)
    if target_path.is_absolute():
        try:
            relative_path = target_path.relative_to(project_root)
            full_path = project_root / relative_path
            if verbose:
                print(f"FILE_CREATOR: Warning - Absolute path '{target_path}' interpreted as '{full_path}' relative to project root '{project_root}'.", file=sys.stderr)
        except ValueError:
            error_msg = f"Error: Absolute path '{target_path}' is outside the project root '{project_root}'. Disallowed."
            return False, error_msg
    else:
        full_path = project_root / target_file

    full_path = full_path.resolve()

    if not full_path.is_relative_to(project_root) and full_path != project_root :
        is_safe = False
        try:
            full_path.relative_to(project_root)
            is_safe = True
        except ValueError:
            is_safe = False
        
        if not is_safe:
            error_msg = f"Error: Resolved path '{full_path}' is outside the project root '{project_root}'. Disallowed."
            return False, error_msg
    # --- End Path Resolution ---

    if verbose:
        print(f"FILE_CREATOR: Attempting to write file: {full_path}", file=sys.stderr)

    # --- Directory Creation ---
    dir_path = full_path.parent
    if dir_path:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"FILE_CREATOR: Ensured directory exists: {dir_path}", file=sys.stderr)
        except Exception as e:
            error_msg = f"Error creating directories '{dir_path}' for '{target_file}': {e}"
            return False, error_msg
    # --- End Directory Creation ---

    # --- File Writing ---
    try:
        file_exists = full_path.exists()
        if file_exists:
            # Do not overwrite. Return a specific message that gemini.py can parse.
            # The actual path of the existing file is in full_path, but target_file is the name model used.
            error_msg = f"FILE_ALREADY_EXISTS:{target_file}" # Special prefix for gemini.py
            if verbose:
                print(f"FILE_CREATOR: Attempt to create '{target_file}' failed because it already exists at '{full_path}'.", file=sys.stderr)
            return False, error_msg

        # If file does not exist, proceed to create
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        action = "created" # It will only be "created" now, as we don't overwrite
        final_output = f"File '{target_file}' successfully {action}. (Path: {full_path})"
        if verbose:
            print(f"FILE_CREATOR: {final_output}", file=sys.stderr)
        return True, final_output
    except Exception as e:
        error_msg = f"Error writing file '{target_file}' to '{full_path}': {e}"
        return False, error_msg
    # --- End File Writing ---

# Keep main for potential standalone testing
# ... (example usage remains the same, but ensure necessary env vars are set) 