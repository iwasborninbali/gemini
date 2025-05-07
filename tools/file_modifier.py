#!/usr/bin/env python3
import os
import sys
import requests
import json
import re
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
import difflib

logger = logging.getLogger(__name__)

# --- Configuration ---
REWRITER_MAX_TOKENS = 100000
VERIFIER_MAX_TOKENS = 100000
MAX_RETRIES = 3
API_TIMEOUT_SECONDS = 300
# --- End Configuration ---

# --- System Prompts (Keep as is) ---
REWRITER_SYSTEM_PROMPT = """
You are an AI assistant that rewrites entire files based on instructions.
The user will provide:
1. The full original content of a file.
2. A high-level description of the desired changes (e.g., "Add a new parameter to the function X", "Refactor the loop in Y to use a list comprehension").
3. One or more specific code snippets illustrating the exact changes to be made. These snippets will use comments like '// ... existing code ...' or '# ... existing code ...' to denote unchanged parts of the file around the modifications.
4. [Optional] Feedback from a previous verification step if the last attempt failed.

Your task is to:
1. Understand the overall goal from the "Desired Changes Description".
2. Carefully analyze the provided "Change Snippets" and the "Original File Content".
3. Apply ONLY the specific modifications shown in the "Change Snippets" to the "Original File Content". Ensure that the context provided by '... existing code ...' comments is correctly matched.
4. The applied changes should align with the high-level "Desired Changes Description".
5. **If verification feedback is provided, pay close attention to it and ensure your new rewrite addresses the described issues.**
6. Return the ENTIRE modified file content as plain text.
7. IMPORTANT: Do NOT add any explanations, comments, or markdown formatting (like ```) around the returned code. Your output should be ONLY the raw, complete, modified file content, suitable for direct writing to a file.
8. Preserve the original formatting, indentation, and structure of the unchanged parts of the file as accurately as possible.
"""

VERIFIER_SYSTEM_PROMPT = """
You are an AI code verification assistant. Your task is to analyze a code rewrite attempt and determine if it was successful according to specific criteria.

You will receive the following inputs:
1.  **Original File Content:** The complete code before any changes.
2.  **Desired Changes Description:** A high-level textual description of what the user wants to achieve.
3.  **Change Snippets:** One or more specific code snippets illustrating the exact changes that were supposed to be made, using comments like '// ... existing code ...' or '# ... existing code ...' to denote unchanged parts.
4.  **Rewritten File Content:** The complete code produced by the rewriter.

Your evaluation criteria:
1.  **Correct Change Application:** Were the specific changes illustrated in the "Change Snippets" applied correctly and in the intended locations within the "Rewritten File Content"?
2.  **Alignment with Description:** Do the applied changes accurately reflect the intent described in the "Desired Changes Description"?
3.  **No Unintended Changes:** Were there any modifications made to the code *outside* the scope of the requested changes as shown in the "Change Snippets"? (Minor whitespace/formatting changes related to the edit are acceptable, but logic/code structure changes are not unless explicitly part of the snippets or description).
4.  **Structure Preservation:** Does the "Rewritten File Content" maintain the overall structure, logic, and comments of the "Original File Content" in the unchanged parts?
5.  **Basic Syntax:** Does the rewritten code appear syntactically valid for its likely language? (This is a basic check, not a full compilation).

Based on your analysis, you MUST return ONLY a JSON object with the following structure:

{
  "result": "success" | "failure",
  "comment": "A brief explanation ONLY if the result is 'failure', describing the specific issue(s) found based on the criteria above. If 'success', this should be an empty string or omitted."
}

Example successful output:
{
  "result": "success"
}

Example failure output:
{
  "result": "failure",
  "comment": "The requested change was applied, but an unrelated function 'calculateTotal' was unexpectedly removed."
}

Example failure output:
{
  "result": "failure",
  "comment": "The change snippet asked to add a parameter, but the rewriter modified the function body instead."
}

IMPORTANT: Respond ONLY with the JSON object. Do not add any introductory text, explanations, or markdown formatting around the JSON.
"""
# --- End System Prompts ---


# --- Helper function to clean Gemini response ---
def clean_gemini_rewriter_response(content: str, verbose: bool = True) -> str:
    """
    Strips a *single* outer wrapper (```‑фенсы, <pre><code>, <code>, <script>)
    that some LLMs add around the file body.  
    Leaves internal markup intact.
    """
    if not content:
        return content

    text = content.strip()

    # (opening‑regex, closing‑regex) pairs — проверяем по порядку
    wrappers: tuple[tuple[str, str], ...] = (
        (r"^```(?:\w+)?\\s*\n?", r"\\n?```$"),                       # ```python … ```
        (r"^<pre[^>]*>\\s*<code[^>]*>\\s*", r"\\s*</code>\\s*</pre>$"), # <pre><code> … </code></pre>
        (r"^<code[^>]*>\\s*", r"\\s*</code>$"),                       # <code> … </code>
        (r"^<script[^>]*>\\s*", r"\\s*</script>$"),                   # <script> … </script>
    )

    for open_pat, close_pat in wrappers:
        if re.match(open_pat, text, re.IGNORECASE | re.DOTALL) and \
           re.search(close_pat, text, re.IGNORECASE | re.DOTALL):
            text = re.sub(open_pat, "", text, 1, re.IGNORECASE | re.DOTALL)
            text = re.sub(close_pat, "", text, 1, re.IGNORECASE | re.DOTALL)
            if verbose:
                logger.debug(
                    f"Removed wrapper matching /{open_pat}/ … /{close_pat}/"
                )
            text = text.strip()

    # Убираем одиночные HTML‑комментарии, которые иногда вставляются LLM‑ом
    text = re.sub(r"^\\s*<!--.*?-->\\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\\s*<!--.*?-->\\s*$", "", text, flags=re.DOTALL)

    return text


# --- NEW helper ---
def strip_lonely_fences(text: str) -> str:
    """Если файл начинается/заканчивается строкой ```*, убираем её."""
    lines = text.splitlines()
    if lines and re.match(r"^```", lines[0].strip()):
        lines = lines[1:]
    if lines and re.match(r"^```", lines[-1].strip()):
        lines = lines[:-1]
    return "\n".join(lines)


# --- API Call Function ---
def call_gemini_api(
    system_prompt: str,
    user_input: str,
    api_key: str,
    model_name: str,
    max_tokens: int = REWRITER_MAX_TOKENS,
    is_json_output: bool = False,
    verbose: bool = True
) -> Optional[str | dict]:
    """Calls the specified Gemini model API."""
    if verbose:
        logger.debug(
            f"Preparing API call to {model_name} (Expects JSON: {is_json_output}) Max Tokens: {max_tokens}"
        )
    if not api_key:
        logger.error(f"API Key missing for model {model_name}.")
        return None
    if not model_name:
        logger.error("Model name missing for API call.")
        return None

    headers = {"Content-Type": "application/json"}
    api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    url = f"{api_endpoint}?key={api_key}"

    generation_config = {"temperature": 0.1, "maxOutputTokens": max_tokens}
    if is_json_output:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_input}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": generation_config,
    }

    key_identifier = f"...{api_key[-4:]}" if len(api_key) > 4 else "key"
    if verbose:
        logger.info(
            f"Attempting API call to {model_name} with key ending {key_identifier}..."
        )

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=API_TIMEOUT_SECONDS
        )
        if verbose:
            logger.debug(f"API Response Status Code: {response.status_code}")
        response.raise_for_status()
        response_data = response.json()

        # Safety/Error Checks (keep as is)
        if "promptFeedback" in response_data:
            block_reason = response_data["promptFeedback"].get("blockReason")
            if block_reason:
                logger.error(f"Prompt blocked by API. Reason: {block_reason}")
                return None
        if not response_data.get("candidates"):
            logger.error(
                "No candidates found in API response.",
                extra={"response_data": response_data},
            )
            return None

        candidate = response_data["candidates"][0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        if verbose:
            logger.info(f"API call finished with reason: {finish_reason}")

        if finish_reason not in ["STOP", "MAX_TOKENS"]:
            logger.error(
                f"API call failed or blocked. Finish Reason: {finish_reason}",
                extra={"candidate": candidate},
            )
            return None
        if not (candidate.get("content") and candidate["content"].get("parts")):
            logger.error(
                f"API finished with reason '{finish_reason}' but no content part found.",
                extra={"candidate": candidate},
            )
            return None

        # Process Content
        api_response_content = candidate["content"]["parts"][0]["text"]
        if verbose:
            logger.info(f"API call successful. Finish Reason: {finish_reason}")
        if finish_reason == "MAX_TOKENS":
            logger.warning("Response may be truncated due to MAX_TOKENS limit!")

        # Handle Expected Output Type
        if is_json_output:
            if verbose:
                logger.debug("Attempting to parse API response as JSON.")
            try:
                cleaned_json_string = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    api_response_content,
                    flags=re.MULTILINE | re.DOTALL,
                ).strip()
                if not cleaned_json_string:
                    logger.error("Verifier returned empty content after cleaning.")
                    return None
                parsed_json = json.loads(cleaned_json_string)
                if "result" not in parsed_json:
                    logger.error(
                        "Verifier JSON response missing 'result' key.",
                        extra={"parsed_json": parsed_json},
                    )
                    return None
                if verbose:
                    logger.info("Successfully parsed Verifier JSON response.")
                return parsed_json
            except json.JSONDecodeError as json_err:
                logger.error(
                    f"Failed to decode JSON response from Verifier: {json_err}",
                    extra={"raw_response": api_response_content},
                )
                return None
            except Exception as e:
                logger.error(
                    f"Unexpected error during JSON parsing: {e}", exc_info=True
                )
                return None
        else:
            # Return the raw text response for rewriter
            return api_response_content

    except requests.exceptions.HTTPError as http_err:
        logger.error(
            f"HTTP error calling {model_name}: {http_err} (Status code: {http_err.response.status_code if http_err.response else 'N/A'})"
        )
        try:
            logger.error(f"Error Response Body: {http_err.response.text}")
        except Exception:
            pass
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(
            f"Request exception occurred calling {model_name}: {req_err}", exc_info=True
        )
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the API call to {model_name}: {e}",
            exc_info=True,
        )
        return None


# --- Path and File Reading Functions (keep as is) ---
def resolve_path(relative_path: str, base_cwd: str | None) -> Path:
    target_path = Path(relative_path)
    if target_path.is_absolute():
        return target_path.resolve()
    elif base_cwd:
        base_path = Path(base_cwd)
        if base_path.is_dir():
            return (base_path / relative_path).resolve()
        else:
            logger.warning(
                f"Base CWD '{base_cwd}' не является директорией. Резолвим относительно os.getcwd()."
            )
            return (Path(os.getcwd()) / relative_path).resolve()
    else:
        logger.warning("base_cwd не указан. Резолвим относительно os.getcwd().")
        return (Path(os.getcwd()) / relative_path).resolve()


def read_original_file(file_path: Path, verbose: bool = True) -> str | None:
    logger.debug(f"Reading target file: {file_path}")
    try:
        if not file_path.is_file():
            logger.error(f"Target file not found or is not a file: {file_path}")
            return None
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        logger.info(f"Read {len(content)} characters from {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading target file {file_path}: {e}", exc_info=True)
        return None


# --- Helper function to safely open and write ---
def write_final_content(path: Path, content: str, verbose: bool = True):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:  # Still use builtins.open here
            f.write(content)
        logger.info(f"Successfully wrote {len(content)} bytes to {path}")
        return True
    except Exception as write_err:
        logger.error(
            f"Failed to write final content to {path}: {write_err}", exc_info=True
        )
        return False


# --- Main Tool Backend Function ---
def handle_apply_diff(
    target_file: str,
    change_snippets: str, # Renamed from diff_content
    desired_changes_description: str, # New parameter
    project_root: Path,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    rewriter_model_override: Optional[str] = None,
    verifier_model_override: Optional[str] = None,
    verbose: bool = True
) -> Tuple[bool, str, Optional[str], Optional[str]]:  # Return includes potential verifier comment AND diff string
    """
    Handles the apply_diff tool request using a rewrite/verify loop.

    Args:
        target_file: Relative path of the file to modify.
        change_snippets: The code snippets illustrating the changes, using '... existing code ...' syntax.
        desired_changes_description: A natural language description of the overall desired changes.
        project_root: The root directory of the project.
        api_key: Gemini API key.
        model_name: Gemini model name for rewriting/verifying (e.g., MODEL_REWRITE).

    Returns:
        Tuple (success: bool, message: str, verifier_comment: Optional[str], diff_str: Optional[str])
    """
    if verbose:
        logger.info("--- Entering handle_apply_diff (Rewriter/Verifier Logic) ---")
    logger.debug(
        f"Target File: {target_file}, Model: {model_name}, Project Root: {project_root}"
    )
    logger.debug(f"Desired Changes Description: {desired_changes_description[:200]}...")
    logger.debug(f"Change Snippets: {change_snippets[:200]}...")

    if not api_key or not model_name:
        msg = f"Error: API Key or Model Name missing for apply_diff ('{target_file}'). Cannot proceed."
        logger.error(msg)
        return False, msg, None, None

    # Resolve path relative to project_root
    target_path_obj = Path(target_file)
    if target_path_obj.is_absolute():
        try:
            relative_path = target_path_obj.relative_to(project_root)
            target_file_abs = project_root / relative_path
            logger.warning(
                f"Absolute path '{target_path_obj}' provided for apply_diff; interpreted as '{target_file_abs}'."
            )
        except ValueError:
            error_msg = f"Error: Absolute path '{target_path_obj}' is outside the project root '{project_root}'. Modification disallowed."
            logger.error(error_msg)
            return False, error_msg, None, None
    else:
        target_file_abs = (project_root / target_file).resolve()

    # Security check after resolution
    if project_root not in target_file_abs.parents and target_file_abs != project_root:
        error_msg = f"Error: Resolved path '{target_file_abs}' for apply_diff is outside the project root '{project_root}'. Modification disallowed."
        logger.error(error_msg)
        return False, error_msg, None, None

    logger.info(f"Resolved absolute path for modification: {target_file_abs}")

    original_file_content = read_original_file(target_file_abs, verbose=verbose)
    if original_file_content is None:
        # Error logged in read_original_file
        return (
            False,
            f"Error: Could not read original file {target_file} (at {target_file_abs}). Cannot apply diff.",
            None,
            None,
        )

    # --- Rewrite and Verification Loop ---
    attempt = 1
    verification_passed = False
    final_rewritten_content = None
    last_verifier_comment = "Initial attempt."

    while attempt <= MAX_RETRIES and not verification_passed:
        logger.info(
            f"--- Rewrite/Verify Attempt {attempt}/{MAX_RETRIES} for {target_file} ---"
        )

        # === Rewriter Step ===
        rewriter_prompt = (
            f"Original File Content:\n```\n{original_file_content}\n```\n\n"
            f"Desired Changes Description:\n{desired_changes_description}\n\n"
            f"Change Snippets:\n```\n{change_snippets}\n```"
            + (
                f"\n\nVerification Feedback from previous attempt:\n{last_verifier_comment}"
                if attempt > 1
                else ""
            )
        )
        logger.info("Calling Rewriter Model...")
        rewritten_content_raw = call_gemini_api(
            system_prompt=REWRITER_SYSTEM_PROMPT,
            user_input=rewriter_prompt,
            api_key=api_key,
            model_name=model_name,
            max_tokens=REWRITER_MAX_TOKENS,
            verbose=verbose
        )

        if rewritten_content_raw is None:
            logger.error(f"Rewriter API call failed on attempt {attempt}. Aborting.")
            # Use the last known verifier comment if available
            error_msg = f"Error: Rewriter API call failed for '{target_file}'. {last_verifier_comment}"
            return False, error_msg, last_verifier_comment, None

        # Clean potential markdown fences from the raw response
        rewritten_content = clean_gemini_rewriter_response(rewritten_content_raw, verbose=verbose)
        rewritten_content = strip_lonely_fences(rewritten_content)

        if not rewritten_content:
            logger.error(
                f"Rewriter returned empty content after cleaning on attempt {attempt}. Aborting."
            )
            error_msg = f"Error: Rewriter returned empty content for '{target_file}'. {last_verifier_comment}"
            return False, error_msg, last_verifier_comment, None

        final_rewritten_content = rewritten_content  # Store the latest attempt
        logger.debug(
            f"Rewriter Output (Attempt {attempt}, first 200 chars): {final_rewritten_content[:200]}..."
        )

        # === Verifier Step ===
        verifier_prompt = (
            f"Original File Content:\n```\n{original_file_content}\n```\n\n"
            f"Desired Changes Description:\n{desired_changes_description}\n\n"
            f"Change Snippets:\n```\n{change_snippets}\n```\n\n"
            f"Rewritten File Content:\n```\n{final_rewritten_content}\n```"
        )
        logger.info("Calling Verifier Model...")
        verifier_response = call_gemini_api(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_input=verifier_prompt,
            api_key=api_key,
            model_name=model_name,  # Use the same model for verification
            max_tokens=VERIFIER_MAX_TOKENS,
            is_json_output=True,
            verbose=verbose
        )

        if verifier_response is None or not isinstance(verifier_response, dict):
            logger.error(
                f"Verifier API call failed or returned invalid data on attempt {attempt}. Treating as verification failure."
            )
            last_verifier_comment = (
                "Verifier API call failed or returned non-JSON data."
            )
            # Optionally, break or continue based on policy? For now, continue to retry.
        elif verifier_response.get("result") == "success":
            logger.info(f"Verification PASSED on attempt {attempt}.")
            verification_passed = True
            last_verifier_comment = None  # Clear comment on success
        else:
            last_verifier_comment = verifier_response.get(
                "comment", "Verification failed, but no specific comment provided."
            )
            logger.warning(
                f"Verification FAILED on attempt {attempt}. Comment: {last_verifier_comment}"
            )

        if not verification_passed:
            attempt += 1
            if attempt <= MAX_RETRIES:
                logger.info(f"Pausing before retry attempt {attempt}...")
                time.sleep(1)  # Simple delay

    # --- End Rewrite/Verify Loop ---

    # --- Final Outcome ---
    if verification_passed and final_rewritten_content is not None:
        try:
            # --- Step 4: Write the validated content ---
            logger.info(f"Attempting to write verified content to: {target_file_abs}")
            write_success = write_final_content(
                target_file_abs, final_rewritten_content, verbose=verbose
            )

            if write_success:
                final_message = f"File '{target_file}' successfully modified and verified. Path: {target_file_abs}"
                logger.info(final_message)
                
                diff_str = None
                if original_file_content is not None: # final_rewritten_content is the content just written
                    diff_lines = difflib.unified_diff(
                        original_file_content.splitlines(keepends=True),
                        final_rewritten_content.splitlines(keepends=True),
                        fromfile=f"{target_file_abs.name}.original",
                        tofile=f"{target_file_abs.name}.modified",
                        lineterm=''
                    )
                    diff_str = "".join(diff_lines)
                return True, final_message, None, diff_str
            else:
                # Write failed after verification passed
                error_msg = f"Verification passed, but failed to write changes to file '{target_file}'. Error during write operation. Original file restored."  # Added restore info
                logger.error(error_msg)
                # Attempt to restore original content if possible
                if original_file_content is not None:
                    if write_final_content(target_file_abs, original_file_content, verbose=verbose):
                        logger.info(f"Restored original content to {target_file_abs}")
                    else:
                        logger.error(
                            f"FAILED TO RESTORE original content to {target_file_abs}"
                        )
                return False, error_msg, last_verifier_comment, diff_str

        except Exception as e:
            error_msg = f"Error: Verification passed, but failed to write changes to file '{target_file}' at '{target_file_abs}': {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, last_verifier_comment, diff_str
    else:
        # Verification failed after all retries or rewrite failed
        error_msg = f"Error: Failed to modify file '{target_file}' after {MAX_RETRIES} verification attempts. Last verifier comment: {last_verifier_comment}"
        logger.error(error_msg)
        # Ensure original content is restored if possible
        if original_file_content is not None:
            if write_final_content(target_file_abs, original_file_content, verbose=verbose):
                logger.info(
                    f"Restored original content to {target_file_abs} after failed attempts."
                )
            else:
                logger.error(
                    f"FAILED TO RESTORE original content to {target_file_abs} after failed attempts."
                )
        return False, error_msg, last_verifier_comment, None


# Keep main for potential standalone testing
if __name__ == "__main__":
    # Example usage for testing
    print("Testing file_modifier standalone.")
    test_target = "test_output/modified_file.py"
    test_original = (
        'def greet(name):\n    print(f"Hello, {name}!")\n\ngreet("World")\n'  # Original
    )
    test_snippets = '// ... existing code ...\ndef greet(name, enthusiasm="!"):\n    print(f"Hello, {name}{enthusiasm}")\n// ... existing code ...'  # Change snippets
    test_description = "Add an optional 'enthusiasm' parameter to the greet function and use it in the print statement."
    test_api_key = os.getenv("GEMINI_API_KEY")
    test_model = os.getenv("MODEL_REWRITE", "gemini-1.5-flash-latest")
    cwd = Path.cwd()

    # Ensure test file exists
    test_dir = cwd / "test_output"
    test_dir.mkdir(exist_ok=True)
    test_file_path = test_dir / "modified_file.py"
    test_file_path.write_text(test_original, encoding="utf-8")
    print(f"Created/Reset test file: {test_file_path}")

    if not test_api_key or not test_model:
        print(
            "Error: GEMINI_API_KEY or MODEL_REWRITE env var not set for testing.",
            file=sys.stderr,
        )
        sys.exit(1)

    success, message, comment, diff = handle_apply_diff(
        target_file=str(test_file_path.relative_to(cwd)),  # Pass relative path
        change_snippets=test_snippets,
        desired_changes_description=test_description,
        project_root=cwd,
        api_key=test_api_key,
        model_name=test_model,
        verbose=True
    )

    print(f"Success: {success}")
    print(f"Message: {message}")
    if comment:
        print(f"Verifier Comment: {comment}")
    if success:
        print(f"Check the modified file: {test_file_path}")
        if diff:
            print(f"Generated Diff:\n{diff}")
