#!/usr/bin/env python3
import os
import json
import argparse
import subprocess # Added for tree command
import glob       # Added for directory reading
import fnmatch    # Added for directory reading (if needed for filtering)
import sys        # Added for stderr
import copy       # Added for deepcopy
import requests   # Added for Gemini API call
from openai import OpenAI
from pathlib import Path # Added for summaries.py call

# --- Configuration ---
DEV_ROOT = Path("dev_helpers") # NEW: Root for helper scripts' data
O3_DIR = DEV_ROOT / "o3" # NEW: Path for o3 data
MESSAGES_FILE = Path("history") / "o3.log" # UPDATED: History file path as per new requirement

# --- Load o3 System Prompt from file ---
# Путь к файлу промпта относительно текущего файла (o3.py)
# o3.py находится в tools/, config/ находится на том же уровне, что и tools/
PROMPT_FILE_PATH = Path(__file__).parent.parent / "config" / "o3_system_prompt.txt"
SYSTEM_PROMPT = ""
try:
    with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f_prompt:
        SYSTEM_PROMPT = f_prompt.read().strip()
    if not SYSTEM_PROMPT:
        print(f"Warning: o3 system prompt file '{PROMPT_FILE_PATH}' is empty.", file=sys.stderr)
        # Можно установить дефолтный промпт или вызвать sys.exit()
except FileNotFoundError:
    print(f"Error: o3 system prompt file not found at '{PROMPT_FILE_PATH}'. Exiting.", file=sys.stderr)
    sys.exit(1) # Выход, если файл промпта не найден
except Exception as e:
    print(f"Error reading o3 system prompt file '{PROMPT_FILE_PATH}': {e}. Exiting.", file=sys.stderr)
    sys.exit(1) # Выход при других ошибках чтения
# --- End Configuration & Prompt Loading ---

# Initialize the client (ensure OPENAI_API_KEY and OPENAI_BASE_URL are set in env)
client = OpenAI()

# --- Function to get project tree structure (like ask.py) ---
def get_tree_output():
    """Runs the tree command excluding specified directories and returns its output."""
    try:
        # Use standard -I for ignoring patterns, matching ask.py exclusions
        command = ["tree", "-I", "node_modules|.git|__pycache__|venv*|dev_helpers", "--noreport"] # UPDATED: Exclude dev_helpers
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        return result.stdout
    except FileNotFoundError:
        print("Warning: 'tree' command not found. Install it (e.g., 'brew install tree') to include directory structure.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing tree command: {e}\n{e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running tree: {e}", file=sys.stderr)
        return None
# --- End Function to get project tree structure ---

# --- Function to read context directories ---
def read_context_directories(dirs_to_read=['dev_helpers/docs', 'dev_helpers/tasks', 'dev_helpers/summaries']):
    """Reads all files from specified directories (now inside dev_helpers) and returns formatted content."""
    full_context = ""
    for dir_path_str in dirs_to_read:
        dir_path = Path(dir_path_str) # Convert to Path object
        if dir_path.is_dir(): # Use pathlib check
            print(f"--- Reading context directory: {dir_path} ---", file=sys.stderr)
            dir_content = f"--- Directory: {dir_path} ---\n"
            found_files = False
            try:
                # Find all files recursively
                # Use pathlib's rglob for potentially better handling of paths
                file_paths = sorted([p for p in dir_path.rglob('*') if p.is_file()])

                for file_path in file_paths:
                    try:
                        # Use pathlib's read_text for simplicity
                        content = file_path.read_text(encoding='utf-8')
                        # Get relative path within the scanned directory
                        relative_path = file_path.relative_to(dir_path)
                        dir_content += f"--- File: {relative_path} ---\n{content}\n"
                        found_files = True
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
                if not found_files:
                     dir_content += "(No files found or read in this directory)\n"
            except Exception as e:
                print(f"Error scanning directory {dir_path}: {e}", file=sys.stderr)
                dir_content += f"(Error scanning directory: {e})\n"
            full_context += dir_content + "\n"
        else:
            print(f"--- Context directory not found, skipping: {dir_path} ---", file=sys.stderr)
    return full_context # Return only the directory context here
# --- End Function to read context directories ---

# --- Function to call Gemini for Summarization ---
def call_gemini_for_summary(history_to_summarize: str):
    """Calls the Gemini API to summarize the provided text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        return None

    model = "gemini-2.5-pro-preview-03-25"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    prompt = f"""Please concisely summarize the following conversation history:

{history_to_summarize}"""
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    })

    try:
        print(f"--- Calling Gemini API for summarization ({model}) ---", file=sys.stderr)
        response = requests.post(url, headers=headers, data=payload, timeout=60) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_json = response.json()
        # Navigate the response structure safely
        candidates = response_json.get('candidates', [])
        if candidates:
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            if parts:
                summary = parts[0].get('text')
                if summary:
                    print("--- Gemini Summary Received ---", file=sys.stderr)
                    return summary.strip()
        # Log if the expected structure wasn't found
        print(f"Warning: Could not extract summary from Gemini response. Response: {response_json}", file=sys.stderr)
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error during Gemini API call: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response from Gemini API: {response.text}", file=sys.stderr)
        return None
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during Gemini summarization: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return None
# --- End Function to call Gemini for Summarization ---

# Function to load messages (ensure directory exists)
def load_messages():
    # Ensure the target directory exists before trying to read the file
    O3_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists using pathlib
    if MESSAGES_FILE.exists():
        with MESSAGES_FILE.open('r', encoding='utf-8') as f:
            try:
                # Handle empty file case
                content = f.read()
                if not content:
                    return []
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Warning: Corrupt or empty {MESSAGES_FILE}. Starting fresh.", file=sys.stderr)
                return [] # Return empty list if file is empty or corrupt
    return []

# Function to save messages (ensure directory exists)
def save_messages(messages):
    # Ensure the target directory exists before writing
    O3_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists using pathlib
    # Ensure file is written with UTF-8 encoding and disable ASCII escaping
    with MESSAGES_FILE.open('w', encoding='utf-8') as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

# Function to process the conversation (modified to accept query content directly)
def process_conversation(user_query_content: str):
    messages = load_messages()

    # Add the user query (passed as argument) to persistent history
    if user_query_content:
        if not messages or messages[-1].get('content') != user_query_content or messages[-1].get('role') != 'user':
             messages.append({'role': 'user', 'content': user_query_content})
             save_messages(messages)
             print(f'--- Query Added to History: "{user_query_content[:50]}..." ---', file=sys.stderr)
        else:
             print(f'--- Query already in History: "{user_query_content[:50]}..." ---', file=sys.stderr)
    else:
        # Эта ветка не должна достигаться при вызове из gemini.py с аргументом
        print("Error: No query content provided to process_conversation.", file=sys.stderr)
        return "Error: No query content was provided to o3 agent."

    print('--- Preparing API Call for o3 Model ---', file=sys.stderr)
    try:
        # 1. Get dynamic context (Tree + Specific Dirs)
        tree_output = get_tree_output()
        dir_context = read_context_directories() # Reads docs, tasks, summaries from dev_helpers

        # --- Read project_manifest.json separately ---
        manifest_content = ""
        manifest_path = DEV_ROOT / "project_manifest.json" # UPDATED: Path inside dev_helpers
        if manifest_path.is_file(): # Use pathlib check
            try:
                with manifest_path.open('r', encoding='utf-8') as f_manifest: # Use pathlib open
                    manifest_content = f_manifest.read()
                print(f"--- Read project manifest file: {manifest_path} ---", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Could not read project manifest file {manifest_path}: {e}", file=sys.stderr)
                manifest_content = f"(Error reading {manifest_path}: {e})"
        else:
            print(f"--- Project manifest file not found, skipping: {manifest_path} ---", file=sys.stderr)
            manifest_content = f"({manifest_path} not found)"
        # --- End reading project_manifest.json ---

        # Format the dynamic context to be appended
        dynamic_context_for_api = f"""\n\n--- Current Project Structure (tree) ---
{tree_output if tree_output else '(tree command not available or failed)'}
--- End Project Structure ---

--- Project Manifest ({manifest_path.name}) ---
{manifest_content}
--- End Project Manifest ---

--- Context Directory Contents ---
{dir_context}--- End Context Directory Contents ---
"""

        # 2. Prepare messages for the API call
        # Start with system prompt
        api_messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        # Add messages from persistent history (use a deep copy to avoid modifying the persistent list)
        persistent_messages = copy.deepcopy([m for m in messages if m.get('content') is not None])
        api_messages.extend(persistent_messages)

        # 3. Inject dynamic context into the *last* message of the *API-bound copy*
        if api_messages and api_messages[-1]['role'] == 'user':
            # Append context to the latest user message content *in the copy*
            original_content = api_messages[-1].get('content', '')
            api_messages[-1]['content'] = original_content + dynamic_context_for_api
            print("--- Appended dynamic context to the last user message for API call (in temporary copy) ---", file=sys.stderr)
        else:
            # If history is empty or last message is not user, log warning.
            # Avoid sending context if it can't be appended naturally.
            print("Warning: Last message in history is not 'user' or history is empty. Dynamic context not appended.", file=sys.stderr)

        # 4. Call the API using the modified api_messages copy
        print('--- Calling OpenAI API (o3 model) --- ', file=sys.stderr)
        response = client.chat.completions.create(
            model='o3',
            reasoning_effort='high',
            messages=api_messages
        )

        # 5. Process the response
        o3_response_message = response.choices[0].message
        # Get the raw response content WITHOUT the dynamic context we added
        o3_response_content = o3_response_message.content if o3_response_message.content else ""
        print(f'--- o3 Raw Response Content Received: "{o3_response_content[:100]}..." ---', file=sys.stderr) # Log truncated response

        # 6. Append ONLY the raw assistant response to the persistent history
        messages.append({
            "role": "assistant",
            "content": o3_response_content
        })
        # --- New Summarization Logic ---
        # Count only user and assistant messages
        interaction_messages = [msg for msg in messages if msg.get('role') in ['user', 'assistant']]
        interaction_count = len(interaction_messages)
        print(f"--- Total user/assistant messages: {interaction_count} ---", file=sys.stderr)

        # Define thresholds
        SUMMARIZE_THRESHOLD = 20 # Trigger summarization when 20 user/assistant messages exist
        NUM_TO_SUMMARIZE = 10    # Summarize the oldest 10 messages
        NUM_TO_KEEP = 10         # Keep the newest 10 messages

        # Summarize if the threshold is met
        if interaction_count >= SUMMARIZE_THRESHOLD:
            print(f"--- Triggering summarization (interaction count >= {SUMMARIZE_THRESHOLD}) ---", file=sys.stderr)

            # Identify the oldest NUM_TO_SUMMARIZE user/assistant messages for the summary content
            messages_to_summarize_content = interaction_messages[:NUM_TO_SUMMARIZE]

            # Identify the newest NUM_TO_KEEP user/assistant messages to retain
            messages_to_keep_content = interaction_messages[-NUM_TO_KEEP:]

            # Format the messages for the summarization prompt
            history_string = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages_to_summarize_content])

            summary = call_gemini_for_summary(history_string)

            if summary:
                # Reconstruct the message list
                new_messages = []

                # Preserve the *original* initial system prompt if it exists
                # Check if the first message is a system prompt AND its content is the known SYSTEM_PROMPT
                if messages and messages[0].get('role') == 'system' and messages[0].get('content') == SYSTEM_PROMPT:
                     new_messages.append(messages[0])
                     print("--- Preserving initial system prompt ---", file=sys.stderr)

                # Add the new summary message
                new_messages.append({'role': 'system', 'content': f"Summary of prior conversation:\n{summary}"})
                print("--- Added new summary message ---", file=sys.stderr)

                # Add the messages we decided to keep
                new_messages.extend(messages_to_keep_content)
                print(f"--- Appended {len(messages_to_keep_content)} newest user/assistant messages ---", file=sys.stderr)

                # Replace the old message list
                messages = new_messages
                print("--- Conversation history summarized and replaced. ---", file=sys.stderr)
            else:
                print("Warning: Summarization failed. Proceeding with full history.", file=sys.stderr)
        else:
            print(f"--- Skipping summarization (interaction count < {SUMMARIZE_THRESHOLD}) ---", file=sys.stderr)

        # --- End New Summarization Logic ---

        save_messages(messages) # Save the updated (potentially summarized) persistent history
        print(f'--- History Saved to ({MESSAGES_FILE}) ---', file=sys.stderr) # Modified log message
        return o3_response_content # Return the raw response content

    except Exception as e:
        # Improve error reporting
        import traceback
        print(f"An error occurred during the API call or processing: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return f"Error: {e}" # Return error message

# --- Function to run summaries.py ---
def run_summaries_sync():
    """Runs the summaries.py script and handles potential errors."""
    print("--- Attempting to run summaries.py synchronization ---", file=sys.stderr, flush=True)
    summaries_script_path = Path(__file__).parent / "summaries.py"
    if not summaries_script_path.is_file():
        print(f"Warning: summaries.py not found at {summaries_script_path}. Skipping synchronization.", file=sys.stderr, flush=True)
        return

    try:
        # Ensure using python3 or the same interpreter running o3.py
        python_executable = sys.executable
        result = subprocess.run(
            [python_executable, str(summaries_script_path)],
            check=False, # Do not raise exception on non-zero exit
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode != 0:
            print(f"Warning: summaries.py finished with errors (return code {result.returncode}).", file=sys.stderr, flush=True)
            print("--- summaries.py stderr ---", file=sys.stderr, flush=True)
            print(result.stderr, file=sys.stderr, flush=True)
            print("--- end summaries.py stderr ---", file=sys.stderr, flush=True)
        else:
            print("--- summaries.py synchronization completed successfully. ---", file=sys.stderr, flush=True)
            # Optionally print stdout if needed for debugging, but it might be verbose
            # print("--- summaries.py stdout ---", file=sys.stderr, flush=True)
            # print(result.stdout, file=sys.stderr, flush=True)
            # print("--- end summaries.py stdout ---", file=sys.stderr, flush=True)

    except FileNotFoundError:
         print(f"Error: Python executable '{sys.executable}' not found while trying to run summaries.py. Skipping synchronization.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"An unexpected error occurred while running summaries.py: {e}", file=sys.stderr, flush=True)
# --- End Function --- 

# Main execution block
if __name__ == '__main__':
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description="o3 Agent: Interface to OpenAI model with context and history.")
    parser.add_argument("query", type=str, help="The query/message to send to the o3 agent.")
    args = parser.parse_args()

    # run_summaries_sync() # Можно оставить, если нужно

    # --- Убираем логику поиска и чтения файла запроса ---
    # print(f"--- Ensuring directory exists: {O3_DIR} ---", file=sys.stderr)
    # try:
    #     os.makedirs(O3_DIR, exist_ok=True)
    #     print(f"Directory '{O3_DIR}' is present.", file=sys.stderr)
    # except Exception as e:
    #     print(f"Error creating directory '{O3_DIR}': {e}", file=sys.stderr)
    #     sys.exit(1)
    #
    # print(f"--- Looking for query file in: {O3_DIR} ---", file=sys.stderr)
    # query_file_path = None
    # user_query = None
    # query_file_to_delete = None
    # try:
    #     # ... (старая логика чтения файла удалена) ...
    # except Exception as e:
    #     print(f"Error reading query file from '{O3_DIR}': {e}", file=sys.stderr)
    #     sys.exit(1)

    user_query_from_arg = args.query.strip()
    if not user_query_from_arg:
        print("Error: Query argument provided is empty.", file=sys.stderr)
        sys.exit(1)

    final_response = process_conversation(user_query_content=user_query_from_arg)
    print(final_response)

    # --- Логика удаления файла запроса больше не нужна ---
    # if query_file_to_delete:
    #     # ...
    # else:
    #      print("Warning: No query file path recorded to delete.", file=sys.stderr)