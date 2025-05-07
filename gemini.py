import os
from dotenv import load_dotenv
import sys
import json
from pathlib import Path
import math
import requests
import threading
import subprocess # Для вызова o3.py
import datetime # Added for network error logging

from utils.summarizer import summarize_history
from utils.history_manager import load_history, save_history
from utils.api_client import call_gemini_api, GeminiApiError
from utils.tool_parser import parse_tool_call, is_known_tool_call
from tools.command_executor import run_command
from tools.file_creator import handle_create_file
from tools.file_modifier import handle_apply_diff

# Определяем директорию, где находится сам скрипт
script_dir = Path(__file__).parent.resolve()

MAX_TOOL_OUTPUT_LINES = 500 # Макс. строк вывода команды для отправки модели
API_LOG_FILE_NAME = "api_interactions.log" # New log file name

def truncate_output_lines(text: str, max_lines: int) -> tuple[str, bool]:
    """Обрезает текст до max_lines строк, добавляя маркер усечения."""
    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        truncated = True
        # Берем половину начала и половину конца
        half_lines = max_lines // 2
        truncated_lines = lines[:half_lines] + ["\n... (output truncated) ...\n"] + lines[-half_lines:]
        return "\n".join(truncated_lines), truncated
    return text, truncated

def log_api_interaction(launch_dir: Path, request_details: dict, response_details: dict = None, error_details: dict = None):
    """Logs API request, response, and error details to a file."""
    log_file_path = launch_dir / API_LOG_FILE_NAME
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "request": request_details,
    }
    if response_details:
        log_entry["response"] = response_details
    if error_details:
        log_entry["error"] = error_details

    try:
        with open(log_file_path, 'a', encoding='utf-8') as f_log:
            f_log.write(json.dumps(log_entry, ensure_ascii=False, indent=2) + "\n---\n")
    except Exception as e_file_log:
        print(f"GEMINI_PY: CRITICAL - Failed to write to {API_LOG_FILE_NAME}: {e_file_log}", file=sys.stderr)

def main():
    # Определяем директорию запуска скрипта
    launch_dir = Path.cwd()
    # print(f"Директория запуска: {launch_dir}", file=sys.stderr)
    # print(f"Директория скрипта: {script_dir}", file=sys.stderr)

    # Путь к конфигу теперь относительно директории скрипта
    config_path = script_dir / "config/config.json"
    config = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model_name = config.get("model_name", "gemini-1.5-pro-latest")
        api_base_url = config.get("api_base_url")
        if not api_base_url:
             raise ValueError("Параметр api_base_url не найден в config.json")
        system_prompt_file = config.get("system_prompt_file")

        # История основного диалога сохраняется относительно директории ЗАПУСКА
        history_dir_name = config.get("history_dir", "history")
        history_file_name = config.get("history_file", "conversation_log.json")
        history_path = launch_dir / history_dir_name / history_file_name
        # Убедимся, что директория для истории существует
        (launch_dir / history_dir_name).mkdir(parents=True, exist_ok=True)

        rewrite_model_name = config.get("rewrite_model_name") # Получаем модель для рерайта
    except FileNotFoundError:
        print(f"Ошибка: Файл конфигурации {config_path} не найден.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Ошибка: Некорректный формат JSON в файле config.json.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении конфигурации: {e}", file=sys.stderr)
        sys.exit(1)

    # Загрузка системного промпта
    system_prompt_text = None
    if system_prompt_file:
        try:
            prompt_path = Path(system_prompt_file)
            if not prompt_path.is_absolute():
                 # Собираем путь относительно директории СКРИПТА
                 prompt_path = script_dir / prompt_path

            if prompt_path.is_file():
                system_prompt_text = prompt_path.read_text(encoding='utf-8').strip()
                # print(f"Загружен системный промпт из {prompt_path}", file=sys.stderr) # Убираем лог
            else:
                print(f"Предупреждение: Файл системного промпта не найден: {prompt_path}", file=sys.stderr)
        except Exception as e:
            print(f"Ошибка при чтении файла системного промпта {system_prompt_file}: {e}", file=sys.stderr)
            # Не прерываем выполнение, просто работаем без системного промпта

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Ошибка: Переменная окружения GEMINI_API_KEY не найдена.", file=sys.stderr)
        sys.exit(1)

    # print("Введите ваш запрос (или 'exit'/'quit' для выхода):", file=sys.stderr) # Removed as per request
    while True:
        final_model_response = None # Финальный текстовый ответ модели для этого хода
        request_log_details = {}
        response_log_details = {}
        error_log_details = {}
        raw_response_object = None # To store the requests.Response object

        try:
            prompt = input("> ")
            if prompt.lower() in ["exit", "quit"]:
                break
            if not prompt:
                continue

            # Начинаем обработку хода пользователя
            current_history = load_history(history_path)
            current_history.append({"role": "user", "parts": [{"text": prompt}]})

            # --- Внутренний цикл обработки API <-> Инструмент --- 
            processing_complete = False
            while not processing_complete:
                if len(current_history) > 1: # Only print if there is actual history beyond initial user prompt for this turn
                    print(f"Отправка запроса к API (История: {len(current_history)} сообщ.)...", file=sys.stderr)
                
                try:
                    # Prepare request details for logging BEFORE the call
                    # Mask API key in URL for logging
                    masked_api_url = f"{api_base_url}/{model_name}:generateContent?key=***REDACTED***"
                    request_log_details = {
                        "url": masked_api_url,
                        "method": "POST", # Assuming POST for Gemini API
                        "payload": current_history, # Log the payload sent
                        "system_prompt_present": bool(system_prompt_text)
                    }
                    response_log_details = {} # Reset for this attempt
                    error_log_details = {}    # Reset for this attempt
                    model_response_text = None
                    raw_response_object = None

                    # --- Modified call_gemini_api or wrapper needed here --- 
                    # Placeholder: direct requests call to illustrate logging, then integrate with call_gemini_api structure
                    
                    # This is where call_gemini_api is usually called.
                    # We need to get the raw requests.Response object from it.
                    # For now, I'll simulate what call_gemini_api does to get the response object.
                    
                    gemini_url_with_key = f"{api_base_url}/{model_name}:generateContent?key={api_key}"
                    payload_for_api = {
                        "contents": current_history,
                        "generationConfig": {"temperature": 0.7, "topP": 1.0, "maxOutputTokens": 100000} # Example config
                    }
                    if system_prompt_text:
                        payload_for_api["system_instruction"] = {"parts": [{"text": system_prompt_text}]}
                    
                    response = requests.post(
                        gemini_url_with_key, 
                        headers={"Content-Type": "application/json"},
                        json=payload_for_api,
                        timeout=120 # Example timeout
                    )
                    raw_response_object = response # Store for logging
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    
                    api_data = response.json()
                    if "candidates" in api_data and api_data["candidates"]:
                        candidate = api_data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                            model_response_text = candidate["content"]["parts"][0]["text"]
                        else:
                            model_response_text = "[API Error: No content in candidate]"
                            error_log_details["reason"] = "No content in candidate"
                    else:
                        model_response_text = "[API Error: No candidates in response]"
                        error_log_details["reason"] = "No candidates in API response"
                        if "promptFeedback" in api_data:
                             error_log_details["promptFeedback"] = api_data["promptFeedback"]
                             model_response_text += f" Feedback: {api_data['promptFeedback']}"

                except requests.exceptions.HTTPError as http_err:
                    raw_response_object = http_err.response # Get response from exception
                    model_response_text = f"[API HTTP Ошибка: {http_err}]"
                    error_log_details = {
                        "type": type(http_err).__name__,
                        "message": str(http_err),
                        "status_code": raw_response_object.status_code if raw_response_object else None
                    }
                    print(f"\nОшибка API (HTTP): {http_err}", file=sys.stderr)
                    # processing_complete = True # No, let the outer loop handle based on model_response_text
                except requests.exceptions.RequestException as req_err:
                    # raw_response_object might not be available here if it's a connection error before response
                    model_response_text = f"[Сетевая Ошибка: {req_err}]"
                    error_log_details = {"type": type(req_err).__name__, "message": str(req_err)}
                    print(f"GEMINI_PY: Ошибка сети: {req_err}", file=sys.stderr)
                    # processing_complete = True
                except GeminiApiError as api_err: # Assuming GeminiApiError is custom and might wrap details
                    model_response_text = f"[Ошибка Gemini API: {api_err}]"
                    error_log_details = {"type": type(api_err).__name__, "message": str(api_err), "details": getattr(api_err, 'details', None)}
                    print(f"\nОшибка API: {api_err}", file=sys.stderr)
                    # processing_complete = True
                except Exception as e: # Catch-all for other unexpected errors during API call phase
                    model_response_text = f"[Неожиданная Ошибка API: {e}]"
                    error_log_details = {"type": type(e).__name__, "message": str(e)}
                    print(f"\nНеожиданная ошибка при вызове API: {e}", file=sys.stderr)
                    # processing_complete = True
                finally:
                    # Log the interaction here, after the try/except for the API call
                    if raw_response_object is not None:
                        response_log_details = {
                            "status_code": raw_response_object.status_code,
                            "headers": dict(raw_response_object.headers),
                            "text_preview": raw_response_object.text[:1000] + ("..." if len(raw_response_object.text) > 1000 else "")
                        }
                    log_api_interaction(launch_dir, request_log_details, response_log_details, error_log_details)

                if model_response_text is None or model_response_text.startswith("[Ошибка") or model_response_text.startswith("[Сетевая Ошибка") or model_response_text.startswith("[API HTTP Ошибка"):
                    # If any error occurred and was translated into model_response_text, 
                    # we consider processing for this attempt complete with an error.
                    # The final_model_response will carry this error message for the user if no tools are called.
                    final_model_response = model_response_text or "[Неизвестная ошибка API]"
                    processing_complete = True # Break from while not processing_complete loop
                    continue # Continue to the next iteration of the main while True loop (prompt again)

                # --- Парсинг ответа модели --- 
                tool_call_info = parse_tool_call(model_response_text, verbose=False)

                if not tool_call_info:
                    # Инструмент не найден - это финальный ответ
                    final_model_response = model_response_text
                    processing_complete = True
                else:
                    # Инструмент найден, обрабатываем
                    tool_name, arguments = tool_call_info
                    tool_result_message = None

                    # Добавляем ответ модели (вызов инструмента) в историю ПЕРЕД выполнением
                    current_history.append({"role": "model", "parts": [{"text": model_response_text}]})

                    # --- Визуальное оформление вызова инструмента для консоли ---
                    tool_title = tool_name.replace('_', ' ').title()
                    box_top_bottom = f"╔{'═' * (len(tool_title) + 18)}╗"
                    box_middle = f"║ Executing Tool: {tool_title} ║"
                    print(box_top_bottom, file=sys.stderr)
                    print(box_middle, file=sys.stderr)
                    print(f"╚{'═' * (len(tool_title) + 18)}╝", file=sys.stderr)
                    # --- Конец визуального оформления ---

                    # --- Выполнение инструмента --- 
                    if tool_name == "execute_terminal_command":
                        command = arguments.get("command")
                        if command:
                            print(f"  Command: {command}", file=sys.stderr)
                            interrupt_event = threading.Event()
                            command_result = run_command(command, cwd=launch_dir, interrupt_event=interrupt_event, verbose=False)
                            
                            # Simplified console output
                            status_msg = f"  Result: {command_result['status']}, Exit Code: {command_result['exit_code']}"
                            if command_result.get('error_message'):
                                status_msg += f", Error: {command_result['error_message']}"
                            print(status_msg, file=sys.stderr)

                            stdout_console_truncated, stdout_console_was_truncated = truncate_output_lines(command_result['stdout'], 3) # Max 3 lines for console
                            stderr_console_truncated, stderr_console_was_truncated = truncate_output_lines(command_result['stderr'], 3) # Max 3 lines for console

                            if stdout_console_truncated:
                                print(f"  Stdout (preview):\n{stdout_console_truncated}", file=sys.stderr)
                            if stderr_console_truncated:
                                print(f"  Stderr (preview):\n{stderr_console_truncated}", file=sys.stderr)
                            
                            if stdout_console_was_truncated or stderr_console_was_truncated or command_result['truncated']: # original 'truncated' refers to byte truncation
                                print(f"  (Console output is truncated. Full output sent to API if applicable.)", file=sys.stderr)

                            # Prepare detailed tool_result_message for API (using MAX_TOOL_OUTPUT_LINES)
                            api_stdout_truncated, api_stdout_was_truncated = truncate_output_lines(command_result['stdout'], MAX_TOOL_OUTPUT_LINES)
                            api_stderr_truncated, api_stderr_was_truncated = truncate_output_lines(command_result['stderr'], MAX_TOOL_OUTPUT_LINES)
                            
                            tool_response_for_api = {
                                "status": command_result['status'], "exit_code": command_result['exit_code'],
                                "stdout": api_stdout_truncated, 
                                "stderr": api_stderr_truncated,
                                "truncated_lines": api_stdout_was_truncated or api_stderr_was_truncated,
                                "truncated_bytes": command_result['truncated'], # byte truncation from run_command
                                "error_message": command_result.get('error_message')
                            }
                            result_text_parts = []
                            if tool_response_for_api['stdout']: result_text_parts.append(f"Command stdout:\n```\n{tool_response_for_api['stdout']}\n```")
                            if tool_response_for_api['stderr']: result_text_parts.append(f"Command stderr:\n```\n{tool_response_for_api['stderr']}\n```")
                            if tool_response_for_api['status'] == 'error': result_text_parts.append(f"Command status: error (exit code: {tool_response_for_api['exit_code']}, message: {tool_response_for_api.get('error_message', 'N/A')})")
                            elif tool_response_for_api['exit_code'] != 0: result_text_parts.append(f"Command finished with exit code: {tool_response_for_api['exit_code']}")
                            else: result_text_parts.append(f"Command finished successfully (exit code: 0)")
                            if tool_response_for_api.get('truncated_lines') or tool_response_for_api.get('truncated_bytes'): result_text_parts.append("(Output was truncated for API message)")
                            tool_result_message = "\n".join(result_text_parts)
                        else:
                            tool_result_message = "[Ошибка парсера: Не найдена команда для execute_terminal_command]"
                            print(f"  Error: Missing command argument for execute_terminal_command.", file=sys.stderr)

                    elif tool_name == "create_file":
                        target_file_arg = arguments.get("target_file")
                        content = arguments.get("content")
                        if target_file_arg and content is not None:
                            print(f"  Target: {target_file_arg}", file=sys.stderr)
                            success, response_message = handle_create_file(
                                target_file=target_file_arg, content=content, project_root=launch_dir,
                                api_key=api_key, rewrite_model_name=rewrite_model_name, api_base_url=api_base_url,
                                verbose=False # Pass verbose=False
                            )
                            
                            if not success and response_message.startswith("FILE_ALREADY_EXISTS:"):
                                existing_file_path_reported = response_message.split(":", 1)[1]
                                print(f"  Info: Tool 'create_file' reported that file '{existing_file_path_reported}' already exists.", file=sys.stderr)
                                
                                # Construct the actual path relative to launch_dir for the cat command
                                # Assuming existing_file_path_reported is the same as target_file_arg used by the model
                                actual_existing_file_path = launch_dir / existing_file_path_reported
                                
                                cat_command = f"cat \"{str(actual_existing_file_path.resolve())}\"" # Use resolved absolute path for cat
                                print(f"  Executing: {cat_command} to get content of existing file.", file=sys.stderr)
                                interrupt_event_cat = threading.Event()
                                cat_result = run_command(cat_command, cwd=launch_dir, interrupt_event=interrupt_event_cat)
                                
                                file_content_for_api = "[Could not read content of existing file]"
                                if cat_result['status'] == 'completed' and cat_result['exit_code'] == 0:
                                    api_cat_stdout, _ = truncate_output_lines(cat_result['stdout'], MAX_TOOL_OUTPUT_LINES)
                                    file_content_for_api = api_cat_stdout
                                elif cat_result.get('error_message') or cat_result.get('stderr'):
                                    error_info = cat_result.get('error_message', '')
                                    if cat_result.get('stderr'):
                                        api_cat_stderr, _ = truncate_output_lines(cat_result['stderr'], MAX_TOOL_OUTPUT_LINES // 2)
                                        error_info += f" Stderr: {api_cat_stderr}"
                                    file_content_for_api = f"[Error reading existing file: {error_info.strip()}]"
                                else:
                                    file_content_for_api = f"[Error reading existing file: status {cat_result['status']}, exit_code {cat_result['exit_code']}]"

                                tool_result_message = (
                                    f"Tool 'create_file' was not executed because the file '{existing_file_path_reported}' already exists.\n"
                                    f"Content of existing file '{existing_file_path_reported}':\n```\n{file_content_for_api}\n```\n"
                                    f"If you want to modify this file, please use the 'apply_diff' tool."
                                )
                                print(f"  Sending notice to API: File '{existing_file_path_reported}' exists, its content is included.", file=sys.stderr)
                            else:
                                tool_result_message = f"File creation result: {response_message}" # For API
                                print(f"  Result: {response_message}", file=sys.stderr) # For console
                        else:
                            tool_result_message = "[Ошибка парсера: Не найдены target_file или content для create_file]"
                            print(f"  Error: Missing target_file or content for create_file.", file=sys.stderr)

                    elif tool_name == "apply_diff":
                        target_file = arguments.get("target_file")
                        diff_content = arguments.get("diff_content")
                        if target_file and diff_content:
                            print(f"  Target: {target_file}", file=sys.stderr)
                            # print(f"  Diff Length: {len(diff_content)} bytes", file=sys.stderr) # Optional
                            success, message, verifier_comment = handle_apply_diff(
                                target_file=target_file,
                                diff_content=diff_content,
                                project_root=launch_dir,
                                api_key=api_key,
                                model_name=rewrite_model_name,
                                verbose=False # Pass verbose=False
                            )
                            tool_result_message = f"File modification result ({target_file}): {message}" # For API
                            print(f"  Result: {message}", file=sys.stderr) # For console
                            if not success and verifier_comment:
                                tool_result_message += f"\nVerifier Comment: {verifier_comment}" # Append to API message
                                print(f"  Verifier Comment: {verifier_comment}", file=sys.stderr) # Show in console
                        else:
                            tool_result_message = "[Ошибка парсера: Не найдены target_file или diff_content для apply_diff]"
                            print(f"  Error: Missing target_file or diff_content for apply_diff.", file=sys.stderr)

                    elif tool_name == "talk_to_o3":
                        message_for_o3 = arguments.get("message_for_o3")
                        if message_for_o3:
                            # print(f"  Message for o3 (preview): {message_for_o3[:70]}...", file=sys.stderr) # Optional:
                            try:
                                o3_script_abs_path = script_dir / "tools" / "o3.py"
                                python_executable = sys.executable
                                # The internal "GEMINI_PY: Вызов..." print for o3.py is in the script itself, cannot remove from here easily.
                                process = subprocess.run(
                                    [python_executable, str(o3_script_abs_path), message_for_o3],
                                    capture_output=True, text=True, cwd=launch_dir, check=False
                                )
                                if process.returncode == 0:
                                    response_preview = process.stdout.strip()[:200] + ('...' if len(process.stdout.strip()) > 200 else '')
                                    tool_result_message = f"Response from o3:\n```\n{process.stdout.strip()}\n```" # Full response for API
                                    print(f"  Result: Success. o3 Response (preview):\n{response_preview}", file=sys.stderr) # Preview for console
                                else:
                                    error_details = f"RC: {process.returncode}"
                                    if process.stderr.strip(): error_details += f", Stderr: {process.stderr.strip()[:100]}..."
                                    if process.stdout.strip(): error_details += f", Stdout: {process.stdout.strip()[:100]}..."
                                    tool_result_message = f"[Ошибка при вызове o3]: Error calling o3.py. {error_details}" # For API
                                    print(f"  Result: Error calling o3.py. {error_details}", file=sys.stderr) # For console
                            except Exception as e_o3_call:
                                tool_result_message = f"[Неожиданная ошибка при подготовке или вызове o3]: {e_o3_call}"
                                print(f"  Result: Unexpected error during o3 call: {e_o3_call}", file=sys.stderr)
                        else:
                            tool_result_message = "[Ошибка парсера: Не найден message_for_o3 для talk_to_o3]"
                            print(f"  Error: Missing message_for_o3 for talk_to_o3.", file=sys.stderr)
                    else:
                        tool_result_message = f"[Ошибка: Неизвестный инструмент '{tool_name}']"
                        print(f"  Error: Unknown tool '{tool_name}'", file=sys.stderr)
                    
                    print("-" * 70, file=sys.stderr) # Adjusted separator line width

                    # Добавляем результат инструмента в историю (как user)
                    current_history.append({"role": "user", "parts": [{"text": tool_result_message}]})
                    # Продолжаем внутренний цикл, чтобы отправить результат модели
            # --- Конец внутреннего цикла обработки --- 

            # --- Вывод финального ответа и сохранение --- 
            if final_model_response:
                # Выводим финальный ответ
                terminal_width = os.get_terminal_size().columns
                lines = final_model_response.splitlines()
                max_line_width = max(len(line) for line in lines) if lines else 0
                box_width = min(terminal_width - 4, max(max_line_width, 10))
                print(f"╔{'═' * (box_width + 2)}╗")
                for line in lines:
                    # Word wrapping for long lines
                    idx = 0
                    while idx < len(line):
                        print(f"║ {line[idx:idx+box_width]:<{box_width}} ║")
                        idx += box_width
                    if not line: # Handle empty lines within the box
                        print(f"║ {' ':<{box_width}} ║")
                print(f"╚{'═' * (box_width + 2)}╝")

                # Добавляем финальный ответ модели в историю (или обновляем, если он там уже есть? Нет, просто добавляем)
                # Убедимся, что не добавляем его дважды, если он уже последний?
                # Проще всего добавить его после внутреннего цикла
                current_history.append({"role": "model", "parts": [{"text": final_model_response}]})

            # Фильтруем историю ПЕРЕД сохранением
            history_to_save = [
                msg for msg in current_history
                if not (msg.get('role') == 'user' and msg.get('parts', [{}])[0].get('text', '').startswith(("Command stdout:", "Command status:", "File creation result:", "File modification result", "Response from o3:", "[Ошибка при вызове o3]:"))) and \
                   not (msg.get('role') == 'model' and is_known_tool_call(msg.get('parts', [{}])[0].get('text', '')))
            ]
            save_history(history_path, history_to_save)

            # === Вызов Суммаризации ===
            summarized_history = summarize_history(current_history, config, api_key)
            if summarized_history is not None:
                summarized_history_to_save = [
                     msg for msg in summarized_history
                     if not (msg.get('role') == 'user' and msg.get('parts', [{}])[0].get('text', '').startswith(("Command stdout:", "Command status:", "File creation result:", "File modification result", "Response from o3:", "[Ошибка при вызове o3]:"))) and \
                        not (msg.get('role') == 'model' and is_known_tool_call(msg.get('parts', [{}])[0].get('text', '')))
                ]
                save_history(history_path, summarized_history_to_save)
            # === Конец Суммаризации ===

        except EOFError:
             break
        except KeyboardInterrupt:
            print("\nВыход.", file=sys.stderr)
            break
        except Exception as e:
            import traceback
            print(f"\nНеожиданная ошибка в главном цикле: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

if __name__ == "__main__":
    main()