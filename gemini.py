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
import time # Added for sleep
import re # Added for parsing model response text

from utils.summarizer import summarize_history
from utils.history_manager import load_history, save_history
from utils.api_client import call_gemini_api, GeminiApiError
from utils.tool_parser import (parse_tool_call, is_known_tool_call,
    EXECUTE_TERMINAL_PATTERN, CREATE_FILE_PATTERN,
    APPLY_DIFF_PATTERN, TALK_TO_O3_PATTERN)
from tools.command_executor import run_command
from tools.file_creator import handle_create_file
from tools.file_modifier import handle_apply_diff

# Определяем директорию, где находится сам скрипт
script_dir = Path(__file__).parent.resolve()

MAX_TOOL_OUTPUT_LINES = 500 # Макс. строк вывода команды для отправки модели
API_LOG_FILE_NAME = "api_interactions.log" # New log file name

# --- Global ID for interrupt mechanism ---
current_processing_id = 0
# --- End Global ID ---

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
    global current_processing_id # Allow modification of global ID

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

    # --- Summarize history on startup ---
    try:
        startup_history = load_history(history_path)
        if startup_history: 
            summarization_config = config.get("summarization", {})
            summarization_enabled_on_startup = summarization_config.get("enabled", True) # Default to True if not specified
            summarization_threshold_on_startup = summarization_config.get("threshold_messages", 20)
            
            if not summarization_enabled_on_startup:
                print("GEMINI_PY: Startup summarization is disabled in config. Skipping.", file=sys.stderr)
            elif len(startup_history) > summarization_threshold_on_startup:
                print(f"GEMINI_PY: Attempting startup summarization for history with {len(startup_history)} messages (threshold: {summarization_threshold_on_startup}).", file=sys.stderr)
                summarized_startup_history = summarize_history(startup_history, config, api_key)
                if summarized_startup_history is not None and summarized_startup_history != startup_history:
                    save_history(history_path, summarized_startup_history)
                    print(f"GEMINI_PY: Startup summarization complete. History updated from {len(startup_history)} to {len(summarized_startup_history)} messages.", file=sys.stderr)
                else:
                    print("GEMINI_PY: No changes made by startup summarization or summarization failed to return new history.", file=sys.stderr)
        else:
            print("GEMINI_PY: History is empty. No startup summarization needed.", file=sys.stderr)
    except Exception as e_startup_summarize:
        print(f"GEMINI_PY: Error during startup summarization: {e_startup_summarize}", file=sys.stderr)
    # --- End startup summarization ---

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
            
            if prompt == "!!":
                current_processing_id += 1
                print("GEMINI_PY: Обработка предыдущего запроса отменена. Введите новый запрос.", file=sys.stderr)
                final_model_response = None 
                continue 

            if not prompt:
                continue

            id_for_this_interaction_sequence = current_processing_id

            current_history = load_history(history_path)
            current_history.append({"role": "user", "parts": [{"text": prompt}]})
            
            # --- Threshold-based Summarization before processing ---
            summarization_config = config.get("summarization", {})
            summarization_enabled_for_turn = summarization_config.get("enabled", True)
            summarization_threshold = summarization_config.get("threshold_messages", 20) 

            if not summarization_enabled_for_turn:
                print("GEMINI_PY: Pre-turn summarization is disabled in config. Skipping.", file=sys.stderr)
            elif len(current_history) > summarization_threshold:
                print(f"GEMINI_PY: History length ({len(current_history)}) exceeds threshold ({summarization_threshold}). Triggering summarization.", file=sys.stderr)
                try:
                    summarized_history_for_turn = summarize_history(current_history, config, api_key)
                    if summarized_history_for_turn is not None and summarized_history_for_turn != current_history:
                        print(f"GEMINI_PY: Pre-turn summarization updated history from {len(current_history)} to {len(summarized_history_for_turn)} messages.", file=sys.stderr)
                        current_history = summarized_history_for_turn
                        # save_history(history_path, current_history) # Saved below
                    elif summarized_history_for_turn is None:
                        print("GEMINI_PY: Pre-turn summarization returned None, original history will be used.", file=sys.stderr)
                    else: # summarized_history_for_turn == current_history
                        print("GEMINI_PY: No changes made by pre-turn summarization.", file=sys.stderr)
                except Exception as e_threshold_summarize:
                     print(f"GEMINI_PY: Error during threshold summarization: {e_threshold_summarize}. Original history will be used.", file=sys.stderr)
            # --- End Threshold-based Summarization ---

            # --- Сохранение пользовательского сообщения (и возможно, суммаризированной истории) в реальном времени ---
            save_history(history_path, current_history) 
            # --- Конец сохранения в реальном времени ---

            # --- Внутренний цикл обработки API <-> Инструмент --- 
            processing_complete_for_turn = False
            while not processing_complete_for_turn:
                # Check if this interaction sequence has been invalidated by '!!'
                if current_processing_id != id_for_this_interaction_sequence:
                    print("GEMINI_PY: Текущая последовательность обработки была отменена пользователем.", file=sys.stderr)
                    final_model_response = "[Обработка отменена пользователем]"
                    break # Exit the inner 'while not processing_complete_for_turn' loop

                if len(current_history) > 1: # Only print if there is actual history beyond initial user prompt for this turn
                    print(f"Отправка запроса к API (История: {len(current_history)} сообщ.)...", file=sys.stderr)
                
                # --- API Call Section ---
                api_result_package = {
                    "text": None,
                    "error": False, # True if API call itself failed or returned error structure
                    "error_message_from_api": None,
                    "original_id": id_for_this_interaction_sequence # Associate with current sequence ID
                }

                try:
                    # Prepare request details for logging BEFORE the call
                    masked_api_url = f"{api_base_url}/{model_name}:generateContent?key=***REDACTED***"
                    request_log_details = {
                        "url": masked_api_url,
                        "method": "POST", 
                        "payload": current_history, 
                        "system_prompt_present": bool(system_prompt_text)
                    }
                    response_log_details = {} 
                    error_log_details = {}    
                    model_response_text_from_api = None # Renamed to avoid confusion
                    raw_response_object = None
                    
                    gemini_url_with_key = f"{api_base_url}/{model_name}:generateContent?key={api_key}"
                    payload_for_api = {
                        "contents": current_history,
                        "generationConfig": {"temperature": 0.7, "topP": 1.0, "maxOutputTokens": 100000} 
                    }
                    if system_prompt_text:
                        payload_for_api["system_instruction"] = {"parts": [{"text": system_prompt_text}]}
                    
                    # *** THE BLOCKING API CALL ***
                    response = requests.post(
                        gemini_url_with_key, 
                        headers={"Content-Type": "application/json"},
                        json=payload_for_api,
                        timeout=120 
                    )
                    raw_response_object = response 
                    response.raise_for_status() 
                    
                    api_data = response.json()
                    if "candidates" in api_data and api_data["candidates"]:
                        candidate = api_data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                            model_response_text_from_api = candidate["content"]["parts"][0]["text"]
                        else:
                            model_response_text_from_api = "[API Error: No content in candidate]"
                            error_log_details["reason"] = "No content in candidate"
                            api_result_package["error"] = True
                    else:
                        model_response_text_from_api = "[API Error: No candidates in response]"
                        error_log_details["reason"] = "No candidates in API response"
                        api_result_package["error"] = True
                        if "promptFeedback" in api_data:
                             error_log_details["promptFeedback"] = api_data["promptFeedback"]
                             model_response_text_from_api += f" Feedback: {api_data['promptFeedback']}"
                    
                    api_result_package["text"] = model_response_text_from_api

                except requests.exceptions.HTTPError as http_err:
                    raw_response_object = http_err.response 
                    error_message = f"[API HTTP Ошибка: {http_err}]"
                    api_result_package["text"] = error_message
                    api_result_package["error"] = True
                    api_result_package["error_message_from_api"] = error_message
                    error_log_details = {
                        "type": type(http_err).__name__,
                        "message": str(http_err),
                        "status_code": raw_response_object.status_code if raw_response_object else None
                    }
                    print(f"\nОшибка API (HTTP): {http_err}", file=sys.stderr)
                except requests.exceptions.RequestException as req_err:
                    error_message = f"[Сетевая Ошибка: {req_err}]"
                    api_result_package["text"] = error_message
                    api_result_package["error"] = True
                    api_result_package["error_message_from_api"] = error_message
                    error_log_details = {"type": type(req_err).__name__, "message": str(req_err)}
                    print(f"GEMINI_PY: Ошибка сети: {req_err}", file=sys.stderr)
                except GeminiApiError as api_err: 
                    error_message = f"[Ошибка Gemini API: {api_err}]"
                    api_result_package["text"] = error_message
                    api_result_package["error"] = True
                    api_result_package["error_message_from_api"] = error_message
                    error_log_details = {"type": type(api_err).__name__, "message": str(api_err), "details": getattr(api_err, 'details', None)}
                    print(f"\nОшибка API: {api_err}", file=sys.stderr)
                except Exception as e: 
                    error_message = f"[Неожиданная Ошибка API: {e}]"
                    api_result_package["text"] = error_message
                    api_result_package["error"] = True
                    api_result_package["error_message_from_api"] = error_message
                    error_log_details = {"type": type(e).__name__, "message": str(e)}
                    print(f"\nНеожиданная ошибка при вызове API: {e}", file=sys.stderr)
                finally:
                    if raw_response_object is not None:
                        response_log_details = {
                            "status_code": raw_response_object.status_code,
                            "headers": dict(raw_response_object.headers),
                            "text_preview": raw_response_object.text[:1000] + ("..." if len(raw_response_object.text) > 1000 else "")
                        }
                    log_api_interaction(launch_dir, request_log_details, response_log_details, error_log_details)

                # --- Check if API call is still valid (not invalidated by '!!') ---
                if current_processing_id != api_result_package["original_id"]:
                    print("GEMINI_PY: Ответ API проигнорирован (устарел из-за отмены пользователем).", file=sys.stderr)
                    final_model_response = "[Ответ модели отменен пользователем]" 
                    processing_complete_for_turn = True 
                    continue # Continue in the inner 'while not processing_complete_for_turn' loop, which will then terminate.

                model_response_text = api_result_package["text"]

                # Check for API call errors even if ID matches
                if api_result_package["error"] or model_response_text is None or \
                   model_response_text.startswith("[API Error:") or \
                   model_response_text.startswith("[API HTTP Ошибка:") or \
                   model_response_text.startswith("[Сетевая Ошибка:") or \
                   model_response_text.startswith("[Ошибка Gemini API:") or \
                   model_response_text.startswith("[Неожиданная Ошибка API:"):
                    final_model_response = model_response_text or "[Неизвестная ошибка API]"
                    processing_complete_for_turn = True 
                    continue 

                # --- Парсинг ответа модели и отображение текстовой части --- 
                tool_call_info = parse_tool_call(model_response_text, verbose=True) # Parse once

                if not tool_call_info:
                    # Инструмент не найден - это финальный ответ
                    final_model_response = model_response_text
                    processing_complete_for_turn = True
                else:
                    # Инструмент найден. model_response_text is the full string.
                    # Попытка извлечь и напечатать текстовый префикс перед JSON-блоком инструмента.
                    text_prefix_to_print = ""
                    json_block_start_index = -1

                    # Новая логика с использованием паттернов из tool_parser.py
                    tool_name_from_parse = tool_call_info[0] # tool_name уже есть из parse_tool_call
                    specific_tool_pattern = None
                    if tool_name_from_parse == "execute_terminal_command":
                        specific_tool_pattern = EXECUTE_TERMINAL_PATTERN
                    elif tool_name_from_parse == "create_file":
                        specific_tool_pattern = CREATE_FILE_PATTERN
                    elif tool_name_from_parse == "apply_diff":
                        specific_tool_pattern = APPLY_DIFF_PATTERN
                    elif tool_name_from_parse == "talk_to_o3":
                        specific_tool_pattern = TALK_TO_O3_PATTERN

                    if specific_tool_pattern:
                        type_match = specific_tool_pattern.search(model_response_text)
                        if type_match:
                            # Ищем открывающую фигурную скобку перед найденным паттерном "tool": "name"
                            # model_response_text.rfind('{', 0, type_match.start()) ищет ПОСЛЕДНЮЮ { перед началом type_match
                            # это должно быть правильно для стандартного JSON объекта.
                            potential_json_start = model_response_text.rfind('{', 0, type_match.start())
                            if potential_json_start != -1:
                                json_block_start_index = potential_json_start
                        else:
                            # Это очень странно: parse_tool_call нашел инструмент, а поиск по его типу - нет.
                            # Возможно, parse_tool_call использует более сложную логику, или текст ответа сильно изменен.
                            print(f"GEMINI_PY: Warning: Tool '{tool_name_from_parse}' parsed, but its specific type pattern not found. Falling back for prefix extraction.", file=sys.stderr)
                    
                    # Запасной или основной (если specific_tool_pattern не сработал) метод поиска начала JSON
                    if json_block_start_index == -1:
                        fallback_match = re.search(r"{\s*\"tool\":\s*\"", model_response_text)
                        if fallback_match:
                            json_block_start_index = fallback_match.start()
                        elif not model_response_text.strip().startswith("{"):
                             # Если и fallback не нашел, но parse_tool_call что-то вернул и это не начинается с {, выводим предупреждение.
                            print("GEMINI_PY: Warning: Tool call parsed, but prefix extraction regex failed. Model output format may be unusual.", file=sys.stderr)
                    
                    if json_block_start_index != -1:
                        text_prefix_to_print = model_response_text[:json_block_start_index].strip()
                    
                    if text_prefix_to_print: # Печатаем префикс, если он есть
                        terminal_width_prefix = os.get_terminal_size().columns
                        lines_prefix = text_prefix_to_print.splitlines()
                        max_line_width_prefix = max(len(line) for line in lines_prefix) if lines_prefix else 0
                        box_width_prefix = min(terminal_width_prefix - 4, max(max_line_width_prefix, 10))
                        
                        print(f"╔{'═' * (box_width_prefix + 2)}╗") # На stdout
                        for line_prefix in lines_prefix:
                            idx_prefix = 0
                            while idx_prefix < len(line_prefix):
                                print(f"║ {line_prefix[idx_prefix:idx_prefix+box_width_prefix]:<{box_width_prefix}} ║")
                                idx_prefix += box_width_prefix
                            if not line_prefix: 
                                print(f"║ {' ':<{box_width_prefix}} ║")
                        print(f"╚{'═' * (box_width_prefix + 2)}╝")

                    # Инструмент найден, обрабатываем (tool_call_info уже есть)
                    tool_name, arguments = tool_call_info
                    tool_result_message = None

                    # Добавляем ответ модели (вызов инструмента) в историю ПЕРЕД выполнением
                    # model_response_text содержит и текстовый префикс, и сам вызов инструмента
                    current_history.append({"role": "model", "parts": [{"text": model_response_text}]})
                    # --- Сохранение вызова инструмента моделью в реальном времени ---
                    save_history(history_path, current_history)
                    # --- Конец сохранения --- 

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
                            
                            status_msg = f"  Result: {command_result['status']}, Exit Code: {command_result['exit_code']}"
                            if command_result.get('error_message'):
                                status_msg += f", Error: {command_result['error_message']}"
                            print(status_msg, file=sys.stderr)

                            stdout_console_truncated, stdout_console_was_truncated = truncate_output_lines(command_result['stdout'], 3)
                            stderr_console_truncated, stderr_console_was_truncated = truncate_output_lines(command_result['stderr'], 3)

                            if stdout_console_truncated:
                                print(f"  Stdout (preview):\n{stdout_console_truncated}", file=sys.stderr)
                            if stderr_console_truncated:
                                print(f"  Stderr (preview):\n{stderr_console_truncated}", file=sys.stderr)
                            
                            if stdout_console_was_truncated or stderr_console_was_truncated or command_result['truncated']:
                                print(f"  (Console output is truncated. Full output sent to API if applicable.)", file=sys.stderr)

                            api_stdout_truncated, api_stdout_was_truncated = truncate_output_lines(command_result['stdout'], MAX_TOOL_OUTPUT_LINES)
                            api_stderr_truncated, api_stderr_was_truncated = truncate_output_lines(command_result['stderr'], MAX_TOOL_OUTPUT_LINES)
                            
                            tool_response_for_api = {
                                "status": command_result['status'], "exit_code": command_result['exit_code'],
                                "stdout": api_stdout_truncated, 
                                "stderr": api_stderr_truncated,
                                "truncated_lines": api_stdout_was_truncated or api_stderr_was_truncated,
                                "truncated_bytes": command_result['truncated'],
                                "error_message": command_result.get('error_message')
                            }

                            result_text_parts = [
                                f"Tool Result: execute_terminal_command",
                                f"Command: {command}",
                                f"Status: {tool_response_for_api['status']} (Exit Code: {tool_response_for_api['exit_code']})"
                            ]
                            if tool_response_for_api.get('error_message') and tool_response_for_api['status'] == 'error':
                                result_text_parts.append(f"Error Message: {tool_response_for_api['error_message']}")
                            if tool_response_for_api['stdout']:
                                result_text_parts.append(f"Stdout:\n```\n{tool_response_for_api['stdout']}\n```")
                            if tool_response_for_api['stderr']:
                                result_text_parts.append(f"Stderr:\n```\n{tool_response_for_api['stderr']}\n```")
                            
                            truncation_info_parts = []
                            if tool_response_for_api.get('truncated_lines'): truncation_info_parts.append("output lines truncated")
                            if tool_response_for_api.get('truncated_bytes'): truncation_info_parts.append("output bytes truncated")
                            if truncation_info_parts:
                                result_text_parts.append(f"(Truncation: {', '.join(truncation_info_parts)})")
                            
                            tool_result_message = "\n".join(result_text_parts)
                        else:
                            tool_result_message = "[Tool Error: execute_terminal_command - Missing command argument]"
                            print(f"  Error: Missing command argument for execute_terminal_command.", file=sys.stderr)

                    elif tool_name == "create_file":
                        target_file_arg = arguments.get("target_file")
                        content_arg = arguments.get("content")
                        if target_file_arg and content_arg is not None:
                            print(f"  Target: {target_file_arg}", file=sys.stderr)
                            success, response_message = handle_create_file(
                                target_file=target_file_arg, content=content_arg, project_root=launch_dir,
                                api_key=api_key, rewrite_model_name=rewrite_model_name, api_base_url=api_base_url,
                                verbose=False
                            )
                            
                            if not success and response_message.startswith("FILE_ALREADY_EXISTS:"):
                                existing_file_path_reported = response_message.split(":", 1)[1]
                                print(f"  Info: Tool 'create_file' reported that file '{existing_file_path_reported}' already exists.", file=sys.stderr)
                                
                                actual_existing_file_path = launch_dir / existing_file_path_reported
                                cat_command = f'cat "{str(actual_existing_file_path.resolve())}"'
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

                                tool_result_message = (
                                    f"Tool Result: create_file\nFile: {existing_file_path_reported}\nStatus: File Already Exists\n"
                                    f"Content of existing file '{existing_file_path_reported}':\n```\n{file_content_for_api}\n```\n"
                                    f"If you want to modify this file, please use the 'apply_diff' tool."
                                )
                                print(f"  Sending notice to API: File '{existing_file_path_reported}' exists, its content is included.", file=sys.stderr)
                            elif success:
                                # --- Добавлена пауза для возможной задержки файловой системы ---
                                time.sleep(0.2) 
                                # --- Конец паузы ---
                                cat_command_create_file = f'cat "{str((launch_dir / target_file_arg).resolve())}"'
                                interrupt_event_cat_create_file = threading.Event()
                                cat_result_create_file = run_command(cat_command_create_file, cwd=launch_dir, interrupt_event=interrupt_event_cat_create_file)

                                created_file_content_for_api = "[Could not read content of created file]"
                                if cat_result_create_file['status'] == 'completed' and cat_result_create_file['exit_code'] == 0:
                                    api_cat_stdout_create_file, _ = truncate_output_lines(cat_result_create_file['stdout'], MAX_TOOL_OUTPUT_LINES)
                                    created_file_content_for_api = api_cat_stdout_create_file
                                elif cat_result_create_file.get('error_message') or cat_result_create_file.get('stderr'):
                                    error_info_create_file = cat_result_create_file.get('error_message', '')
                                    if cat_result_create_file.get('stderr'):
                                        api_cat_stderr_create_file, _ = truncate_output_lines(cat_result_create_file['stderr'], MAX_TOOL_OUTPUT_LINES // 2)
                                        error_info_create_file += f" Stderr: {api_cat_stderr_create_file}"
                                    created_file_content_for_api = f"[Error reading created file: {error_info_create_file.strip()}]"
                                
                                tool_result_message = (
                                    f"Tool Result: create_file\nFile: {target_file_arg}\nStatus: Success\n"
                                    f"Message: {response_message}\n"
                                    f"Content:\n```\n{created_file_content_for_api}\n```"
                                )
                                print(f"  Result: {response_message}", file=sys.stderr)
                            else: # Generic error from handle_create_file
                                tool_result_message = f"Tool Result: create_file\nFile: {target_file_arg}\nStatus: Error\nMessage: {response_message}"
                                print(f"  Result: {response_message}", file=sys.stderr)
                        else:
                            tool_result_message = "[Tool Error: create_file - Missing target_file or content arguments]"
                            print(f"  Error: Missing target_file or content for create_file.", file=sys.stderr)

                    elif tool_name == "apply_diff":
                        target_file_apply_diff_arg = arguments.get("target_file") # Renamed variable
                        # --- Используем правильные ключи, возвращаемые парсером ---
                        change_snippets_arg = arguments.get("change_snippets") \
                        or arguments.get("diff_content")
                        desired_changes_description_arg = arguments.get(
                            "desired_changes_description", "No overall description provided.")

                        if target_file_apply_diff_arg and change_snippets_arg: # Проверяем change_snippets_arg
                            print(f"  Target: {target_file_apply_diff_arg}", file=sys.stderr)
                            success, message, verifier_comment, diff_str = handle_apply_diff( 
                                target_file=target_file_apply_diff_arg,
                                change_snippets=change_snippets_arg, # Используем change_snippets_arg
                                desired_changes_description=desired_changes_description_arg, # Это поле уже было правильным
                                project_root=launch_dir,
                                api_key=api_key,
                                model_name=rewrite_model_name, 
                                verbose=False
                            )
                            
                            if success:
                                # --- Добавлена пауза для возможной задержки файловой системы ---
                                time.sleep(0.2) 
                                # --- Конец паузы ---
                                cat_command_apply_diff = f'cat "{str((launch_dir / target_file_apply_diff_arg).resolve())}"'
                                interrupt_event_cat_apply_diff = threading.Event()
                                cat_result_apply_diff = run_command(cat_command_apply_diff, cwd=launch_dir, interrupt_event=interrupt_event_cat_apply_diff)
                                
                                modified_file_content_for_api = "[Could not read content of modified file after apply_diff]"
                                if cat_result_apply_diff['status'] == 'completed' and cat_result_apply_diff['exit_code'] == 0:
                                    api_cat_stdout_apply_diff, _ = truncate_output_lines(cat_result_apply_diff['stdout'], MAX_TOOL_OUTPUT_LINES)
                                    modified_file_content_for_api = api_cat_stdout_apply_diff
                                elif cat_result_apply_diff.get('error_message') or cat_result_apply_diff.get('stderr'):
                                    error_info_apply_diff = cat_result_apply_diff.get('error_message', '')
                                    if cat_result_apply_diff.get('stderr'):
                                        api_cat_stderr_apply_diff, _ = truncate_output_lines(cat_result_apply_diff['stderr'], MAX_TOOL_OUTPUT_LINES // 2)
                                        error_info_apply_diff += f" Stderr: {api_cat_stderr_apply_diff}"
                                    modified_file_content_for_api = f"[Error reading modified file: {error_info_apply_diff.strip()}]"

                                tool_result_message_parts = [
                                    f"Tool Result: apply_diff",
                                    f"File: {target_file_apply_diff_arg}",
                                    f"Status: Success",
                                    f"Message: {message}"
                                ]
                                # Verifier comment is None on success from handle_apply_diff
                                if diff_str:
                                    tool_result_message_parts.append(f"Diff:\n```diff\n{diff_str}\n```")
                                tool_result_message_parts.append(f"New Content:\n```\n{modified_file_content_for_api}\n```")
                                tool_result_message = "\n".join(tool_result_message_parts)
                                print(f"  Result: {message}", file=sys.stderr)
                            else:
                                tool_result_message = f"Tool Result: apply_diff\nFile: {target_file_apply_diff_arg}\nStatus: Failure\nMessage: {message}"
                                if verifier_comment:
                                    tool_result_message += f"\nVerifier Comment: {verifier_comment}"
                                print(f"  Result: {message}", file=sys.stderr)
                                if verifier_comment:
                                    print(f"  Verifier Comment: {verifier_comment}", file=sys.stderr)
                        else:
                            # Сообщение об ошибке теперь должно быть более точным
                            missing_for_apply_diff = []
                            if not target_file_apply_diff_arg: missing_for_apply_diff.append("target_file")
                            if not change_snippets_arg:
                                missing_for_apply_diff.append("change_snippets") # Проверяем правильный отсутствующий аргумент
                            # desired_changes_description_arg имеет значение по умолчанию, поэтому его отсутствие не критично для этого условия
                            tool_result_message = f"[Tool Error: apply_diff - Missing arguments: {', '.join(missing_for_apply_diff)}]"
                            print(f"  Error: Missing arguments for apply_diff: {', '.join(missing_for_apply_diff)}.", file=sys.stderr)

                    elif tool_name == "talk_to_o3":
                        message_for_o3 = arguments.get("message_for_o3")
                        if message_for_o3:
                            try:
                                o3_script_abs_path = script_dir / "tools" / "o3.py"
                                python_executable = sys.executable
                                process = subprocess.run(
                                    [python_executable, str(o3_script_abs_path), message_for_o3],
                                    capture_output=True, text=True, cwd=launch_dir, check=False
                                )
                                if process.returncode == 0:
                                    response_from_o3 = process.stdout.strip()
                                    tool_result_message = (
                                        f"Tool Result: talk_to_o3\nStatus: Success\n"
                                        f"Ваш запрос на о3:\n```\n{message_for_o3}\n```\n"
                                        f"Ответ о3 на ваш запрос:\n```\n{response_from_o3}\n```"
                                    )
                                    print(f"  Result: Success. o3 Response (preview):\n{response_from_o3[:200]}...", file=sys.stderr)
                                else:
                                    error_details = f"RC: {process.returncode}"
                                    if process.stderr.strip(): error_details += f", Stderr: {process.stderr.strip()[:100]}..."
                                    if process.stdout.strip(): error_details += f", Stdout: {process.stdout.strip()[:100]}..."
                                    tool_result_message = (
                                        f"Tool Result: talk_to_o3\nStatus: Error\n"
                                        f"Ваш запрос на о3:\n```\n{message_for_o3}\n```\n"
                                        f"Details: {error_details}"
                                    )
                                    print(f"  Result: Error calling o3.py. {error_details}", file=sys.stderr)
                            except Exception as e_o3_call:
                                tool_result_message = f"[Tool Error: talk_to_o3 - Unexpected error during call to o3.py with request:\n```\n{message_for_o3}\n```\nError: {e_o3_call}]"
                                print(f"  Result: Unexpected error during o3 call: {e_o3_call}", file=sys.stderr)
                        else:
                            tool_result_message = "[Tool Error: talk_to_o3 - Missing message_for_o3 argument]"
                            print(f"  Error: Missing message_for_o3 for talk_to_o3.", file=sys.stderr)
                    else:
                        tool_result_message = f"[System Error: Unknown tool '{tool_name}']"
                        print(f"  Error: Unknown tool '{tool_name}'", file=sys.stderr)
                    
                    print("-" * 70, file=sys.stderr)

                    # Добавляем результат инструмента в историю (как user)
                    current_history.append({"role": "user", "parts": [{"text": tool_result_message}]})
                    # --- Сохранение результата инструмента в реальном времени ---
                    save_history(history_path, current_history)
                    # --- Конец сохранения --- 
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
            # history_to_save = [
            #     msg for msg in current_history
            #     if not (msg.get('role') == 'model' and is_known_tool_call(msg.get('parts', [{}])[0].get('text', '')))
            # ]
            # Теперь сохраняем всю историю без фильтрации вызовов инструментов
            history_to_save = current_history
            save_history(history_path, history_to_save)

            # === Вызов Суммаризации ===
            # current_history для суммаризации уже содержит детализированные логи инструментов и вызовы инструментов
            # This is the end-of-turn summarization.
            # It might operate on an already summarized history if the threshold was met earlier.
            # The summarize_history function should ideally handle this gracefully.
            summarized_history_at_end_of_turn = summarize_history(current_history, config, api_key)
            if summarized_history_at_end_of_turn is not None:
                # Save regardless of whether it changed, as summarize_history might have internal reasons to return the same
                # if no summarization was deemed necessary by its internal logic, but we still want to ensure
                # what it decided upon is what's saved.
                if summarized_history_at_end_of_turn != current_history:
                     print(f"GEMINI_PY: End-of-turn summarization updated history from {len(current_history)} to {len(summarized_history_at_end_of_turn)} messages.", file=sys.stderr)
                else:
                    print("GEMINI_PY: No changes made by end-of-turn summarization.", file=sys.stderr)
                save_history(history_path, summarized_history_at_end_of_turn) 
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