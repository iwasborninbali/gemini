import re
import sys
from typing import Optional, Dict, Tuple, Any

# Паттерны для определения ТИПА инструмента
EXECUTE_TERMINAL_PATTERN = re.compile(r"(['\"]?)tool\1?\s*:\s*(['\"]?)execute_terminal_command\2", re.IGNORECASE)
CREATE_FILE_PATTERN = re.compile(r"(['\"]?)tool\1?\s*:\s*(['\"]?)create_file\2", re.IGNORECASE)
APPLY_DIFF_PATTERN = re.compile(r"(['\"]?)tool\1?\s*:\s*(['\"]?)apply_diff\2", re.IGNORECASE)
TALK_TO_O3_PATTERN = re.compile(r"(['\"]?)tool\1?\s*:\s*(['\"]?)talk_to_o3\2", re.IGNORECASE)

# Паттерны для извлечения аргументов (более гибкие)
# Ищет ключ (target_file или command), двоеточие, кавычку, захватывает значение, кавычка
COMMAND_ARG_PATTERN = re.compile(r"(['\"]?)command\1?\s*:\s*(['\"])(.*?)\2", re.IGNORECASE | re.DOTALL)
TARGET_FILE_ARG_PATTERN = re.compile(r"(['\"]?)target_file\1?\s*:\s*(['\"])(.*?)\2", re.IGNORECASE | re.DOTALL)
# Ищет ключ content, двоеточие, кавычку, захватывает ВСЕ до ПОСЛЕДНЕЙ такой же кавычки (.*?) перед опциональными другими ключами или концом блока/файла
# Это не идеально, но должно работать для большинства случаев. DOTALL позволяет `.` матчить новые строки.
CONTENT_ARG_PATTERN = re.compile(r"(['\"]?)content\1?\s*:\s*(['\"])(.*?)\2\s*[,}]?$", re.IGNORECASE | re.DOTALL | re.MULTILINE)
# Для apply_diff нужен diff_content вместо content
DIFF_CONTENT_ARG_PATTERN = re.compile(r"(['\"]?)diff_content\1?\s*:\s*(['\"])(.*?)\2\s*[,}]?$", re.IGNORECASE | re.DOTALL | re.MULTILINE)
MESSAGE_FOR_O3_ARG_PATTERN = re.compile(r"(['\"]?)message_for_o3\1?\s*:\s*(['\"])(.*?)\2\s*[,}]?$", re.IGNORECASE | re.DOTALL | re.MULTILINE)

# List of known tool names
KNOWN_TOOLS = {"execute_terminal_command", "create_file", "apply_diff", "talk_to_o3"}

def parse_tool_call(text: str, verbose: bool = True) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Парсит текст на вызов инструмента и извлекает аргументы.

    Returns:
        Кортеж (имя_инструмента, словарь_аргументов) или None.
    """
    if EXECUTE_TERMINAL_PATTERN.search(text):
        tool_name = "execute_terminal_command"
        # Ищем аргумент command
        command_match = COMMAND_ARG_PATTERN.search(text)
        if command_match:
            command = command_match.group(3)
            if verbose:
                print(f"PARSER: Обнаружен '{tool_name}', команда: {repr(command)}", file=sys.stderr)
            return tool_name, {"command": command}
        else:
            if verbose:
                print(f"PARSER: Обнаружен '{tool_name}', но не найден аргумент 'command'", file=sys.stderr)
            return None # Ошибка парсинга аргументов

    elif CREATE_FILE_PATTERN.search(text):
        tool_name = "create_file"
        # Ищем аргументы target_file и content
        target_match = TARGET_FILE_ARG_PATTERN.search(text)
        # Для content ищем с конца текста, чтобы лучше обработать многострочность
        content_match = CONTENT_ARG_PATTERN.search(text)

        if target_match and content_match:
            target_file = target_match.group(3).strip() # Убираем лишние пробелы по краям
            content = content_match.group(3) # Не strip() контент, пробелы могут быть важны

            # Простая проверка, что не захватили лишнего (например, следующий ключ)
            # Эту проверку можно усложнить при необходимости
            # if '}' in content or 'tool:' in content:
                 # if verbose: print(f"PARSER: Warning - Potential over-capture in content for '{target_file}'", file=sys.stderr)
                 # pass # Пока разрешаем

            if verbose:
                print(f"PARSER: Обнаружен '{tool_name}', target: '{target_file}', content_len: {len(content)}", file=sys.stderr)
            return tool_name, {"target_file": target_file, "content": content}
        else:
            missing_args = []
            if not target_match: missing_args.append("target_file")
            if not content_match: missing_args.append("content")
            if verbose:
                print(f"PARSER: Обнаружен '{tool_name}', но не найден(ы) аргумент(ы): {', '.join(missing_args)}", file=sys.stderr)
            return None # Ошибка парсинга аргументов

    elif APPLY_DIFF_PATTERN.search(text):
        tool_name = "apply_diff"
        target_match = TARGET_FILE_ARG_PATTERN.search(text)
        diff_content_match = DIFF_CONTENT_ARG_PATTERN.search(text)

        if target_match and diff_content_match:
            target_file = target_match.group(3).strip()
            diff_content = diff_content_match.group(3) # Не strip()
            if verbose:
                print(f"PARSER: Обнаружен '{tool_name}', target: '{target_file}', diff_content_len: {len(diff_content)}", file=sys.stderr)
            return tool_name, {"target_file": target_file, "diff_content": diff_content}
        else:
            missing_args = []
            if not target_match: missing_args.append("target_file")
            if not diff_content_match: missing_args.append("diff_content")
            if verbose:
                print(f"PARSER: Обнаружен '{tool_name}', но не найден(ы) аргумент(ы): {', '.join(missing_args)}", file=sys.stderr)
            return None # Ошибка парсинга аргументов

    elif TALK_TO_O3_PATTERN.search(text):
        tool_name = "talk_to_o3"
        message_match = MESSAGE_FOR_O3_ARG_PATTERN.search(text)

        if message_match:
            message_for_o3 = message_match.group(3) # Не strip()
            if verbose:
                print(f"PARSER: Обнаружен '{tool_name}', message_len: {len(message_for_o3)}", file=sys.stderr)
            return tool_name, {"message_for_o3": message_for_o3}
        else:
            if verbose:
                print(f"PARSER: Обнаружен '{tool_name}', но не найден аргумент 'message_for_o3'", file=sys.stderr)
            return None

    return None # Инструмент не распознан

def is_known_tool_call(text: str) -> bool:
     """Проверяет, похож ли текст ответа модели на вызов известного инструмента."""
     return EXECUTE_TERMINAL_PATTERN.search(text) is not None or \
            CREATE_FILE_PATTERN.search(text) is not None or \
            APPLY_DIFF_PATTERN.search(text) is not None or \
            TALK_TO_O3_PATTERN.search(text) is not None 