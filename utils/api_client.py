import requests
import json
import sys
from typing import List, Dict, Optional, Any

# Импортируем тип истории из history_manager
from utils.history_manager import History

class GeminiApiError(Exception):
    """Custom exception for API related errors."""
    pass

def call_gemini_api(
    api_key: str,
    api_base_url: str,
    model_name: str,
    payload_contents: List[Dict[str, Any]], # Принимаем только "contents"
    timeout: int = 90,
    system_prompt: Optional[str] = None # Новый опциональный аргумент
) -> Optional[str]:
    """Выполняет вызов к Gemini API (generateContent) и возвращает текстовый ответ.

    Args:
        api_key: Ключ API.
        api_base_url: Базовый URL API (https://generativelanguage.googleapis.com/v1beta/models).
        model_name: Имя модели.
        payload_contents: Список сообщений для поля "contents" в JSON payload.
        timeout: Таймаут запроса в секундах.
        system_prompt: Опциональный системный промпт.

    Returns:
        Текстовый ответ модели или None в случае ошибки.

    Raises:
        GeminiApiError: Если произошла ошибка API (неверный ключ, модель и т.д.).
        requests.exceptions.RequestException: Если произошла сетевая ошибка.
    """
    api_url = f"{api_base_url}/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": payload_contents}

    # Добавляем system_instruction, если передан system_prompt
    if system_prompt:
        payload["system_instruction"] = {
             # API ожидает role/parts даже для system_instruction
            "role": "system", # Или можно не указывать? Документация неясна, начнем без роли.
             "parts": [{'text': system_prompt}]
        }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)

        # Проверка на ошибки HTTP (4xx, 5xx)
        response.raise_for_status()

        response_json = response.json()

        # Безопасное извлечение текста ответа
        candidates = response_json.get('candidates')
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            content = candidates[0].get('content')
            if content and isinstance(content, dict):
                parts = content.get('parts')
                if parts and isinstance(parts, list) and len(parts) > 0:
                     if isinstance(parts[0], dict) and 'text' in parts[0]:
                          return parts[0].get('text')

        # Если текст не найден в ожидаемой структуре
        print(f"Предупреждение: Не удалось извлечь текст из ответа API. Ответ: {response_json}", file=sys.stderr)
        # Возможно, стоит возбудить исключение, т.к. ответ некорректен
        # raise GeminiApiError(f"Не удалось извлечь текст из ответа API: {response_json}")
        return "[Ошибка: Некорректный формат ответа API]"

    except requests.exceptions.HTTPError as http_err:
        # Обработка специфических HTTP ошибок
        print(f"Ошибка HTTP при вызове API: {http_err}", file=sys.stderr)
        # Попытка извлечь детали ошибки из тела ответа, если возможно
        try:
            error_details = response.json()
            print(f"Детали ошибки API: {error_details}", file=sys.stderr)
            raise GeminiApiError(f"Ошибка API: {error_details}") from http_err
        except json.JSONDecodeError:
            print(f"Тело ответа при ошибке HTTP: {response.text}", file=sys.stderr)
            raise GeminiApiError(f"Ошибка HTTP {response.status_code}") from http_err
        except Exception as e:
            # На случай других проблем при обработке ошибки
            print(f"Дополнительная ошибка при обработке HTTPError: {e}", file=sys.stderr)
            raise GeminiApiError(f"Ошибка HTTP {response.status_code}") from http_err

    except requests.exceptions.RequestException as req_err:
        # Сетевые ошибки, таймауты и т.д.
        print(f"Ошибка сети или подключения при вызове API: {req_err}", file=sys.stderr)
        raise # Передаем исключение дальше

    except json.JSONDecodeError:
        # Ошибка парсинга JSON ответа (даже если статус 200 OK)
        resp_text = response.text if 'response' in locals() and hasattr(response, 'text') else "[Нет текста ответа]"
        print(f"Ошибка декодирования JSON от API: {resp_text}", file=sys.stderr)
        raise GeminiApiError(f"Ошибка декодирования JSON: {resp_text}")

    except Exception as e:
        # Другие непредвиденные ошибки
        print(f"Неожиданная ошибка в API клиенте: {e}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        raise GeminiApiError(f"Неожиданная ошибка клиента API: {e}") 