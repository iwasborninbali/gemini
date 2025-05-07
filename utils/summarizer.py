import requests
import json
import math
import sys
from pathlib import Path # Хотя Path не используется напрямую в этой функции, но может понадобиться для расширения
from typing import List, Dict, Optional, Union # Используем typing для лучшей читаемости

# Импортируем тип истории и новый API клиент
from utils.history_manager import History
from utils.api_client import call_gemini_api, GeminiApiError

# Определяем тип для истории, чтобы было понятнее
HistoryItem = Dict[str, Union[str, List[Dict[str, str]]]]
History = List[HistoryItem]

def summarize_history(history: History, config: dict, api_key: str) -> Optional[History]:
    """Суммаризирует историю чата, если достигнут порог, используя requests."""
    sum_config = config.get('summarization', {})
    if not sum_config.get('enabled', False):
        return None

    threshold = sum_config.get('threshold_messages', 10)
    interaction_messages = [m for m in history if m['role'] in ['user', 'model']]
    if len(interaction_messages) < threshold:
        return None

    summarize_ratio = sum_config.get('summarize_ratio', 0.5)
    keep_recent = sum_config.get('context_keep_messages', 5)
    api_base_url = config.get('api_base_url')
    if not api_base_url:
        print("Ошибка: api_base_url не найден в config.json для суммаризации.", file=sys.stderr)
        return None

    num_interactions = len(interaction_messages)
    num_to_summarize = math.ceil((num_interactions - keep_recent) * summarize_ratio)
    num_to_summarize = max(1, min(num_to_summarize, num_interactions - keep_recent))

    if num_to_summarize <= 0:
        print("--- Недостаточно сообщений для суммаризации после учета keep_recent. ---", file=sys.stderr)
        return None

    messages_to_summarize = interaction_messages[:num_to_summarize]
    messages_to_keep = interaction_messages[-keep_recent:]
    system_messages = [m for m in history if m['role'] == 'system']

    # Формируем текст для запроса на суммаризацию, проверяя структуру parts
    history_text_parts = []
    for msg in messages_to_summarize:
        text_content = "[Содержимое не найдено]"
        parts = msg.get('parts')
        if parts and isinstance(parts, list) and len(parts) > 0:
            if isinstance(parts[0], dict) and 'text' in parts[0]:
                 text_content = parts[0]['text']
            elif isinstance(parts[0], str): # На случай если parts это просто список строк
                 text_content = parts[0]
        history_text_parts.append(f"{msg.get('role', 'unknown')}: {text_content}")
    history_text = "\n".join(history_text_parts)


    try:
        summarizer_model_name = sum_config.get('model_name', 'gemini-1.5-flash-latest')

        # Формируем payload для запроса на суммаризацию
        summary_prompt = f"Please concisely summarize the following conversation history:\n\n{history_text}"
        summarization_payload_contents = [{
            "role": "user",
            "parts": [{'text': summary_prompt}]
        }]

        # Используем новую функцию API клиента
        summary_text = call_gemini_api(
            api_key=api_key,
            api_base_url=api_base_url,
            model_name=summarizer_model_name,
            payload_contents=summarization_payload_contents
        )

        # Если call_gemini_api вернула None или текст ошибки, считаем это неудачей
        # (call_gemini_api теперь возбуждает исключения при серьезных ошибках)
        if summary_text is None or summary_text.startswith("[Ошибка"):
            print(f"Предупреждение: Не удалось получить корректную сводку от API клиента. Результат: {summary_text}", file=sys.stderr)
            return None

        new_history: History = []
        new_history.extend(system_messages)
        new_history.append({
            'role': 'model',
            'parts': [{'text': f"Summary of prior conversation:\n{summary_text}"}]
        })
        new_history.extend(messages_to_keep)

        return new_history

    # Перехватываем специфичные ошибки от API клиента или requests
    except GeminiApiError as api_err:
        print(f"Ошибка API во время суммаризации: {api_err}", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as req_err:
        # Ошибка сети уже логируется в call_gemini_api, здесь просто возвращаем None
        # print(f"Ошибка сети во время вызова API суммаризации: {req_err}", file=sys.stderr)
        return None
    except Exception as e:
        # Обработка других неожиданных ошибок, которые могли произойти здесь
        import traceback
        print(f"Неожиданная ошибка в summarize_history: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return None 