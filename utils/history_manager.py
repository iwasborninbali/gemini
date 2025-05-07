import json
import sys
from pathlib import Path
from typing import List, Dict, Union

# Определяем тип для истории здесь, т.к. обе функции его используют
HistoryItem = Dict[str, Union[str, List[Dict[str, str]]]]
History = List[HistoryItem]

def load_history(history_path: Path) -> History:
    """Загружает историю из JSON файла."""
    if history_path.exists():
        try:
            with history_path.open('r', encoding='utf-8') as f:
                content = f.read()
                if not content:
                    return []
                history = json.loads(content)
                if isinstance(history, list) and all(isinstance(msg, dict) and 'role' in msg and 'parts' in msg for msg in history):
                     return history
                else:
                     print(f"Предупреждение: Некорректный формат истории в {history_path}. Начинаем новую историю.", file=sys.stderr)
                     return []
        except json.JSONDecodeError:
            print(f"Предупреждение: Ошибка декодирования JSON в {history_path}. Начинаем новую историю.", file=sys.stderr)
            return []
        except Exception as e:
            print(f"Предупреждение: Не удалось загрузить историю из {history_path}: {e}. Начинаем новую историю.", file=sys.stderr)
            return []
    return []

def save_history(history_path: Path, history: History):
    """Сохраняет историю в JSON файл."""
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open('w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Ошибка сохранения истории в {history_path}: {e}", file=sys.stderr) 