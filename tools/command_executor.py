import subprocess
import sys
import os
import shlex
import threading
import select
import time
from typing import Dict, List, Optional

MAX_OUTPUT_BYTES = 1024 * 1024  # 1MB limit
POLL_INTERVAL = 0.1
READ_CHUNK_SIZE = 4096

# Default return structure for errors
DEFAULT_ERROR_RETURN = {
    "status": "error",
    "exit_code": None,
    "stdout": "",
    "stderr": "",
    "truncated": False,
}

def _read_stream(stream, buffer: List[bytes], limit: int, verbose: bool = True) -> bool:
    """Reads from a non-blocking stream into a buffer, respects limit. Returns True if truncated."""
    truncated_in_this_call = False
    try:
        while True:
            current_bytes_in_buffer = sum(len(c) for c in buffer)
            remaining_capacity = limit - current_bytes_in_buffer
            if remaining_capacity <= 0:
                ready_to_read, _, _ = select.select([stream], [], [], 0)
                if ready_to_read:
                    truncated_in_this_call = True
                    # if verbose: print(f"Предупреждение: Буфер потока ({stream.name}) полон. Возможно усечение.", file=sys.stderr)
                break

            read_size = min(READ_CHUNK_SIZE, remaining_capacity)
            try:
                 chunk = os.read(stream.fileno(), read_size)
            except BlockingIOError:
                 break # Ничего не готово для чтения
            except (BrokenPipeError, OSError):
                 break # Поток закрыт

            if not chunk:
                break # End of stream

            buffer.append(chunk)

            if len(chunk) == read_size and remaining_capacity - len(chunk) == 0:
                ready_to_read, _, _ = select.select([stream], [], [], 0)
                if ready_to_read:
                    truncated_in_this_call = True
                    # if verbose: print(f"Предупреждение: Достигнут лимит буфера ({limit} байт) для {stream.name}. Возможно усечение.", file=sys.stderr)
                    break

    except Exception as e:
        if verbose: # Added verbose check
            print(f"Неожиданная ошибка чтения потока: {e}", file=sys.stderr)

    # Final check
    if not truncated_in_this_call and sum(len(c) for c in buffer) >= limit:
        ready_to_read, _, _ = select.select([stream], [], [], 0)
        if ready_to_read:
            truncated_in_this_call = True
            # if verbose: print(f"Предупреждение: Буфер {stream.name} полон после чтения. Возможно усечение.", file=sys.stderr)

    return truncated_in_this_call

def run_command(
    command_str: str,
    cwd: Optional[str] = None,
    interrupt_event: Optional[threading.Event] = None,
    verbose: bool = True # Added verbose
) -> Dict:
    """Executes a terminal command using Popen, returns dict result."""
    if interrupt_event is None:
        interrupt_event = threading.Event() # Dummy event if not provided

    if not command_str:
        return {**DEFAULT_ERROR_RETURN, "error_message": "Пустая команда."}

    try:
        command_parts = shlex.split(command_str)
        if not command_parts:
            return {**DEFAULT_ERROR_RETURN, "error_message": "Пустая команда после парсинга."}
    except ValueError as e:
        return {**DEFAULT_ERROR_RETURN, "error_message": f"Ошибка парсинга команды: {e}"}

    effective_cwd = cwd or os.getcwd()
    if verbose: # Added verbose check
        print(f"ИСПОЛНЕНИЕ: Запуск команды {repr(command_str)} в {effective_cwd}", file=sys.stderr)

    process = None
    stdout_truncated = False
    stderr_truncated = False
    try:
        process = subprocess.Popen(
            command_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=effective_cwd,
            text=False, # Работаем с байтами
            bufsize=0,
            # stdin=subprocess.PIPE # stdin пока не поддерживаем
        )

        stdout_buffer: List[bytes] = []
        stderr_buffer: List[bytes] = []

        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)

        while process.poll() is None:
            if interrupt_event.is_set():
                if verbose: # Added verbose check
                    print("ИСПОЛНЕНИЕ: Получен сигнал прерывания. Завершение команды...", file=sys.stderr)
                try:
                    process.terminate()
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    if verbose: # Added verbose check
                        print("ИСПОЛНЕНИЕ: Процесс не завершился штатно. Принудительное завершение.", file=sys.stderr)
                    process.kill()
                except Exception as term_err:
                    if verbose: # Added verbose check
                        print(f"ИСПОЛНЕНИЕ: Ошибка при завершении процесса: {term_err}", file=sys.stderr)

                # Читаем остатки после попытки завершения
                stdout_truncated = _read_stream(process.stdout, stdout_buffer, MAX_OUTPUT_BYTES, verbose=verbose) or stdout_truncated
                stderr_truncated = _read_stream(process.stderr, stderr_buffer, MAX_OUTPUT_BYTES, verbose=verbose) or stderr_truncated

                stdout_bytes = b"".join(stdout_buffer)
                stderr_bytes = b"".join(stderr_buffer)
                stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                stderr_str = stderr_bytes.decode("utf-8", errors="replace")

                return {
                    "status": "error",
                    "exit_code": process.poll(),
                    "error_message": "Команда прервана.",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "truncated": stdout_truncated or stderr_truncated,
                }

            stdout_read_truncated = _read_stream(process.stdout, stdout_buffer, MAX_OUTPUT_BYTES, verbose=verbose)
            stderr_read_truncated = _read_stream(process.stderr, stderr_buffer, MAX_OUTPUT_BYTES, verbose=verbose)
            stdout_truncated = stdout_truncated or stdout_read_truncated
            stderr_truncated = stderr_truncated or stderr_read_truncated

            time.sleep(POLL_INTERVAL)

        # Процесс завершился сам, читаем остатки
        stdout_read_truncated = _read_stream(process.stdout, stdout_buffer, MAX_OUTPUT_BYTES, verbose=verbose)
        stderr_read_truncated = _read_stream(process.stderr, stderr_buffer, MAX_OUTPUT_BYTES, verbose=verbose)
        stdout_truncated = stdout_truncated or stdout_read_truncated
        stderr_truncated = stderr_truncated or stderr_read_truncated

        exit_code = process.returncode
        stdout_bytes = b"".join(stdout_buffer)
        stderr_bytes = b"".join(stderr_buffer)
        stdout_str = stdout_bytes.decode("utf-8", errors="replace")
        stderr_str = stderr_bytes.decode("utf-8", errors="replace")

        if verbose: # Added verbose check
            print(f"ИСПОЛНЕНИЕ: Команда завершилась с кодом {exit_code}. stdout: {len(stdout_bytes)} байт, stderr: {len(stderr_bytes)} байт.", file=sys.stderr)

        final_truncated_flag = stdout_truncated or stderr_truncated

        if exit_code == 0:
            return {
                "status": "success",
                "exit_code": exit_code,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "truncated": final_truncated_flag,
            }
        else:
            return {
                "status": "error",
                "exit_code": exit_code,
                "error_message": f"Команда завершилась с ошибкой (код: {exit_code})",
                "stdout": stdout_str,
                "stderr": stderr_str,
                "truncated": final_truncated_flag,
            }

    except FileNotFoundError:
        if verbose: # Added verbose check
            print(f"ИСПОЛНЕНИЕ: Ошибка - команда не найдена: {command_parts[0]}", file=sys.stderr)
        return {
            **DEFAULT_ERROR_RETURN,
            "error_message": f"Команда не найдена: {command_parts[0]}",
        }
    except Exception as e:
        if verbose: # Added verbose check
            print(f"ИСПОЛНЕНИЕ: Неожиданная ошибка при выполнении команды: {e}", file=sys.stderr)
        if process and process.poll() is None:
             try: process.kill() # Пытаемся убить процесс при ошибке
             except Exception: pass
        return {
            **DEFAULT_ERROR_RETURN,
            "error_message": f"Неожиданная ошибка выполнения: {e}",
            "stderr": str(e), # Добавляем текст ошибки в stderr
        }
    finally:
        # Закрываем потоки, если процесс был создан
        if process:
            for stream in [process.stdout, process.stderr]: # process.stdin не открывали
                if stream:
                    try: stream.close()
                    except Exception: pass
            try: process.wait(timeout=0.1) # Ждем завершения
            except Exception: pass 