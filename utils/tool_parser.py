import re
import sys
import json
from typing import Optional, Dict, Tuple, Any

# Предполагаем, что json5 и PyYAML установлены в окружении
import yaml
import json5

# ---------------------------------------------------------------------------
#  PATTERNS FOR QUICK INTENT DETECTION
# ---------------------------------------------------------------------------
EXECUTE_TERMINAL_PATTERN = re.compile(r"(['\"]?)tool\1?\s*:\s*(['\"]?)execute_terminal_command\2", re.IGNORECASE)
CREATE_FILE_PATTERN      = re.compile(r"(['\"]?)tool\1?\s*:\s*(['\"]?)create_file\2",              re.IGNORECASE)
APPLY_DIFF_PATTERN       = re.compile(r"(['\"]?)tool\1?\s*:\s*(['\"]?)apply_diff\2",               re.IGNORECASE)
TALK_TO_O3_PATTERN       = re.compile(r"(['\"]?)tool\1?\s*:\s*(['\"]?)talk_to_o3\2",               re.IGNORECASE)

# Simple arg extractor for execute_terminal_command
COMMAND_ARG_PATTERN = re.compile(r"(['\"]?)command\1?\s*:\s*(['\"])(.*?)\2\s*[,}]?", re.IGNORECASE | re.DOTALL)

KNOWN_TOOLS = {"execute_terminal_command", "create_file", "apply_diff", "talk_to_o3"}

# ---------------------------------------------------------------------------
#  RESILIENT JSON(ish) LOADER
# ---------------------------------------------------------------------------

def _extract_first_braced_block(text: str) -> str | None:
    """Return first {...} block with balanced braces or None."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _smart_load_jsonish(blob: str) -> dict | None:
    """Attempt to parse *something* like JSON/YAML returning dict or None."""
    # strip triple‑backtick fences first
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", blob.strip(), flags=re.I | re.S).strip()
    loaders = [json.loads, json5.loads, yaml.safe_load]

    for loader in loaders:
        try:
            obj = loader(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # try again on first {...} blob only
    braced = _extract_first_braced_block(cleaned)
    if braced:
        for loader in loaders:
            try:
                obj = loader(braced)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None

# ---------------------------------------------------------------------------
#  MAIN PARSER
# ---------------------------------------------------------------------------

def parse_tool_call(text: str, verbose: bool = True) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Detect tool call and return (tool_name, arguments) or None."""
    if verbose:
        snippet = text[:400].replace("\n", "<LF>")
        print(f"PARSER_INPUT_DEBUG: {snippet}...", file=sys.stderr)

    # ---------- Fast path for execute_terminal_command ----------
    if EXECUTE_TERMINAL_PATTERN.search(text):
        cmd_match = COMMAND_ARG_PATTERN.search(text)
        if cmd_match:
            if verbose:
                print("PARSER: 'execute_terminal_command' detected.", file=sys.stderr)
            return "execute_terminal_command", {"command": cmd_match.group(3)}
        if verbose:
            print("PARSER: command arg not found.", file=sys.stderr)
        return None

    # ---------- Generic JSON‑first attempt for other tools ----------
    tool_specs = [
        ("create_file", CREATE_FILE_PATTERN, ["target_file", "content"]),
        (
            "apply_diff",
            APPLY_DIFF_PATTERN,
            ["target_file", "desired_changes_description", ["change_snippets", "diff_content"]],
        ),
        ("talk_to_o3", TALK_TO_O3_PATTERN, ["message_for_o3"]),
    ]

    for name, quick_pat, required in tool_specs:
        if not quick_pat.search(text):
            continue

        if verbose:
            print(f"PARSER: '{name}' quick pattern hit – trying smart load.", file=sys.stderr)

        args = _smart_load_jsonish(text)
        if args is not None and args.get("tool") == name:
            # validate required keys (supporting alternatives)
            missing = []
            for spec in required:
                if isinstance(spec, list):
                    if not any(k in args for k in spec):
                        missing.append("/".join(spec))
                elif spec not in args:
                    missing.append(spec)
            if missing:
                if verbose:
                    print(f"PARSER: missing keys {missing} after smart load.", file=sys.stderr)
                # fall through to regex fallback later
            else:
                # unify field name
                if "change_snippets" not in args and "diff_content" in args:
                    args["change_snippets"] = args.pop("diff_content")
                return name, args
        else:
            if verbose:
                print("PARSER: smart load failed or 'tool' mismatch.", file=sys.stderr)

    # ---------- Regex fallback ONLY for apply_diff ----------
    if APPLY_DIFF_PATTERN.search(text):
        if verbose:
            print("PARSER: applying regex fallback for apply_diff.", file=sys.stderr)
        tgt  = re.search(r'["\']target_file["\']\s*:\s*["\']([^"\']+)["\']', text)
        desc = re.search(r'["\']desired_changes_description["\']\s*:\s*["\']([^"\']+)["\']', text)
        diff = re.search(r'["\'](?:change_snippets|diff_content)["\']\s*:\s*(.*?)(?=\n?\s*}[,\s]*$)', text, re.S)

        if tgt and desc and diff:
            diff_val = diff.group(1).strip()
            # strip common fences/quotes
            for fence in ("```", "'''", '"""', '"', "'"):
                if diff_val.startswith(fence) and diff_val.endswith(fence):
                    diff_val = diff_val[len(fence):-len(fence)].strip()
                    break
            if verbose:
                print("PARSER: regex fallback succeeded.", file=sys.stderr)
            return "apply_diff", {
                "tool": "apply_diff",
                "target_file": tgt.group(1),
                "desired_changes_description": desc.group(1),
                "change_snippets": diff_val,
            }

    if verbose:
        print("PARSER: no tool detected.", file=sys.stderr)
    return None


def is_known_tool_call(text: str) -> bool:
    """Quick boolean check for any tool signature in text."""
    return any(pat.search(text) for pat in (
        EXECUTE_TERMINAL_PATTERN,
        CREATE_FILE_PATTERN,
        APPLY_DIFF_PATTERN,
        TALK_TO_O3_PATTERN,
    ))
