"""Shared GrepParameters tool schema for the Neural Grep Transpiler.

Used identically by data_gen.py (training data), cli.py (inference), and
eval_grep.py (scoring) so the tool definition seen by the model at finetune
time matches what it sees at inference time.
"""

import json

TOOL_NAME = "ripgrep_search"

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Search files for matching lines using ripgrep.",
    "parameters": {
        "terms": {
            "type": "array",
            "description": "Literal strings or fully escaped regular expressions to search for.",
            "required": True,
        },
        "is_regex": {
            "type": "boolean",
            "description": "True if terms are regex patterns, false if terms are literal strings.",
            "required": True,
        },
        "case_insensitive": {
            "type": "boolean",
            "description": "True if the search should ignore case.",
            "required": True,
        },
    },
}

TOOLS_JSON = json.dumps([TOOL_SCHEMA], separators=(",", ":"))


def make_answer(terms, is_regex, case_insensitive):
    return [
        {
            "name": TOOL_NAME,
            "arguments": {
                "terms": terms,
                "is_regex": is_regex,
                "case_insensitive": case_insensitive,
            },
        }
    ]


def validate_grep_params(obj):
    """Validate a decoded {"terms":..,"is_regex":..,"case_insensitive":..} dict.

    Returns True/False. Also checks that regex terms actually compile.
    """
    import re

    if not isinstance(obj, dict):
        return False
    terms = obj.get("terms")
    is_regex = obj.get("is_regex")
    case_insensitive = obj.get("case_insensitive")
    if not isinstance(terms, list) or not terms:
        return False
    if not all(isinstance(t, str) and t for t in terms):
        return False
    if not isinstance(is_regex, bool) or not isinstance(case_insensitive, bool):
        return False
    if is_regex:
        for t in terms:
            try:
                re.compile(t)
            except re.error:
                return False
    return True
