import json
import sys
from pathlib import Path
from typing import List, cast, Any


def parse_function_definition(file_path: Path) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
    except FileNotFoundError:
        print(
            f"Error: Function definitions file not found: {file_path}",
            file=sys.stderr
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    parsed_data = {}
    for item in raw_list:
        fn_name = item["name"]
        fn_desc = item["description"]

        clean_params = {}
        for param_name, param_info in item.get("parameters", {}).items():
            clean_params[param_name] = param_info["type"]

        parsed_data[fn_name] = {
            "description": fn_desc,
            "parameters": clean_params,
        }

    return parsed_data


def parse_calling_tests(file_path: Path) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return cast(List[dict], json.load(f))
    except FileNotFoundError:
        print(
            f"Error: Input tests file not found: {file_path}",
            file=sys.stderr
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_argument_to_dict(
        raw_text: str, expected_params: dict[str, str]) -> dict[str, Any]:
    text = raw_text.strip()

    if not text.startswith("{"):
        if not text.startswith('"'):
            text = '{"' + text
        else:
            text = '{' + text

    if not text.endswith("}"):
        text = text + "}"

    try:
        result: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        try:
            result = json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            return {}

    for key, val in result.items():
        excepted_type = expected_params.get(key)
        if excepted_type == "number":
            try:
                result[key] = float(val)
            except (ValueError, TypeError):
                pass

    return result
