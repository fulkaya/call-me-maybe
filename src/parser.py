import json
from pathlib import Path


def parse_function_definition(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

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
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
