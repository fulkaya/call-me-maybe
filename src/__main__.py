from pathlib import Path
from src.parser import parse_calling_tests, parse_function_definition
from src.prompt_builder import build_prompt


def main() -> None:
    functions_path = Path("data/input/function_definition.json")
    tests_path = Path("data/input/function_calling_tests.json")

    functions = parse_function_definition(functions_path)
    tests = parse_calling_tests(tests_path)

    first_query = tests[0]["prompt"]
    final_prompt = build_prompt(first_query, functions)

    print(final_prompt)


if __name__ == "__main__":
    main()
