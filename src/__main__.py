from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.parser import parse_calling_tests, parse_function_definition
from src.prompt_builder import build_prompt
from src.inference import run_inference


def main() -> None:
    functions_path = Path("data/input/function_definition.json")
    tests_path = Path("data/input/function_calling_tests.json")

    functions = parse_function_definition(functions_path)
    tests = parse_calling_tests(tests_path)

    first_query = tests[0]["prompt"]
    final_prompt = build_prompt(first_query, functions)

    print("--- prompt ---\n")
    print(f"{final_prompt}\n")

    print("model is loading...\n")
    model = Small_LLM_Model()
    response = run_inference(model, final_prompt)

    print("function that chosen by models\n")
    print(response)


if __name__ == "__main__":
    main()
