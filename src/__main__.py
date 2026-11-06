from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.parser import parse_calling_tests, parse_function_definition
from src.parser import parse_argument_to_dict
from src.prompt_builder import build_prompt, build_argument_prompt
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
    response = run_inference(model, final_prompt).strip()

    print("function that chosen by models\n")
    print(f"{response}\n")

    print("--- argument prompt ---\n")
    arg_prompt = build_argument_prompt(
        first_query, response, functions[response])
    raw_text = run_inference(model, arg_prompt)

    print(arg_prompt)
    print(f"{raw_text}\n")

    parsed_arguments = parse_argument_to_dict(raw_text)
    print(parsed_arguments)


if __name__ == "__main__":
    main()
