import json
import argparse
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from llm_sdk import Small_LLM_Model
from src.parser import parse_calling_tests, parse_function_definition
from src.parser import parse_argument_to_dict
from src.prompt_builder import build_prompt, build_argument_prompt
from src.inference import run_inference


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json")
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    functions_path: Path = args.functions_definition
    tests_path: Path = args.input
    output_path: Path = args.output

    functions = parse_function_definition(functions_path)
    tests = parse_calling_tests(tests_path)

    print("model is loading...\n")
    model = Small_LLM_Model()

    results: list[dict[str, Any]] = []

    for test in tests:
        query = test["prompt"]
        print(f"Processing: {query}")

        fn_prompt = build_prompt(query, functions)
        chosen_fn = run_inference(model, fn_prompt).strip()
        chosen_fn = chosen_fn.split("(")[0].strip()

        arg_prompt = build_argument_prompt(
            query, chosen_fn, functions[chosen_fn])
        raw_text = run_inference(model, arg_prompt)
        parsed_args = parse_argument_to_dict(
            raw_text, functions[chosen_fn]["parameters"])

        call_result = FunctionCallResult(
            prompt=query,
            name=chosen_fn,
            parameters=parsed_args,
        )
        results.append(call_result.model_dump())
        print(f"-> Chosen Function: {chosen_fn}")
        print(f"-> Parameters: {parsed_args}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
