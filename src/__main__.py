"""Main entry point for the function calling pipeline.

Coordinates CLI parsing, prompt generation, LLM inference, argument
extraction, and structured output formatting.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, ValidationError

from llm_sdk import Small_LLM_Model
from src.parser import parse_calling_tests, parse_function_definition
from src.parser import parse_argument_to_dict
from src.prompt_builder import build_prompt, build_argument_prompt
from src.inference import run_inference


class FunctionCallResult(BaseModel):
    """Represents a validated function call execution result.

    Attributes:
        prompt: The raw natural language input prompt.
        name: Name of the identified target function.
        parameters: Dictionary containing validated argument names and values.
    """
    prompt: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for input and output configuration.

    Returns:
        Parsed CLI arguments containing file paths for function definitions,
        test queries, and target output.
    """
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Visualize prompt and generation process.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full function calling pipeline and save results to JSON.

    Reads test queries and function definitions, runs two-step inference
    (function selection followed by parameter extraction), validates
    results via Pydantic, and writes output to disk.
    """
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
        if args.verbose:
            print(f"\n--- Function Prompt ---\n{fn_prompt}")
        chosen_fn = run_inference(
            model, fn_prompt, verbose=args.verbose).strip()
        chosen_fn = chosen_fn.split("(")[0].strip()

        arg_prompt = build_argument_prompt(
            query, chosen_fn, functions[chosen_fn])
        if args.verbose:
            print(f"\n--- Argument Prompt ---\n{arg_prompt}")
        raw_text = run_inference(model, arg_prompt, verbose=args.verbose)
        parsed_args = parse_argument_to_dict(
            raw_text, functions[chosen_fn]["parameters"])

        try:
            call_result = FunctionCallResult(
                prompt=query,
                name=chosen_fn,
                parameters=parsed_args,
            )
            results.append(call_result.model_dump())
        except ValidationError as e:
            print(
                f"Warning: Validation error for prompt '{query}': {e}",
                file=sys.stderr
            )
            continue

        print(f"-> Chosen Function: {chosen_fn}")
        print(f"-> Parameters: {parsed_args}\n")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    except OSError as e:
        print(
            f"Error: Could not write output to {output_path}: {e}",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
