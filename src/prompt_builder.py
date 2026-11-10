"""Prompt construction utilities for two-stage function calling.

Builds structured textual prompts for LLM inference to identify the target
function and extract argument values into JSON format.
"""

from typing import Any


def build_prompt(query: str, parsed_data: dict[str, Any]) -> str:
    """Construct a prompt listing available functions to select the target one.

    Formats the schema and signature of each function, appends the user
    request, and guides the model to return solely the matching function name.

    Args:
        query: Natural language user instruction.
        parsed_data: Mapping of function names to their metadata containing
            descriptions and parameter schemas.

    Returns:
        The fully formatted prompt string for function classification.
    """
    prompt = "You have access to the following functions:\n"

    for fn_name, fn_info in parsed_data.items():
        desc = fn_info["description"]
        params = fn_info["parameters"]

        param_list = []
        for p_name, p_type in params.items():
            param_list.append(f"{p_name}: {p_type}")
        param_str = ", ".join(param_list)

        prompt += f"- {fn_name}({param_str}): {desc}\n"

    prompt += (
        "\nUse the functions above to answer the user request. "
        "Output only the chosen function name.\n"
    )
    prompt += f"User Request: {query}\n"
    prompt += "Chosen function: "

    return prompt


def build_argument_prompt(
        query: str, fn_name: str, fn_info: dict[str, Any]) -> str:
    """Construct a prompt to extract function arguments from a query.

    Provides the target function schema and constraints, prefilling the JSON
    prefix to guide the model into outputting key-value argument pairs.

    Args:
        query: Natural language user instruction containing argument values.
        fn_name: Name of the selected target function.
        fn_info: Dictionary containing the selected function's description
            and parameter definitions.

    Returns:
        The formatted prompt string prompting the model for argument JSON.
    """
    desc = fn_info["description"]
    params = fn_info["parameters"]

    prompt = (
        f"Function: {fn_name}\n"
        f"Description: {desc}\n"
        f"Parameters: {params}\n"
        f"User query: {query}\n"
        "Extract the exact input arguments from the user query. "
        "Do NOT execute the function or transform the values. "
        "Use only the parameter names defined above.\n"
        f"Output arguments as JSON:\n"
        'Arguments: {"'
    )
    return prompt
