def build_prompt(query: str, parsed_data: dict) -> str:
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
