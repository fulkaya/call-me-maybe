"""Inference module for executing autoregressive generation with
Small_LLM_Model.

Handles vocabulary extraction and token-by-token greedy decoding with
early-stopping criteria.
"""

import json
from typing import List
from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """Load and parse the tokenizer vocabulary mapping from the model
    directory.

    Args:
        model: An instance of Small_LLM_Model providing access to model assets.

    Returns:
        A dictionary mapping token strings to their corresponding integer IDs.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        data: dict[str, int] = json.load(f)
        return dict(data)


def run_inference(
        model: Small_LLM_Model, prompt: str, max_new_tokens: int = 64,
        verbose: bool = False) -> str:
    """Generate completion text for a given prompt using greedy decoding.

    Encodes input text, iteratively predicts the next token with the highest
    logit probability, and stops when max tokens are reached or an exit
    character (newline or closing brace) is encountered.

    Args:
        model: The language model instance used for encoding and logits.
        prompt: The formatted input text provided to the model.
        max_new_tokens: Maximum number of tokens to generate before stopping.

    Returns:
        The decoded and stripped generated string.
    """
    input_ids_tensor = model.encode(prompt)
    input_ids: List[int] = input_ids_tensor[0].tolist()
    generated_tokens: List[int] = []

    for _ in range(max_new_tokens):
        current_sequence = input_ids + generated_tokens
        logits = model.get_logits_from_input_ids(current_sequence)

        next_token: int = max(range(len(logits)), key=lambda i: logits[i])

        generated_tokens.append(next_token)

        if verbose:
            token_text = model.decode([next_token])
            if len(generated_tokens) == 1:
                token_text = token_text.lstrip()
            print(token_text, end="", flush=True)

        decoded_text = model.decode(generated_tokens)
        if "\n" in decoded_text or "}" in decoded_text:
            break

    if verbose:
        print()

    return str(model.decode(generated_tokens).strip())
