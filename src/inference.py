"""Inference module for executing autoregressive generation with
Small_LLM_Model.

Handles vocabulary extraction and token-by-token greedy decoding with
early-stopping criteria.
"""

import json
from typing import List, Optional
from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    """Load and invert the tokenizer vocabulary mapping from the model
    directory.

    Args:
        model: An instance of Small_LLM_Model providing access to model assets.

    Returns:
        A dictionary mapping integer token IDs to their string representations.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        data: dict[str, int] = json.load(f)
        return {
            idx: token.replace("Ġ", " ").replace("Ċ", "\n")
            for token, idx in data.items()
        }


def constrained_mask(
    logits: List[float],
    vocab: dict[int, str],
    generated_text: str,
    allowed_functions: Optional[List[str]] = None
) -> List[float]:
    """Set logits of invalid tokens to -inf."""
    masked = list(logits)
    neg_inf = -float("inf")

    if allowed_functions is not None:
        for token_id, token_str in vocab.items():
            if not token_str:
                masked[token_id] = neg_inf
                continue
            candidate = (generated_text + token_str).lstrip()
            if not candidate:
                masked[token_id] = neg_inf
                continue
            if (
                token_str in ("\n", " ", "(")
                and generated_text.strip() in allowed_functions
            ):
                continue
            if not any(fn.startswith(candidate) for fn in allowed_functions):
                masked[token_id] = neg_inf
        return masked

    for token_id, token_str in vocab.items():
        if not token_str or "```" in token_str:
            masked[token_id] = neg_inf
        elif any(ord(ch) < 32 for ch in token_str):
            masked[token_id] = neg_inf
    return masked


def run_inference(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    max_new_tokens: int = 64,
    allowed_functions: Optional[list[str]] = None,
    verbose: bool = False
) -> str:
    """Generate completion text for a given prompt using constrained decoding.

    Encodes input text, masks invalid token logits with -inf, iteratively
    predicts the next token, and stops when max tokens or an exit character
    is encountered.

    Args:
        model: The language model instance used for encoding and logits.
        vocab: Dictionary mapping token IDs to decoded token strings.
        prompt: The formatted input text provided to the model.
        max_new_tokens: Maximum number of tokens to generate before stopping.
        verbose: Whether to stream generated tokens to stdout.
        allowed_functions: Optional list of valid function names to constrain.

    Returns:
        The decoded and stripped generated string.
    """
    input_ids_tensor = model.encode(prompt)
    input_ids: List[int] = input_ids_tensor[0].tolist()
    generated_tokens: List[int] = []

    for _ in range(max_new_tokens):
        current_sequence = input_ids + generated_tokens
        logits = model.get_logits_from_input_ids(current_sequence)
        so_far = (
            str(model.decode(generated_tokens)) if generated_tokens else ""
        )
        logits = constrained_mask(logits, vocab, so_far, allowed_functions)
        next_token: int = max(range(len(logits)), key=lambda i: logits[i])

        generated_tokens.append(next_token)

        if verbose:
            token_text = model.decode([next_token])
            if len(generated_tokens) == 1:
                token_text = token_text.lstrip()
            print(token_text, end="", flush=True)

        decoded_text = model.decode(generated_tokens)
        if allowed_functions is not None:
            if decoded_text.strip() in allowed_functions:
                break
        elif decoded_text.count("}") > decoded_text.count("{"):
            break

    if verbose:
        print()

    return str(model.decode(generated_tokens).strip())
