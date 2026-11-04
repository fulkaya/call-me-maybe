import json
from typing import List, Any
from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)
        return dict(data)


def run_inference(
        model: Small_LLM_Model, prompt: str, max_new_tokens: int = 64) -> str:
    input_ids_tensor = model.encode(prompt)
    input_ids: List[int] = input_ids_tensor[0].tolist()
    generated_tokens: List[int] = []

    for _ in range(max_new_tokens):
        current_sequence = input_ids + generated_tokens
        logits = model.get_logits_from_input_ids(current_sequence)

        next_token: int = max(range(len(logits)), key=lambda i: logits[i])

        generated_tokens.append(next_token)
        decoded_text = model.decode(generated_tokens)
        if "\n" in decoded_text or "}" in decoded_text:
            break

    return str(model.decode(generated_tokens).strip())
