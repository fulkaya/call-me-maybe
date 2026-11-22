*This project has been created as part of the 42 curriculum by fulkaya.*

# Call Me Maybe

## Description

**Call Me Maybe** is a reliable function-calling system that translates natural language user queries into structured, schema-compliant JSON function calls using a lightweight 0.6B parameter language model (`Qwen/Qwen3-0.6B`).

Small language models frequently struggle to output syntactically valid JSON or adhere strictly to predefined parameter schemas when relying on prompting alone. This project solves that bottleneck by combining a **two-stage generation pipeline** with **token-level constrained decoding**, manipulating raw logits before greedy token selection so the model produces deterministic, parseable JSON without external structured-generation frameworks.

---

## Instructions

### Prerequisites

* Python 3.10 or later
* [`uv`](https://docs.astral.sh/uv/) package manager
* Local `llm_sdk/` directory placed at the repository root alongside `src/`

### Installation

Install project dependencies into an isolated virtual environment using `uv`:

```bash
make install
# or directly:
uv sync
```

### Execution

Run the pipeline with default input/output paths (`data/input/` -> `data/output/function_calls.json`):

```bash
make run
# or directly:
uv run python -m src
```

Run with custom file paths or verbose token-by-token streaming:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json \
  --verbose
```

### Code Quality & Maintenance

```bash
make lint          # Run flake8 and mypy static type checks
make lint-strict   # Run flake8 and mypy --strict
make debug         # Run the pipeline under pdb
make clean         # Remove __pycache__ and .mypy_cache artifacts
```

---

## Algorithm Explanation

To maximize reliability on a 0.6B parameter model, inference is split into two constrained autoregressive stages per prompt:

### Stage 1: Function Selection (Prefix-Tree Logit Masking)
1. `load_vocab()` reads the model's `vocab.json` via `Small_LLM_Model.get_path_to_vocab_file()`, inverts the mapping (`token_id -> token_str`), and normalizes BPE subword markers (`Ġ` to space, `Ċ` to newline).
2. `build_prompt()` formats all available function signatures and descriptions, ending with `"Chosen function: "`.
3. At every decoding step in `run_inference()`, `constrained_mask()` checks each vocabulary token against the currently generated prefix (`candidate = (generated_text + token_str).lstrip()`). Any token whose resulting prefix does not match the start of at least one valid function name in `allowed_functions` has its logit set to `-inf`.
4. Greedy decoding (`argmax`) selects the highest-scoring valid token until the decoded string matches an allowed function name.

### Stage 2: Argument Extraction (JSON Structural Masking & Brace-Depth Stopping)
1. `build_argument_prompt()` supplies the chosen function's schema and prefills the JSON object opening (`Arguments: {"`) so the model immediately generates key-value pairs.
2. During generation, `constrained_mask()` enforces RFC 8259 JSON structural constraints by masking out empty tokens, Markdown code fences (`` ``` ``), and raw ASCII control characters (`ord(ch) < 32`, such as unescaped newlines) with `-inf`.
3. Generation stops early as soon as brace depth closes the outer JSON object (`decoded_text.count("}") > decoded_text.count("{")`), preventing premature truncation when string arguments themselves contain curly braces (e.g., template variables like `{user}`).
4. `parse_argument_to_dict()` repairs minor formatting artifacts, parses the JSON, filters unknown keys, and coerces values to match the schema (`float` for `"number"`, `int` for `"integer"`, `bool` for `"boolean"`). Finally, Pydantic (`FunctionCallResult`) validates the output object before serialization.

---

## Design Decisions

* **Two-Stage Decomposition:** Separating function routing from argument extraction prevents the small model from hallucinating malformed nested JSON while still deciding which tool to invoke.
* **Prompt Prefilling (`Arguments: {"`):** Anchoring the opening brace and quote eliminates conversational preamble and forces the first generated token to be a parameter key name.
* **Global JSON Control-Character Masking vs. Whitelisting:** Instead of hardcoding an ASCII alphanumeric whitelist—which breaks legitimate arguments containing SQL wildcards (`*`), POSIX/Windows paths (`/`, `\`), or parentheses (`(`, `)`)—masking only control characters and code fences preserves full character expressiveness while keeping JSON single-line and parseable.
* **Strict Schema Coercion + Pydantic Validation:** JSON numbers are explicitly cast to `float` when `"number"` is specified (e.g., `2.0` instead of `2`) and `int` for `"integer"`, ensuring strict compliance with output schema rules.
* **Graceful Error Handling:** Missing input files, malformed JSON inputs, and empty function definitions are caught before model initialization with descriptive `stderr` messages and clean non-zero exits.

---

## Performance Analysis

* **JSON Validity:** **100%** of generated outputs are syntactically valid, schema-compliant JSON objects with zero extra keys or prose.
* **Accuracy:** Achieved **>90% accuracy** across custom validation suites covering arithmetic operations, integer/boolean checks, SQL queries, filesystem paths, and string templates, meeting the project's reliability benchmark.
* **Latency:** Loading `Qwen/Qwen3-0.6B` and evaluating an 11-prompt test suite completes in approximately **15–25 seconds** on Apple Silicon (`mps`) / CPU—well within the 5-minute threshold.
* **Determinism:** Greedy decoding (`argmax` over masked logits) guarantees reproducible results across repeated runs.

---

## Challenges Faced

1. **BPE Subword Tokenization Artifacts:** Raw tokens in `vocab.json` encode leading spaces as `Ġ` and newlines as `Ċ`. Without normalizing these markers during vocabulary loading, prefix matching in Stage 1 failed on tokens starting with a leading space.
2. **Over-Restrictive Character Whitelists:** An initial alphanumeric/punctuation whitelist blocked valid characters in SQL queries (`*`, `(`, `)`) and filesystem paths (`/`, `\`). Replacing the whitelist with RFC 8259 control-character masking solved all path and query edge cases universally.
3. **Premature Stopping on Nested Braces:** Stopping generation on the first `}` truncated template strings such as `"Hello {user}'s profile!"`. Switching the stopping condition to brace-balance tracking (`count("}") > count("{")`) resolved premature termination while still halting immediately when the outer JSON object closes.
4. **Numeric Type Distinction (`number` vs. `integer`):** Ensuring `"number"` fields always serialize as JSON floats (`40.0`) while `"integer"` fields serialize as integers (`4`) required explicit schema-guided coercion in `parse_argument_to_dict()`.

---

## Testing Strategy

* **Functional & Schema Validation:** Tested the pipeline across diverse prompt categories—including multi-parameter arithmetic functions, integer and boolean type checks, SQL queries with special symbols, Unix/Windows paths, and brace-containing string templates—verifying exact schema compliance and JSON float formatting (`2.0`).
* **Edge-Case & Fault Injection Testing:** Verified graceful error handling on missing files (`FileNotFoundError`), corrupted JSON inputs (`JSONDecodeError`), empty `functions_definition.json` arrays, and custom CLI `--input` / `--output` paths.
* **Static Analysis:** Enforced PEP 8 compliance with `flake8 .` and static type safety with `mypy` (`--disallow-untyped-defs`, `--check-untyped-defs`, and `--strict`).

---

## Example Usage

Given a prompt in `data/input/function_calling_tests.json`:

```json
[
  { "prompt": "What is the sum of 40 and 2?" },
  { "prompt": "Read the file at /home/user/data.json with utf-8 encoding" }
]
```

Running `uv run python -m src` produces `data/output/function_calls.json`:

```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 40.0,
      "b": 2.0
    }
  },
  {
    "prompt": "Read the file at /home/user/data.json with utf-8 encoding",
    "name": "fn_read_file",
    "parameters": {
      "path": "/home/user/data.json",
      "encoding": "utf-8"
    }
  }
]
```

---

## Resources

* [JSON Official Specification (json.org)](https://www.json.org/) – Reference for valid JSON syntax, string escaping rules, and control-character restrictions.
* [Python `json` Module Documentation](https://docs.python.org/3/library/json.html) – Standard library reference for parsing and serializing JSON outputs.
* [Pydantic Models Documentation](https://docs.pydantic.dev/latest/concepts/models/) – Reference for defining `BaseModel` schemas to validate structured output objects (`FunctionCallResult`).
* [A Guide to Constrained Decoding (Aidan Cooper)](https://www.aidancooper.co.uk/constrained-decoding/) – Visual, step-by-step explanation of logit masking (`-inf`) to enforce structured JSON generation in LLMs.
* Project subject and `llm_sdk` (`Small_LLM_Model`) module documentation.

### AI Usage Disclosure

AI assistance (Gemini) was used during development as a learning and debugging partner for:
* Understanding BPE subword token representations (`Ġ`, `Ċ`) and logit-masking mechanics.
* Reviewing edge cases in JSON parsing, type coercion (`number` vs. `integer`), and brace-depth early stopping.
* Structuring and proofreading this `README.md` documentation.

All code was reviewed, tested against edge cases and static linters (`flake8`, `mypy`), and verified for full comprehension prior to submission.