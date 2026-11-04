PYTHON = uv run python
SRC = src
LLM_SDK = llm_sdk

.PHONY: run lint clean

run:
	$(PYTHON) -m $(SRC)

lint:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=skip

clean:
	rm -rf __pycache__ $(SRC)/__pycache__ $(LLM_SDK)/__pycache__ .mypy_cache $(SRC)/.mypy_cache