PYTHON = uv run python
SRC = src
LLM_SDK = llm_sdk

.PHONY: all install run debug clean lint lint-strict

all: run

install:
	uv sync

run:
	$(PYTHON) -m $(SRC) $(ARGS)

debug:
	$(PYTHON) -m pdb -m $(SRC)

clean:
	rm -rf __pycache__ $(SRC)/__pycache__ $(LLM_SDK)/__pycache__ .mypy_cache $(SRC)/.mypy_cache

lint:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 $(SRC)
	uv run mypy $(SRC) --strict