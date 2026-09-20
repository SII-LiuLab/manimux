.PHONY: format lint typecheck test test-integration

format:
	uv run ruff format manimux tests
	uv run ruff check --fix manimux tests

lint:
	uv run ruff format --check manimux tests
	uv run ruff check manimux tests

typecheck:
	uv run mypy manimux

test:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration
