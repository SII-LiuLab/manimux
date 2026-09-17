.PHONY: format lint typecheck test test-integration mock-run

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

mock-run:
	uv run python scripts/validation/run_headless.py --config configs/mock.yaml
