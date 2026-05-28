setup:
	uv sync

test:
	pytest tests/ -v

mlflow:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

format:
	ruff check . --fix

run:
	python src/main.py