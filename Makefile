.PHONY: install test lint type format demo dashboard api build clean cov

install:
	pip install -e ".[dev,langgraph,openai]"

test:
	pytest -q

cov:
	pytest -q --cov=agentguard --cov-report=term-missing

lint:
	ruff check .

type:
	mypy agentguard

format:
	ruff format .

demo:
	python examples/mcp_proxy_demo.py

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	cd dashboard && npm install && npm run dev

build:
	python -m build

clean:
	rm -rf dist build *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
