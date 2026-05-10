.PHONY: install install-dev test test-llm coverage lint format clean run

# 安装
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# 测试
test:
	python -m pytest tests/ -v --tb=short

test-llm:
	python -m pytest tests/ -v --tb=short -m llm

test-coverage:
	python -m pytest tests/ --cov=src/thesischeck --cov-report=term-missing --cov-report=html

# 代码质量
lint:
	ruff check src/ tests/
	mypy src/thesischeck/ --ignore-missing-imports

format:
	black src/ tests/
	ruff check --fix src/ tests/

# 运行
run:
	python -m src.thesischeck.api.main

run-llm:
	LLM_PROVIDER=claude CLAUDE_API_KEY=$$CLAUDE_API_KEY python -m src.thesischeck.api.main

# 清理
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Docker
docker-build:
	docker build -t thesischeck .

docker-run:
	docker run -p 8000:8000 -e LLM_PROVIDER=$$LLM_PROVIDER -e $$LLM_PROVIDER\ _API_KEY thesischeck