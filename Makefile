.PHONY: help install install-dev lint format test clean run-simple demo

help:
	@echo "CodeAgentX 常用命令："
	@echo "  make install      安装运行时依赖"
	@echo "  make install-dev  安装开发依赖 + 可编辑安装"
	@echo "  make lint         ruff 静态检查"
	@echo "  make format      ruff 自动格式化"
	@echo "  make test        运行 pytest"
	@echo "  make run-simple  运行 SimpleAgent 示例"
	@echo "  make clean       清理缓存"

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements.txt
	python -m pip install -e .

lint:
	python -m ruff check src tests examples

format:
	python -m ruff format src tests examples
	python -m ruff check --fix src tests examples

test:
	python -m pytest

run-simple:
	python examples/simple_agent.py

clean:
	python -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil;shutil.rmtree('.pytest_cache',ignore_errors=True);shutil.rmtree('.ruff_cache',ignore_errors=True)"
