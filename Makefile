all: run

run:
	uv run nsdget/__init__.py

test:
	uv run pytest nsdget/test.py