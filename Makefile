all: run

run:
	uv run nsdget/__init__.py

test:
	uv run pytest nsdget/test.py

publish:
	rm -fr dist/

	uv build
	uv publish --token $(TOKEN) 

	rm -fr dist/