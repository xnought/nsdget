all: run

run:
	uv run nsdget

publish:
	rm -fr dist/

	uv build
	uv publish --token $(TOKEN) 

	rm -fr dist/