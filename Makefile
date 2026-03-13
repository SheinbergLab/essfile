.PHONY: build publish clean

build: clean
	uv tool run --from build pyproject-build

publish: build
	uv tool run twine upload dist/*

clean:
	rm -rf dist/ build/ src/*.egg-info
