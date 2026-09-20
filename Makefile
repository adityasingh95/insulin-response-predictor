.PHONY: install test lint synthetic validate episodes assess templates forward demo demo-docker

install:
	python -m pip install -e '.[dev]'

test:
	python -m unittest discover -s tests -v

lint:
	ruff check .

synthetic:
	irp generate-synthetic --output data/synthetic --days 21

validate:
	irp validate --input data/synthetic

episodes:
	irp build-episodes --input data/synthetic --output data/interim/episodes.csv

assess:
	irp assess --input data/synthetic --output reports/generated/synthetic

templates:
	irp create-templates --output templates

forward:
	irp evaluate-forward --input data/synthetic --output reports/generated/forward

demo:
	irp run-demo --output demo-output --days 90

demo-docker:
	docker compose run --rm demo
