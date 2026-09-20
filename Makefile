.PHONY: install test lint synthetic validate episodes

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
