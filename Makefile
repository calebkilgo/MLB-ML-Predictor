.PHONY: install etl features train backtest app test

install:
	pip install -e ".[dev]"

etl:
	python -m src.etl.build_dataset

features:
	python -m src.features.assemble

train:
	python -m src.models.train

backtest:
	python -m src.models.backtest

app:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q
