PYTHON ?= python3
PROJECT_SUFFIX ?= proj99
MLFLOW_TRACKING_URI ?= http://127.0.0.1:5000

.PHONY: setup demo-data train-local docker-build docker-train

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

demo-data:
	$(PYTHON) scripts/make_demo_dataset.py --output-dir data/demo

train-local:
	PYTHONPATH=src MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) $(PYTHON) src/train.py \
		--config configs/candidates.yaml \
		--train-csv data/demo/gemspot_train.csv \
		--val-csv data/demo/gemspot_val.csv \
		--experiment-name GemSpot-WillVisit

docker-build:
	docker build -t gemspot-train-$(PROJECT_SUFFIX) .

docker-train:
	docker run --rm \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		-v "$(PWD)/data:/app/data" \
		-v "$(PWD)/artifacts:/app/artifacts" \
		gemspot-train-$(PROJECT_SUFFIX) \
		python src/train.py \
			--config configs/candidates.yaml \
			--train-csv data/demo/gemspot_train.csv \
			--val-csv data/demo/gemspot_val.csv \
			--experiment-name GemSpot-WillVisit
