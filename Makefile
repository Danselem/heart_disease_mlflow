install-python:
	uv python install 3.10


init:
	uv init && rm hello.py
	uv tool install black

install:
	uv venv
	. .venv/bin/activate
	uv pip install --all-extras --requirement pyproject.toml
	# uv pip sync requirements.txt
	# uv add -r requirements.txt

env:
	cp example.env .env

spdata:
	echo "Splitting data"
	uv run -m src.data_split

cleandata:
	echo "Cleaning data"
	uv run -m src.data_cleaning

model:
	echo "Training and performing optimization"
	uv run -m src.modeling

save_model:
	echo "Fetching and saving model"
	uv run -m src.gather_mlflow_model

sample:
	uv run -m src.create_input_example

serve_local:
	uv run -m src.serve_local

serve:
	uv run -m src.serve


quality_checks:
	echo "Running quality checks"
	isort .

test:
	echo "Running tests"
	pytest --disable-warnings tests/

dvc:
	uv run dvc repro

build_docker:
	uv run docker build -t heart-disease .

run_docker:
	uv run docker run -d -p 80:80 heart-disease

build: quality_checks test
	echo "Building package"
	uv run -m src.gather_mlflow_model
	docker build . -t Danselem/heart-disease:latest

publish: build
	echo "Publishing package"
	docker push Danselem/heart-disease:latest