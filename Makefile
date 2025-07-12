########################################################################
# Kubernetes cluster operations (MLOps)
########################################################################
# create the local Kubernetes cluster
export CLUSTER_NAME=cluster-789

cluster:
	kind create cluster --config kind.yaml --name $(CLUSTER_NAME)
	kubectl config use-context kind-$(CLUSTER_NAME)
	
	@echo "Listing the nodes in the cluster"
	kubectl get nodes

# delete the local Kubernetes cluster
delete-cluster:
	kind delete cluster --name $(CLUSTER_NAME)

# list Docker images registered in the local cluster
list-images:
	docker exec -it $(CLUSTER_NAME)-control-plane crictl images

########################################################################
# ML engineer operations (ML)
########################################################################
export PORT=5005

install-python:
	uv python install 3.9


init:
	uv venv --python 3.9
	uv init && rm hello.py
	uv tool install black

install:
	. .venv/bin/activate
	# uv pip install --all-extras --requirement pyproject.toml
	# uv pip sync requirements.txt
	uv add -r requirements.txt

delete:
	rm uv.lock pyproject.toml .python-version && rm -rf .venv

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

observe:
	uv run -m src.observability

quality_checks:
	echo "Running quality checks"
	isort .

test:
	echo "Running tests"
	pytest --disable-warnings tests/

dvc:
	uv run dvc repro

build:
	docker build -t heart-disease:v1.0.0 .

run:
	docker run -d -p $(PORT):5000 heart-disease:v1.0.0

# push the Docker image to the local Kubernetes image registry
push:
	kind load docker-image heart-disease:v1.0.0 --name $(CLUSTER_NAME)

# deploy the Docker image to the local Kubernetes cluster
deploy: build push
	kubectl apply -f manifests/deployment.yaml
	kubectl apply -f manifests/service.yaml
	kubectl wait --for=condition=ready pod -l app=heart-disease --timeout=60s
	kubectl port-forward svc/heart-disease $(PORT):5000

build-docker: quality_checks test
	echo "Building package"
	uv run -m src.gather_mlflow_model
	docker build . -t Danselem/heart-disease:latest

publish: build
	echo "Publishing package"
	docker push Danselem/heart-disease:latest