<p align="center">
  <a href="" rel="noopener">
    <img width=400px height=200px src="https://dvl2h13awlxkt.cloudfront.net/assets/general-images/Knowledge/_1200x630_crop_center-center_82_none/CVD-iStock-1266230179.jpg?mtime=1653282867" alt="Heart Disease">
  </a>
</p>

<h2 align="center">🩺 Predicting Heart Disease: An MLOps Approach</h2>

<p align="center">
  A hands-on project leveraging MLOps best practices to build, deploy, and monitor a heart disease prediction model. <br>
</p>


# 🧐 Problem Description <a name="about"></a>

Heart disease remains one of the leading causes of death worldwide, with key risk factors including high blood pressure, high cholesterol, obesity, smoking, and lack of physical activity. Early detection and prediction of heart disease can play a crucial role in preventive healthcare and patient outcomes.

This project leverages machine learning techniques to predict the likelihood of heart disease based on self-reported health indicators collected via telephonic surveys. The dataset, sourced from the Centers for Disease Control and Prevention (CDC), is part of the Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual health-related surveys across the United States.

### 📌 Project Scope
The goal of this project is to develop an end-to-end MLOps pipeline that automates the training, deployment, and monitoring of a machine learning model capable of predicting heart disease risk. This includes:
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and feature engineering.
- **Model Training & Evaluation:** Applying classification algorithms such as logistic regression, random forests, and gradient boosting.
- **Deployment & MLOps Integration:** Serving the trained model via a REST API using Flask and Docker, while leveraging CI/CD pipelines and model monitoring to ensure performance and reliability.
- **Scalability & Reproducibility:** Utilizing cloud-based storage and MLflow for model tracking and versioning.

### 📊 Dataset Information
The dataset consists of 40 features derived from nearly 300 original variables, carefully curated to represent key indicators of heart disease. It includes factors such as BMI, smoking status, alcohol consumption, diabetes history, and physical activity levels. Given its real-world nature, the dataset presents challenges such as class imbalance, requiring thoughtful model selection and evaluation strategies.

- **Source:** [Kaggle - Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
- **Original Data Provider:** CDC’s Behavioral Risk Factor Surveillance System (BRFSS)
- **Class Distribution:** Imbalanced (fewer positive cases of heart disease)

### 🚀 Why This Matters
With the increasing availability of health-related data, applying MLOps principles to healthcare prediction models ensures not only the accuracy and reliability of predictions but also the ability to seamlessly deploy, maintain, and monitor models in production. This project demonstrates the power of machine learning in solving real-world healthcare challenges while emphasizing best practices in MLOps.

This repository serves as an educational portfolio project and was developed as part of the [MLOps Zoomcamp](https://datatalks.club/courses.html) by DataTalks.Club.

---

## 🏆 Modeling

The model's performance is evaluated using the **F1 score**, which is particularly effective for handling imbalanced datasets. Given that heart disease is relatively rare in the dataset, the F1 score ensures a balanced assessment between precision and recall, helping to optimize the model for real-world use cases.

## 🔍 Overview <a name="overview"></a>

This project integrates multiple tools and frameworks to streamline the machine learning lifecycle. The image below highlights the core technologies used:

![Tools](docs/tools.jpg)

## 📊 Exploratory Data Analysis (EDA) <a name="eda"></a>

Before model development, a comprehensive **Exploratory Data Analysis (EDA)** is conducted to gain insights into the dataset’s structure, distributions, and feature correlations. EDA helps identify missing values, outliers, and potential data transformations to improve model performance.

The **`notebooks/`** directory contains a dedicated EDA notebook where the dataset is thoroughly analyzed. This step lays the foundation for feature selection and engineering.

## 🧪 Experiment Tracking & Model Registry <a name="experiment"></a>

To ensure reproducibility and effective model management, this project utilizes **MLflow** for experiment tracking and model registry. MLflow facilitates:
- Logging and comparing multiple model runs.
- Tracking hyperparameters and performance metrics.
- Registering and versioning models for deployment.

### 📌 Key Resources:
- **MLflow Documentation:** [Read More](https://www.mlflow.org/docs/latest/index.html)
- **Project Hosting on DagsHub:** [View Project](https://dagshub.com/Danselem/heart_disease_mlflow)
- **MLflow Experiment Server on DagsHub:** [Access Here](https://dagshub.com/Danselem/heart_disease_mlflow/experiments)

DagsHub integrates **DVC**, **MLflow**, and **Git**, providing a unified environment for managing experiments, model artifacts, and version control.

## 🔄 Workflow Orchestration <a name="workflow"></a>

This project employs **Data Version Control (DVC)** to orchestrate the workflow. DVC provides:
- Version control for datasets, models, and intermediate files.
- Seamless integration with Git, ensuring that data and code are versioned together.
- Reproducibility by enabling easy rollback to previous states.

### 📌 Key Resource:
- **DVC Documentation:** [Read More](https://dvc.org/doc)

## ⚙️ Model Deployment <a name="deployment"></a>

The trained model is deployed as a **REST API** using **FastAPI**, making it accessible for real-time predictions. The deployment pipeline includes:
- **Containerization with Docker** to ensure portability.
- **Scalability** for serving multiple inference requests.
- **Endpoint exposure** via `app/main.py` for seamless integration.

The containerized image is publicly available and can be pulled from Docker Hub:
- **Docker Repository:** [View Here](https://hub.docker.com/repository/docker/Danselem/indicators-of-heart-disease)

## 📈 Model Monitoring <a name="monitoring"></a>

To track model performance over time, the project integrates **Evidently**, **PostgreSQL**, and **Grafana** for interactive monitoring and analytics.

### 📌 Monitoring Components:
1. **Evidently** – Provides insights into model drift, data drift, and feature importance.  
   - **Documentation:** [Read More](https://evidentlyai.com/)
2. **PostgreSQL** – Stores model predictions and performance metrics for historical analysis.  
   - **Documentation:** [Read More](https://www.postgresql.org/)
3. **Grafana** – Visualizes key metrics and trends, helping to detect performance degradation.  
   - **Documentation:** [Read More](https://grafana.com/docs/grafana/latest/)

### 📊 Simulation & Monitoring Process:
- The model is simulated on multiple data batches (500 samples per batch).
- A **daily batch processing scenario** is simulated, where new data is processed each day.
- **Metrics are stored in PostgreSQL** and visualized in **Grafana dashboards**.

Below is an example of the real-time monitoring dashboard:

![Dashboard](docs/dashboard.png)
  

# 🖥️ Reproducibility <a name="reproducibility"></a>

To ensure **reproducibility**, the entire pipeline is defined in `dvc.yaml`. Running the pipeline will automatically execute necessary steps, ensuring consistency in data processing, model training, and evaluation.

## 🚀 Running the Pipeline

To execute the full pipeline, run:

```bash
make dvc
```

This command checks which stages have already been completed and only runs the remaining ones. Upon completion, the trained model and metrics will be stored in the **MLflow server**.

Once the model is trained, it can be downloaded with:

```bash
make save_model
```

## 🛠️ Step-by-Step Workflow

Follow these steps to set up and execute the project:

### 1️⃣ Installation
---
Clone the repository:

```bash
git clone https://github.com/Danselem/heart_disease_mlflow.git
```

Navigate into the project directory:

```bash
cd heart_disease_mlflow
```

### 2️⃣ Set Up the Environment
---
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) according to your platform, then install dependencies:

```bash
make install
```

Set up environment variables:

```bash
make env
```

Then, update `.env` with the required credentials and **DagsHub repository** details.

### 3️⃣ Load and Prepare Data
---
Split the dataset:

```bash
make spdata
```

Clean the data:

```bash
make cleandata
```

### 4️⃣ Train and Optimize the Model
---
Train the model:

```bash
make model
```

Modify `params.yaml` to experiment with different hyperparameters, then retrain the model using the command above.  

Once satisfied with the performance, fetch the best model:

```bash
make save_model
```

This downloads the **best-performing model** as `model.pkl` for deployment.

### 5️⃣ Model Serving
---
Generate a sample input JSON:

```bash
make sample
```

Run the model locally:

```bash
make serve_local
```

### 6️⃣ Deploy with Docker
---
To containerize the model, build a **Docker image**:

```bash
make build_docker
```

Run the **Docker container**:

```bash
make run_docker
```

Once the container is running, generate predictions by executing:

```bash
make serve
```

This ensures the model is deployed and can be accessed via API for real-world inference.


## 🪖 Best Practices <a name="best_practices"></a>

This project follows **best practices** to ensure code quality, maintainability, and smooth deployment. 

### ✅ Continuous Integration & Code Quality Checks  

Every commit triggers a **CI/CD pipeline** that performs **static code analysis** using `flake8`. If any errors are detected, the pipeline fails, ensuring that only high-quality code is merged.  

To enforce code quality checks locally, **pre-commit hooks** are configured in `.pre-commit-config.yaml`. These hooks can be installed and executed before committing changes, avoiding delays caused by waiting for CI/CD validation.

### 🛠️ Setting Up Pre-Commit Locally

Install pre-commit hooks:

```bash
pre-commit install
```

Run pre-commit checks on all files:

```bash
pre-commit run --all-files
```

### 📦 Makefile Automation  

A **Makefile** is provided to streamline development tasks, including:

- Running tests  
- Checking code quality  
- Building and pushing Docker images  

To execute all necessary checks and publish the Docker image, simply run:

```bash
make publish
```

---

### 📝 License  

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for full details.


