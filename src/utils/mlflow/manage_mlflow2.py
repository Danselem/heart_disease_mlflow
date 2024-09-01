"""Manage MLflow credentials and experiments"""
import os
import random
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import yaml
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.utils.models.plotoutputs import plot_confusion_matrix
from src import logger



# Configuration constants
DAGSHUB_REPO_OWNER = "Danselem"
DAGSHUB_REPO = "heart_disease_mlflow"
seed = 1024

def config_mlflow() -> None:
    """Configure MLflow to log metrics to the Dagshub repository"""
    load_dotenv()
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO}.mlflow")

    mlflow.autolog()
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_mlflow_experiment(experiment_name: str) -> None:
    """Create or set an MLflow experiment"""
    config_mlflow()
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            mlflow.create_experiment(experiment_name)
            logger.info(f"MLflow experiment '{experiment_name}' created.")
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to create MLflow experiment '{experiment_name}': {e}")
            raise
    else:
        logger.info(f"MLflow experiment '{experiment_name}' already exists.")
    
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to '{experiment_name}'.")

def register_best_model(model_family: str, loss_function: str) -> None:
    """Register the best model after the optimization process"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"{model_family}_experiment")
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{loss_function} ASC"]  # Corrected to ASC for minimum loss
    )[0]

    run_id = best_run.info.run_id
    mlflow.register_model(f"runs:/{run_id}/", f"{model_family}_best_model")

def register_best_catboost_experiment(
        x_train: ArrayLike, y_train: ArrayLike,
        best_params: dict) -> str:
    """Register the best CatBoost experiment"""
    # total_samples_size = x_train.shape[0]
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed)

    with open("params.yaml", encoding="utf-8") as file:
        dvc_params = yaml.safe_load(file)

    best_params["cat_features"] = dvc_params["categorical_features"]

    model = CatBoostClassifier(**best_params, verbose=False)
    model.fit(x_train, y_train, eval_set=(x_val, y_val))

    test_hat = model.predict(x_val)
    train_hat = model.predict(x_train)

    plt.switch_backend("agg")
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy_score(y_val, test_hat))
        mlflow.log_metric("f1", f1_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("precision", precision_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, train_hat))
        mlflow.log_metric("train_f1", f1_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_precision", precision_score(y_train, train_hat, pos_label=1))

        mlflow.catboost.log_model(model, "model")

        plot_confusion_matrix(y_train, train_hat, "train")
        mlflow.log_artifact("train_confusion_matrix.png")
        plot_confusion_matrix(y_val, test_hat, "test")
        mlflow.log_artifact("test_confusion_matrix.png")

        os.remove("train_confusion_matrix.png")
        os.remove("test_confusion_matrix.png")
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("dvc.yaml")
        mlflow.log_artifact("data/processed/heart_train_cleaned.parquet")

    return run.info.run_id

def register_best_xgboost_experiment(
        x_train: ArrayLike, y_train: ArrayLike,
        best_params: dict) -> str:
    """Register the best XGBoost experiment"""
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed)

    # Ensure all necessary parameters are of correct type
    best_params = {
        'max_depth': best_params.get('max_depth'),
        'eta': best_params.get('eta'),
        'gamma': best_params.get('gamma'),
        'reg_alpha': best_params.get('reg_alpha'),
        'reg_lambda': best_params.get('reg_lambda'),
        'colsample_bytree': best_params.get('colsample_bytree'),
        'colsample_bynode': best_params.get('colsample_bynode'),
        'colsample_bylevel': best_params.get('colsample_bylevel'),
        'n_estimators': best_params.get('n_estimators'),
        'learning_rate': best_params.get('learning_rate'),
        'min_child_weight': best_params.get('min_child_weight'),
        'max_delta_step': best_params.get('max_delta_step'),
        'subsample': best_params.get('subsample'),
        'objective': best_params.get('objective', 'binary:logistic'),
        'eval_metric': best_params.get('eval_metric', 'aucpr'),
        'random_state': seed
    }
    
    # best_params = {
    #     'max_depth': int(best_params.get('max_depth', 3)),  # Ensure this is an integer
    #     'eta': float(best_params.get('eta', 0.3)),
    #     'gamma': float(best_params.get('gamma', 0)),
    #     'reg_alpha': float(best_params.get('reg_alpha', 0)),
    #     'reg_lambda': float(best_params.get('reg_lambda', 1)),
    #     'colsample_bytree': float(best_params.get('colsample_bytree', 1)),
    #     'colsample_bynode': float(best_params.get('colsample_bynode', 1)),
    #     'colsample_bylevel': float(best_params.get('colsample_bylevel', 1)),
    #     'n_estimators': int(best_params.get('n_estimators', 100)),  # Ensure this is an integer
    #     'learning_rate': float(best_params.get('learning_rate', 0.1)),
    #     'min_child_weight': float(best_params.get('min_child_weight', 1)),
    #     'max_delta_step': float(best_params.get('max_delta_step', 0)),
    #     'subsample': float(best_params.get('subsample', 1)),
    #     'objective': best_params.get('objective', 'binary:logistic'),
    #     'eval_metric': best_params.get('eval_metric', 'aucpr'),
    #     'random_state': seed
    # }


    model = XGBClassifier(**best_params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], ) # early_stopping_rounds=10

    test_hat = model.predict(x_val)
    train_hat = model.predict(x_train)

    plt.switch_backend("agg")
    with mlflow.start_run() as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy_score(y_val, test_hat))
        mlflow.log_metric("f1", f1_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("precision", precision_score(y_val, test_hat, pos_label=1, zero_division=1))
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, train_hat))
        mlflow.log_metric("train_f1", f1_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_precision", precision_score(y_train, train_hat, pos_label=1))

        # Ensure file paths are correct
        plot_confusion_matrix(y_train, train_hat, "train")
        mlflow.log_artifact("train_confusion_matrix.png")
        plot_confusion_matrix(y_val, test_hat, "test")
        mlflow.log_artifact("test_confusion_matrix.png")

        # Clean up files
        os.remove("train_confusion_matrix.png")
        os.remove("test_confusion_matrix.png")
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("dvc.yaml")
        mlflow.log_artifact("data/processed/heart_train_cleaned.parquet")

    return run.info.run_id

def register_best_random_forest_experiment(
        x_train: ArrayLike, y_train: ArrayLike,
        best_params: dict) -> str:
    """Register the best Random Forest experiment found by the optimization process.

    Args:
        x_train: Training data features.
        y_train: Training data labels.
        best_params: Hyperparameter dictionary for Random Forest.

    Returns:
        run_id: The ID of the run in MLflow.
    """
    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=seed)

    # Initialize the Random Forest model with the best parameters
    model = RandomForestClassifier(**best_params, random_state=seed)
    model.fit(x_train, y_train)

    # Predict on validation and training sets
    test_hat = model.predict(x_val)
    train_hat = model.predict(x_train)
    test_probs = model.predict_proba(x_val)[:, 1]
    train_probs = model.predict_proba(x_train)[:, 1]

    plt.switch_backend("agg")
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params(best_params)
        mlflow.log_param("total_samples_size", x_train.shape[0])

        # Log metrics for validation set
        mlflow.log_metric("accuracy", accuracy_score(y_val, test_hat))
        mlflow.log_metric("f1", f1_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("precision", precision_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("recall", recall_score(y_val, test_hat, pos_label=1))
        mlflow.log_metric("roc_auc", roc_auc_score(y_val, test_probs))

        # Log metrics for training set
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, train_hat))
        mlflow.log_metric("train_f1", f1_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_precision", precision_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_recall", recall_score(y_train, train_hat, pos_label=1))
        mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, train_probs))

        # Log the Random Forest model
        mlflow.sklearn.log_model(model, "model")

        # Plot train confusion matrix
        plot_confusion_matrix(y_train, train_hat, "train")
        mlflow.log_artifact("train_confusion_matrix.png")
        # Plot test confusion matrix
        plot_confusion_matrix(y_val, test_hat, "test")
        mlflow.log_artifact("test_confusion_matrix.png")
        # Delete the confusion matrix files
        os.remove("train_confusion_matrix.png")
        os.remove("test_confusion_matrix.png")
        # Log additional artifacts
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("dvc.yaml")
        mlflow.log_artifact("data/processed/heart_train_cleaned.parquet")

    return run.info.run_id

def load_model_by_name(model_name: str):
    """Load a pre-trained model from an MLflow server"""
    config_mlflow()
    client = mlflow.MlflowClient()
    registered_model = client.get_registered_model(model_name)
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model
