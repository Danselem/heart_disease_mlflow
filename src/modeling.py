"""
Script to create Catboost Models for the dataset.

This script performs the following tasks:
1. Loads configuration parameters from a YAML file.
2. Loads a pre-processed training dataset.
3. Optimizes hyperparameters for a classification model using the CatBoost algorithm.
4. Registers the best experiment and model with MLflow.

Usage:
    This script is intended to be run as a standalone script.
    Ensure that the required environment variables are set (e.g., using a .env file).

Author: [Daniel Egbo]
Date: [2024-08-30]

"""

import pandas as pd
import yaml
from dotenv import load_dotenv
from pathlib import Path

from src.utils.mlflow.manage_mlflow import (
    create_mlflow_experiment,
    register_best_experiment,
    register_best_model
)

from src.utils.models.hpo import classification_optimization
from src import logger

def main():
    """Main function to run the optimization process. It loads the training
    dataset, optimizes the hyperparameters, registers the best experiment,
    and registers the best model."""
    load_dotenv()
    
    # Load params.yaml file
    params_file = Path("params.yaml")
    modeling_params = yaml.safe_load(
        open(params_file, encoding="utf-8"))["modeling"]
    n_trials = modeling_params["n_trials"]
    selected_loss_function = modeling_params["loss_function"]
    selected_model_family = modeling_params["model_family"]
    selected_objective_function = modeling_params["objective_function"]
    
    create_mlflow_experiment(f"{selected_model_family}_experiment")

    # Load the training dataset
    data_path = Path("data/processed/heart_train_cleaned.parquet")
    df_train_heart = pd.read_parquet(data_path)

    # logger.info(f"Columns in the training dataset: {df_train_heart.columns.to_list()}")

    x_train_heart = df_train_heart.copy()
    x_train_heart = x_train_heart.drop(columns=["HadHeartAttack"])
    
    y_train_heart = df_train_heart.copy()["HadHeartAttack"]

    best_classification_params = classification_optimization(
        x_train=x_train_heart, y_train=y_train_heart,
        model_family=selected_model_family,
        loss_function=selected_loss_function,
        objective_function=selected_objective_function,
        num_trials=n_trials, diagnostic=True)
    
    register_best_experiment(
        x_train=x_train_heart, y_train=y_train_heart,
        model_family=selected_model_family,
        loss_function=selected_loss_function,
        best_params=best_classification_params)
    
    register_best_model(
        model_family=selected_model_family,
        loss_function=selected_loss_function)


if __name__ == "__main__":
    main()
