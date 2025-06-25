import mlflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_preparation import load_and_prepare_data
from model_functions import train_and_log_model
from hyperparameter_tuning import hyperparameter_tuning, get_run_metrics

if __name__ == "__main__":
    # Set up the MLflow tracking URI (local)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Loan Approval Classification")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_prepare_data()
    
    # Define hyperparameter grids for each model
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    lr_param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['liblinear', 'saga'],
        'penalty': ['l1', 'l2']
    }

    svc_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1]
    }

    # Run hyperparameter tuning for each model
    print("Training Random Forest model with hyperparameter tuning...")
    best_rf, rf_run_id = hyperparameter_tuning("RandomForest", rf_param_grid, 
                                             X_train_scaled, X_test_scaled, y_train, y_test)

    print("Training Logistic Regression model with hyperparameter tuning...")
    best_lr, lr_run_id = hyperparameter_tuning("LogisticRegression", lr_param_grid, 
                                             X_train_scaled, X_test_scaled, y_train, y_test)

    print("Training SVC model with hyperparameter tuning...")
    best_svc, svc_run_id = hyperparameter_tuning("SVC", svc_param_grid, 
                                               X_train_scaled, X_test_scaled, y_train, y_test)

    # Get metrics for each model
    rf_metrics = get_run_metrics(rf_run_id)
    lr_metrics = get_run_metrics(lr_run_id)
    svc_metrics = get_run_metrics(svc_run_id)

    # Create a comparison DataFrame
    models = ["RandomForest", "LogisticRegression", "SVC"]
    metrics_df = pd.DataFrame({
        "Model": models,
        "Accuracy": [rf_metrics.get("best_cv_score", 0), 
                    lr_metrics.get("best_cv_score", 0), 
                    svc_metrics.get("best_cv_score", 0)],
        "Run ID": [rf_run_id, lr_run_id, svc_run_id]
    })

    # Sort by accuracy to find the best model
    metrics_df = metrics_df.sort_values("Accuracy", ascending=False)

    print("\nModel Comparison:")
    print(metrics_df)

    print(f"\nBest Model: {metrics_df.iloc[0]['Model']} with accuracy {metrics_df.iloc[0]['Accuracy']:.4f}")
    print(f"Run ID: {metrics_df.iloc[0]['Run ID']}")

    # Instructions for launching the MLflow UI
    print("\nTo view the experiments in the MLflow UI, run the following command in your terminal:")
    print("mlflow ui")