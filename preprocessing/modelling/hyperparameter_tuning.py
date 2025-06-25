import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlflow.models.signature import infer_signature

def hyperparameter_tuning(model_type, param_grid, X_train, X_test, y_train, y_test):
    """
    Perform hyperparameter tuning with grid search
    
    Args:
        model_type: Type of model ("RandomForest", "LogisticRegression", or "SVC")
        param_grid: Hyperparameter grid
        X_train, X_test, y_train, y_test: Training and testing data
        
    Returns:
        best_model, run_id
    """
    with mlflow.start_run(run_name=f"{model_type}_tuning") as parent_run:
        mlflow.log_param("model_type", model_type)
        
        if model_type == "RandomForest":
            base_model = RandomForestClassifier(random_state=42)
        elif model_type == "LogisticRegression":
            base_model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "SVC":
            base_model = SVC(random_state=42, probability=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Perform Grid Search
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Log best parameters and score
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Train and log the best model as a child run
        with mlflow.start_run(run_name=f"{model_type}_best", nested=True) as child_run:
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Calculate ROC AUC if probabilities are available
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Log parameters
            for key, value in grid_search.best_params_.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc_score", roc_auc)
            
            # Create and log confusion matrix figure
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_type} (Best)')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            
            # Save and log the figure
            confusion_matrix_path = f"confusion_matrix_{model_type}_best.png"
            plt.savefig(confusion_matrix_path)
            mlflow.log_artifact(confusion_matrix_path)
            plt.close()
            
            # Generate and log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_path = f"classification_report_{model_type}_best.csv"
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)
            
            # Log the model
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(best_model, f"{model_type}_best", signature=signature)
            
            print(f"Best {model_type} Model")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test F1 Score: {f1:.4f}")
            if roc_auc is not None:
                print(f"Test ROC AUC: {roc_auc:.4f}")
            print("-" * 50)
            
        return best_model, parent_run.info.run_id

def get_run_metrics(run_id):
    """
    Get metrics from a run
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Metrics dictionary
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics