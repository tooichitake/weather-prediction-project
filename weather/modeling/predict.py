"""Model prediction and evaluation utilities for weather prediction models."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from typing import Dict, Any, Tuple, Union, Optional
import lightgbm as lgb
from sklearn.pipeline import Pipeline


def evaluate_model(
    model: Union[Pipeline, lgb.Booster],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    categorical_features: list = None,
    model_type: str = 'sklearn',
    best_iteration: int = None
) -> Tuple[Dict[str, float], str]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model (sklearn Pipeline or LightGBM Booster)
        X_test: Test features
        y_test: Test target
        threshold: Classification threshold
        categorical_features: List of categorical feature names (for LightGBM)
        model_type: Type of model ('sklearn' or 'lightgbm')
        best_iteration: Best iteration for LightGBM model
        
    Returns:
        Tuple of (metrics_dict, classification_report_str)
    """
    # Prepare test data based on model type
    if model_type == 'lightgbm' and categorical_features:
        # Convert categorical features to 'category' dtype for LightGBM
        for cat_col in categorical_features:
            if cat_col in X_test.columns:
                X_test[cat_col] = X_test[cat_col].astype('category')
    
    # Get predictions
    if model_type == 'sklearn':
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:  # lightgbm
        y_test_proba = model.predict(X_test, num_iteration=best_iteration)
    
    # Apply threshold
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    # Calculate metrics
    from .train import calculate_metrics
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    # Generate classification report
    report = classification_report(
        y_test, 
        y_test_pred, 
        target_names=['No Rain', 'Rain'],
        digits=4
    )
    
    # Create formatted output
    output = f"\n{'='*60}\n"
    output += f"CLASSIFICATION REPORT - TEST SET\n"
    output += f"{'='*60}\n"
    output += report
    output += f"\nAUC-ROC: {test_metrics['auc']:.4f}\n"
    output += f"Using threshold: {threshold:.3f}\n"
    
    return test_metrics, output


def predict_with_model(
    model: Union[Pipeline, lgb.Booster],
    X: pd.DataFrame,
    threshold: float = 0.5,
    categorical_features: list = None,
    model_type: str = 'sklearn',
    best_iteration: int = None,
    return_proba: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Make predictions with a trained model.
    
    Args:
        model: Trained model (sklearn Pipeline or LightGBM Booster)
        X: Features to predict on
        threshold: Classification threshold
        categorical_features: List of categorical feature names (for LightGBM)
        model_type: Type of model ('sklearn' or 'lightgbm')
        best_iteration: Best iteration for LightGBM model
        return_proba: Whether to return probabilities along with predictions
        
    Returns:
        Predictions array or tuple of (predictions, probabilities) if return_proba=True
    """
    # Prepare data based on model type
    if model_type == 'lightgbm' and categorical_features:
        # Convert categorical features to 'category' dtype for LightGBM
        for cat_col in categorical_features:
            if cat_col in X.columns:
                X[cat_col] = X[cat_col].astype('category')
    
    # Get probability predictions
    if model_type == 'sklearn':
        y_proba = model.predict_proba(X)[:, 1]
    else:  # lightgbm
        y_proba = model.predict(X, num_iteration=best_iteration)
    
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    if return_proba:
        return y_pred, y_proba
    return y_pred


def load_model_and_info(
    model_path: str,
    info_path: str,
    model_type: str = 'sklearn'
) -> Dict[str, Any]:
    """
    Load a saved model and its associated information.
    
    Args:
        model_path: Path to the saved model file
        info_path: Path to the model info JSON file
        model_type: Type of model ('sklearn' or 'lightgbm')
        
    Returns:
        Dictionary containing model, threshold, and other metadata
    """
    import json
    import joblib
    
    # Load model
    if model_type == 'sklearn':
        model = joblib.load(model_path)
    else:  # lightgbm
        model = lgb.Booster(model_file=model_path)
    
    # Load model info
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    # Combine model and info
    result = {
        'model': model,
        'model_type': model_type,
        'threshold': model_info.get('best_threshold', 0.5),
        'best_iteration': model_info.get('best_iteration'),
        'feature_names': model_info.get('feature_names'),
        'categorical_features': model_info.get('categorical_features'),
        'validation_metrics': model_info.get('validation_metrics'),
        'best_params': model_info.get('best_params'),
        'final_params': model_info.get('final_params')
    }
    
    return result


def evaluate_regression_model(
    model: Union[Any, lgb.Booster],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_features: list = None,
    model_type: str = 'sklearn',
    best_iteration: int = None
) -> Tuple[Dict[str, float], str]:
    """
    Evaluate a trained regression model on test data.
    
    Args:
        model: Trained model (sklearn model or LightGBM Booster)
        X_test: Test features
        y_test: Test target
        categorical_features: List of categorical feature names (for LightGBM)
        model_type: Type of model ('sklearn' or 'lightgbm')
        best_iteration: Best iteration for LightGBM model
        
    Returns:
        Tuple of (metrics_dict, evaluation_report_str)
    """
    # Prepare test data based on model type
    if model_type == 'lightgbm' and categorical_features:
        # Convert categorical features to 'category' dtype for LightGBM
        for cat_col in categorical_features:
            if cat_col in X_test.columns:
                X_test[cat_col] = X_test[cat_col].astype('category')
    
    # Get predictions
    if model_type == 'sklearn':
        y_test_pred = model.predict(X_test)
    else:  # lightgbm
        y_test_pred = model.predict(X_test, num_iteration=best_iteration)
    
    # Calculate metrics
    from .train import calculate_regression_metrics
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    
    # Create formatted output
    output = f"\n{'='*60}\n"
    output += f"REGRESSION MODEL EVALUATION - TEST SET\n"
    output += f"{'='*60}\n"
    output += f"Root Mean Squared Error (RMSE): {test_metrics['rmse']:.4f}\n"
    output += f"Mean Absolute Error (MAE):      {test_metrics['mae']:.4f}\n"
    output += f"R-squared (R²):                 {test_metrics['r2']:.4f}\n"
    output += f"Mean Squared Error (MSE):       {test_metrics['mse']:.4f}\n"
    output += f"{'='*60}\n"
    
    # Add additional statistics
    residuals = y_test - y_test_pred
    output += f"\nAdditional Statistics:\n"
    output += f"Mean of predictions:     {np.mean(y_test_pred):.4f}\n"
    output += f"Std of predictions:      {np.std(y_test_pred):.4f}\n"
    output += f"Mean of actual values:   {np.mean(y_test):.4f}\n"
    output += f"Std of actual values:    {np.std(y_test):.4f}\n"
    output += f"Mean residual:          {np.mean(residuals):.4f}\n"
    output += f"Std of residuals:       {np.std(residuals):.4f}\n"
    
    return test_metrics, output