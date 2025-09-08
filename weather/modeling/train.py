"""Model training utilities for weather prediction models."""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from typing import Tuple, List, Dict, Any, Optional, Callable
import lightgbm as lgb
import optuna
from optuna.logging import set_verbosity


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/val/test sets with optional stratification.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random seed
        stratify: Whether to use stratification. If None, auto-detect based on target type
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Auto-detect whether to use stratification if not specified
    if stratify is None:
        # Check if target is binary/categorical (for classification) or continuous (for regression)
        unique_values = y.nunique()
        is_classification = unique_values < 20 and (y.dtype == 'int' or y.dtype == 'bool')
        stratify = is_classification
    
    # First split: separate test set
    if stratify:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Calculate validation size from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    
    # Second split: separate validation set
    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    use_standard_scaler: bool = True
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for features.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        use_standard_scaler: Whether to standardize numerical features
        
    Returns:
        ColumnTransformer for preprocessing
    """
    transformers = []
    
    # Numerical features preprocessing
    if numerical_features:
        if use_standard_scaler:
            transformers.append(
                ('num', Pipeline([('scaler', StandardScaler())]), numerical_features)
            )
        else:
            # For LightGBM, we might not need scaling
            transformers.append(
                ('num', 'passthrough', numerical_features)
            )
    
    # Categorical features preprocessing
    if categorical_features:
        transformers.append(
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        )
    
    return ColumnTransformer(transformers=transformers)


def prepare_lightgbm_datasets(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    categorical_features: List[str] = None
) -> Tuple[lgb.Dataset, lgb.Dataset]:
    """
    Prepare LightGBM datasets with native categorical support.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        categorical_features: List of categorical feature names
        
    Returns:
        Tuple of (train_data, val_data) as LightGBM Dataset objects
    """
    # Convert categorical features to category dtype for LightGBM
    if categorical_features:
        for cat_col in categorical_features:
            if cat_col in X_train.columns:
                X_train[cat_col] = X_train[cat_col].astype('category')
                X_val[cat_col] = X_val[cat_col].astype('category')
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=categorical_features if categorical_features else 'auto'
    )
    
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_data,
        categorical_feature=categorical_features if categorical_features else 'auto'
    )
    
    return train_data, val_data


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    n_trials: int = 80
) -> Tuple[float, float]:
    """
    Optimize classification threshold for a given metric.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
        n_trials: Number of Optuna trials
        
    Returns:
        Tuple of (best_threshold, best_score)
    """
    set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        threshold = trial.suggest_float('threshold', 0.1, 0.9)
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            from sklearn.metrics import f1_score
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            from sklearn.metrics import precision_score
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            from sklearn.metrics import recall_score
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params['threshold'], study.best_value


def optimize_logistic_regression(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame, 
    y_train: pd.Series,
    y_val: pd.Series,
    fixed_params: Dict[str, Any],
    metric: str = 'auc',
    n_trials: int = 100
) -> Tuple[Dict[str, Any], float, LogisticRegression]:
    """
    Optimize logistic regression hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        fixed_params: Fixed hyperparameters (e.g., solver, max_iter, random_state)
        metric: Metric to optimize ('auc' or any sklearn metric)
        n_trials: Number of optimization trials
        
    Returns:
        Tuple of (best_params, best_score, trained_model)
    """
    set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        # Suggest C parameter
        C = trial.suggest_float('C', 0.001, 100.0, log=True)
        
        # Create model with suggested parameters
        model_params = fixed_params.copy()
        model_params['C'] = C
        
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
        
        # Get predictions
        if metric == 'auc':
            y_val_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_val_proba)
        else:
            y_val_pred = model.predict(X_val)
            if metric == 'f1':
                score = f1_score(y_val, y_val_pred)
            elif metric == 'precision':
                score = precision_score(y_val, y_val_pred)
            elif metric == 'recall':
                score = recall_score(y_val, y_val_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train final model with best parameters
    best_params = study.best_params
    final_params = fixed_params.copy()
    final_params.update(best_params)
    
    final_model = LogisticRegression(**final_params)
    final_model.fit(X_train, y_train)
    
    return best_params, study.best_value, final_model


def optimize_lightgbm(
    train_data: lgb.Dataset,
    val_data: lgb.Dataset,
    fixed_params: Dict[str, Any],
    param_space: Dict[str, Any] = None,
    metric: str = 'auc',
    n_trials: int = 50,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50
) -> Tuple[Dict[str, Any], float, lgb.Booster]:
    """
    Optimize LightGBM hyperparameters using Optuna.
    
    Args:
        train_data: LightGBM training dataset
        val_data: LightGBM validation dataset  
        fixed_params: Fixed hyperparameters (e.g., objective, seed, verbose)
        param_space: Optional custom parameter space for optimization
        metric: Metric to optimize 
        n_trials: Number of optimization trials
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Early stopping patience
        
    Returns:
        Tuple of (best_params, best_score, trained_model)
    """
    set_verbosity(optuna.logging.WARNING)
    
    # Default parameter space if not provided
    if param_space is None:
        param_space = {
            'num_leaves': ('int', 20, 300),
            'learning_rate': ('float', 0.01, 0.3, True),  # log=True
            'feature_fraction': ('float', 0.4, 1.0),
            'bagging_fraction': ('float', 0.4, 1.0),
            'bagging_freq': ('int', 1, 7),
            'lambda_l1': ('float', 1e-8, 10.0, True),  # log=True
            'lambda_l2': ('float', 1e-8, 10.0, True),  # log=True
        }
    
    def objective(trial):
        # Suggest hyperparameters based on param_space
        params = fixed_params.copy()
        
        for param_name, param_config in param_space.items():
            if param_config[0] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_config[0] == 'float':
                log = param_config[3] if len(param_config) > 3 else False
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=log)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(0)
            ]
        )
        
        # Return validation score
        best_score = model.best_score['valid_0'][metric]
        return best_score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train final model with best parameters
    best_params = study.best_params
    final_params = fixed_params.copy()
    final_params.update(best_params)
    
    final_model = lgb.train(
        final_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(0)
        ]
    )
    
    return best_params, study.best_value, final_model


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate common classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metric names and values
    """
    from sklearn.metrics import accuracy_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def save_model(
    model: Any,
    model_path: str,
    model_info: Dict[str, Any],
    info_path: str,
    model_type: str = 'sklearn',
    best_iteration: int = None
) -> None:
    """
    Save a trained model and its associated information.
    
    Args:
        model: Trained model (sklearn Pipeline or LightGBM Booster)
        model_path: Path to save the model file
        model_info: Dictionary containing model metadata
        info_path: Path to save the model info JSON file
        model_type: Type of model ('sklearn' or 'lightgbm')
        best_iteration: Best iteration for LightGBM model (used for saving)
        
    Returns:
        None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model based on type
    if model_type == 'sklearn':
        joblib.dump(model, model_path)
    else:  # lightgbm
        model.save_model(model_path, num_iteration=best_iteration)
    
    print(f"Model saved to: {model_path}")
    
    # Save model info
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"Model info saved to: {info_path}")


def save_lightgbm_model(
    model: lgb.Booster,
    best_params: Dict[str, Any],
    fixed_params: Dict[str, Any],
    best_threshold: float,
    best_iteration: int,
    primary_metric: str,
    secondary_metric: str,
    val_metrics: Dict[str, float],
    feature_names: List[str],
    categorical_features: List[str],
    model_dir: str,
    model_name: str = 'rain_or_not_lightgbm'
) -> Tuple[str, str]:
    """
    Save a LightGBM model with all its metadata.
    
    Args:
        model: Trained LightGBM Booster
        best_params: Best hyperparameters from optimization
        fixed_params: Fixed parameters used in training
        best_threshold: Optimized classification threshold
        best_iteration: Best iteration from early stopping
        primary_metric: Primary evaluation metric
        secondary_metric: Secondary evaluation metric
        val_metrics: Validation metrics
        feature_names: List of feature names
        categorical_features: List of categorical feature names
        model_dir: Directory to save the model
        model_name: Base name for the model files
        
    Returns:
        Tuple of (model_path, info_path)
    """
    # Prepare paths
    model_path = os.path.join(model_dir, f'{model_name}.txt')
    info_path = os.path.join(model_dir, f'{model_name}_info.json')
    
    # Prepare final parameters
    final_params = fixed_params.copy()
    final_params.update(best_params)
    
    # Prepare model info
    model_info = {
        'best_params': best_params,
        'final_params': final_params,
        'best_threshold': best_threshold,
        'best_iteration': best_iteration,
        'primary_metric': primary_metric,
        'secondary_metric': secondary_metric,
        'validation_metrics': val_metrics,
        'feature_names': feature_names,
        'categorical_features': categorical_features
    }
    
    # Save model and info
    save_model(
        model=model,
        model_path=model_path,
        model_info=model_info,
        info_path=info_path,
        model_type='lightgbm',
        best_iteration=best_iteration
    )
    
    return model_path, info_path


def save_sklearn_model(
    model: Any,
    best_params: Dict[str, Any],
    fixed_params: Dict[str, Any],
    best_threshold: float,
    primary_metric: str,
    secondary_metric: str,
    val_metrics: Dict[str, float],
    feature_names: List[str],
    model_dir: str,
    model_name: str = 'rain_or_not_sklearn'
) -> Tuple[str, str]:
    """
    Save a scikit-learn model (e.g., LogisticRegression pipeline) with all its metadata.
    
    Args:
        model: Trained sklearn model/pipeline
        best_params: Best hyperparameters from optimization
        fixed_params: Fixed parameters used in training
        best_threshold: Optimized classification threshold
        primary_metric: Primary evaluation metric
        secondary_metric: Secondary evaluation metric
        val_metrics: Validation metrics
        feature_names: List of feature names
        model_dir: Directory to save the model
        model_name: Base name for the model files
        
    Returns:
        Tuple of (model_path, info_path)
    """
    # Prepare paths
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    info_path = os.path.join(model_dir, f'{model_name}_info.json')
    
    # Prepare final parameters
    final_params = fixed_params.copy()
    final_params.update(best_params)
    
    # Prepare model info
    model_info = {
        'best_params': best_params,
        'final_params': final_params,
        'best_threshold': best_threshold,
        'primary_metric': primary_metric,
        'secondary_metric': secondary_metric,
        'validation_metrics': val_metrics,
        'feature_names': feature_names
    }
    
    # Save model and info
    save_model(
        model=model,
        model_path=model_path,
        model_info=model_info,
        info_path=info_path,
        model_type='sklearn'
    )
    
    return model_path, info_path


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate common regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric names and values
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }
    
    return metrics


def optimize_linear_regression_reg(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    fixed_params: Dict[str, Any] = None,
    metric: str = 'rmse',
    n_trials: int = 100
) -> Tuple[Dict[str, Any], float, Any]:
    """
    Optimize Ridge regression hyperparameters for regression tasks.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        fixed_params: Fixed hyperparameters
        metric: Metric to optimize ('rmse', 'mae', 'r2')
        n_trials: Number of optimization trials
        
    Returns:
        Tuple of (best_params, best_score, trained_model)
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    set_verbosity(optuna.logging.WARNING)
    
    if fixed_params is None:
        fixed_params = {'random_state': 42}
    
    def objective(trial):
        # Suggest alpha parameter for Ridge regression
        alpha = trial.suggest_float('alpha', 0.0001, 100.0, log=True)
        
        # Create Ridge model
        model = Ridge(alpha=alpha, **fixed_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and calculate metric
        y_val_pred = model.predict(X_val)
        
        if metric == 'rmse':
            score = -np.sqrt(mean_squared_error(y_val, y_val_pred))
        elif metric == 'mae':
            score = -mean_absolute_error(y_val, y_val_pred)
        elif metric == 'r2':
            score = r2_score(y_val, y_val_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return score
    
    # Optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train final model with best parameters
    best_params = study.best_params
    alpha = best_params['alpha']
    
    final_model = Ridge(alpha=alpha, **fixed_params)
    final_model.fit(X_train, y_train)
    
    return best_params, -study.best_value if metric != 'r2' else study.best_value, final_model


def optimize_lightgbm_regression(
    train_data: lgb.Dataset,
    val_data: lgb.Dataset,
    fixed_params: Dict[str, Any],
    param_space: Dict[str, Any] = None,
    metric: str = 'rmse',
    n_trials: int = 50,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50
) -> Tuple[Dict[str, Any], float, lgb.Booster]:
    """
    Optimize LightGBM hyperparameters for regression tasks.
    
    Args:
        train_data: LightGBM training dataset
        val_data: LightGBM validation dataset  
        fixed_params: Fixed hyperparameters
        param_space: Optional custom parameter space for optimization
        metric: Metric to optimize 
        n_trials: Number of optimization trials
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Early stopping patience
        
    Returns:
        Tuple of (best_params, best_score, trained_model)
    """
    set_verbosity(optuna.logging.WARNING)
    
    # Default parameter space if not provided
    if param_space is None:
        param_space = {
            'num_leaves': ('int', 20, 300),
            'learning_rate': ('float', 0.01, 0.3, True),  # log=True
            'feature_fraction': ('float', 0.4, 1.0),
            'bagging_fraction': ('float', 0.4, 1.0),
            'bagging_freq': ('int', 1, 7),
            'lambda_l1': ('float', 1e-8, 10.0, True),  # log=True
            'lambda_l2': ('float', 1e-8, 10.0, True),  # log=True
        }
    
    def objective(trial):
        # Suggest hyperparameters based on param_space
        params = fixed_params.copy()
        
        for param_name, param_config in param_space.items():
            if param_config[0] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_config[0] == 'float':
                log = param_config[3] if len(param_config) > 3 else False
                params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=log)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(0)
            ]
        )
        
        # Return validation score (negative for minimization metrics)
        best_score = model.best_score['valid_0'][metric]
        return -best_score if metric in ['rmse', 'mae', 'mse'] else best_score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train final model with best parameters
    best_params = study.best_params
    final_params = fixed_params.copy()
    final_params.update(best_params)
    
    final_model = lgb.train(
        final_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(0)
        ]
    )
    
    best_score = final_model.best_score['valid_0'][metric]
    return best_params, best_score, final_model