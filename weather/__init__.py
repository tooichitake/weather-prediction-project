"""
Weather prediction package for Sydney weather forecasting
"""

__version__ = "0.1.0"

# Data collection
from .dataset import WeatherDataCollector, download_sydney_weather_data

# Feature engineering
from .features import (
    create_rain_on_day_7_target,
    create_precipitation_3day_target,
    extract_month_feature,
    get_feature_list,
    get_precipitation_feature_list,
    split_features_by_type
)

# Model training utilities
from .modeling.train import (
    train_val_test_split,
    create_preprocessor,
    prepare_lightgbm_datasets,
    optimize_threshold,
    optimize_logistic_regression,
    optimize_lightgbm,
    calculate_metrics,
    calculate_regression_metrics,
    optimize_linear_regression_reg,
    optimize_lightgbm_regression,
    save_model,
    save_lightgbm_model,
    save_sklearn_model
)

# Model prediction utilities
from .modeling.predict import (
    evaluate_model,
    evaluate_regression_model,
    predict_with_model,
    load_model_and_info
)

# Visualization functions
from .plots import (
    set_plot_style,
    plot_target_distribution,
    plot_feature_exploration_4panels,
    plot_feature_exploration_2panels,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_monthly_probabilities,
    plot_model_performance,
    plot_regression_performance,
    plot_precipitation_distribution,
    plot_feature_vs_target_regression
)

__all__ = [
    # Dataset
    "WeatherDataCollector", 
    "download_sydney_weather_data",
    # Features
    "create_rain_on_day_7_target",
    "create_precipitation_3day_target",
    "extract_month_feature", 
    "get_feature_list",
    "get_precipitation_feature_list",
    "split_features_by_type",
    # Training
    "train_val_test_split",
    "create_preprocessor",
    "prepare_lightgbm_datasets",
    "optimize_threshold",
    "optimize_logistic_regression",
    "optimize_lightgbm",
    "calculate_metrics",
    "calculate_regression_metrics",
    "optimize_linear_regression_reg",
    "optimize_lightgbm_regression",
    "save_model",
    "save_lightgbm_model",
    "save_sklearn_model",
    # Prediction
    "evaluate_model",
    "evaluate_regression_model",
    "predict_with_model",
    "load_model_and_info",
    # Plots
    "set_plot_style",
    "plot_target_distribution",
    "plot_feature_exploration_4panels",
    "plot_feature_exploration_2panels",
    "plot_correlation_heatmap",
    "plot_feature_importance",
    "plot_monthly_probabilities",
    "plot_model_performance",
    "plot_regression_performance",
    "plot_precipitation_distribution",
    "plot_feature_vs_target_regression"
]