"""Feature engineering functions for weather prediction models."""

import calendar
import pandas as pd
import numpy as np
from typing import Union, List, Tuple


def create_rain_on_day_7_target(df: pd.DataFrame) -> pd.Series:
    """
    Create target variable indicating if it will rain on day 7 (exactly 7 days from today).
    
    Args:
        df: DataFrame with 'rain_sum' column
        
    Returns:
        Series with binary target variable (1 for rain, 0 for no rain)
    """
    target = []
    
    for i in range(len(df) - 7):
        # Check if it will rain exactly 7 days from today
        rain_7_days_ahead = df['rain_sum'].iloc[i+7]
        will_rain = int(rain_7_days_ahead > 0)
        target.append(will_rain)
    
    # Last 7 days cannot have target (no future data)
    target.extend([np.nan] * 7)
    
    return pd.Series(target, index=df.index)


def extract_month_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract month from time column as categorical feature.
    
    Args:
        df: DataFrame with 'time' column
        
    Returns:
        DataFrame with 'month' column added
    """
    df_copy = df.copy()
    
    # Convert time column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy['time']):
        df_copy['date'] = pd.to_datetime(df_copy['time'])
    else:
        df_copy['date'] = df_copy['time']
    
    # Extract month number and convert to English month name
    df_copy['month'] = df_copy['date'].dt.month.apply(lambda x: calendar.month_name[x])
    
    return df_copy


def get_feature_list(df: pd.DataFrame, target_name: str = 'rain_on_day_7') -> List[str]:
    """
    Get the list of features for modeling, excluding redundant features.
    
    Args:
        df: DataFrame with all columns
        target_name: Name of the target column to exclude
        
    Returns:
        List of feature names
    """
    # Exclude redundant features:
    # - precipitation_sum, snowfall_sum (redundant with rain_sum in Sydney)
    # - sunrise, sunset (redundant with daylight_duration)
    # - target variable
    exclude_features = [
        target_name,
        'precipitation_sum', 
        'snowfall_sum',
        'sunrise', 
        'sunset',
        'date',  # Temporary datetime column
        'time'   # Exclude time since we use month instead
    ]
    
    features_list = [col for col in df.columns if col not in exclude_features]
    
    return features_list


def split_features_by_type(
    features: List[str],
    categorical_features: List[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Split features into categorical and numerical lists.
    
    Args:
        features: List of all feature names
        categorical_features: List of known categorical features
        
    Returns:
        Tuple of (categorical_features, numerical_features)
    """
    if categorical_features is None:
        categorical_features = ['month', 'weather_code']
    
    # Filter to only include categorical features that exist in the features list
    cat_features = [f for f in categorical_features if f in features]
    
    # All other features are numerical
    num_features = [f for f in features if f not in cat_features]
    
    return cat_features, num_features


def create_precipitation_3day_target(df: pd.DataFrame) -> pd.Series:
    """
    Create target variable: cumulative precipitation for the next 3 days.
    
    Args:
        df: DataFrame with 'precipitation_sum' column
        
    Returns:
        Series with 3-day cumulative precipitation (next 3 days from tomorrow)
    """
    target = []
    
    for i in range(len(df) - 3):
        # Sum precipitation for the next 3 days (days i+1, i+2, i+3)
        # Not including today (day i)
        precip_next_3days = df['precipitation_sum'].iloc[i+1:i+4].sum()
        target.append(precip_next_3days)
    
    # Last 3 days cannot have target (no future data)
    target.extend([np.nan] * 3)
    
    return pd.Series(target, index=df.index)


def get_precipitation_feature_list(df: pd.DataFrame, target_name: str = 'precipitation_3day') -> List[str]:
    """
    Get the list of features for precipitation regression modeling.
    
    Args:
        df: DataFrame with all columns
        target_name: Name of the target column to exclude
        
    Returns:
        List of feature names
    """
    # For precipitation prediction, we include precipitation_sum as a feature
    # since current precipitation can be predictive of future precipitation
    exclude_features = [
        target_name,
        'snowfall_sum',  # No snow in Sydney
        'rain_sum',  # Same as precipitation_sum in Sydney (no snow)
        'sunrise', 
        'sunset',
        'date',  # Temporary datetime column
        'time'   # Exclude time since we use month instead
    ]
    
    features_list = [col for col in df.columns if col not in exclude_features]
    
    return features_list