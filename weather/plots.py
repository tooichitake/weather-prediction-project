"""Visualization functions for weather prediction models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve, 
                           roc_auc_score, f1_score)
from typing import Dict, List, Optional, Tuple, Union


# Set default style
def set_plot_style():
    """Set consistent plot styling."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    labels_dict: Optional[Dict[int, str]] = None,
    colors: List[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot binary target variable distribution with counts and percentages.
    
    Args:
        df: DataFrame containing target variable
        target_col: Name of target column
        labels_dict: Dictionary mapping target values to labels
        colors: List of colors for each class
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if colors is None:
        colors = ['#3498db', '#e74c3c']
    
    if labels_dict is None:
        labels_dict = {0: 'No Rain', 1: 'Rain'}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create visualization copy
    df_viz = df.copy()
    df_viz['Target_Label'] = df_viz[target_col].map(labels_dict)
    
    # Create countplot
    order = [labels_dict[0], labels_dict[1]]
    sns.countplot(data=df_viz, x='Target_Label', palette=colors, order=order, ax=ax)
    
    ax.set_xlabel('Target Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    
    if title is None:
        title = f'Distribution of {target_col}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add count and percentage labels
    counts = df_viz['Target_Label'].value_counts()
    total = counts.sum()
    
    for i, label in enumerate(order):
        count = counts[label]
        percentage = count / total * 100
        ax.text(i, count + 10, f'{count:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_feature_exploration_4panels(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    figsize: Tuple[int, int] = (15, 10),
    rolling_window: int = 30,
    n_bins: int = 10,
    time_col: str = 'time'
) -> plt.Figure:
    """
    Create 4-panel exploration plot for a feature.
    
    Args:
        df: DataFrame containing data
        feature_col: Name of feature to explore
        target_col: Name of target column
        figsize: Figure size
        rolling_window: Window size for rolling mean
        n_bins: Number of bins for quantile analysis
        time_col: Name of time column
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Distribution of feature
    ax1 = axes[0, 0]
    sns.histplot(data=df, x=feature_col, bins=50, kde=True, ax=ax1, color='#3498db')
    ax1.set_xlabel(f'{feature_col}', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Distribution of {feature_col}', fontsize=14, fontweight='bold')
    
    # 2. Box plot by target
    ax2 = axes[0, 1]
    df_viz = df.copy()
    df_viz['Target_Category'] = df_viz[target_col].map({0: 'No Rain', 1: 'Rain'})
    sns.boxplot(data=df_viz, x='Target_Category', y=feature_col, 
                order=['No Rain', 'Rain'], ax=ax2, palette=['#3498db', '#e74c3c'])
    ax2.set_xlabel('Target', fontsize=12)
    ax2.set_ylabel(feature_col, fontsize=12)
    ax2.set_title(f'{feature_col} by Target', fontsize=14, fontweight='bold')
    
    # 3. Time series with rolling mean
    ax3 = axes[1, 0]
    df_ts = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_ts[time_col]):
        df_ts['date_temp'] = pd.to_datetime(df_ts[time_col])
    else:
        df_ts['date_temp'] = df_ts[time_col]
    
    rolling_mean = df_ts.set_index('date_temp')[feature_col].rolling(rolling_window).mean()
    sns.lineplot(data=rolling_mean, ax=ax3, color='#16a085', linewidth=2)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel(f'{rolling_window}-day Rolling Mean', fontsize=12)
    ax3.set_title(f'{feature_col} Trend Over Time', fontsize=14, fontweight='bold')
    
    # 4. Binned feature vs target probability
    ax4 = axes[1, 1]
    # Filter only non-zero values if appropriate
    df_analysis = df[df[feature_col] > 0] if feature_col in ['precipitation_sum', 'rain_sum'] else df
    
    try:
        feature_bins = pd.qcut(df_analysis[feature_col], q=n_bins, duplicates='drop')
        prob_by_bin = df_analysis.groupby(feature_bins)[target_col].mean()
        
        # Get bin midpoints
        bin_midpoints = [interval.mid for interval in prob_by_bin.index]
        
        # Plot
        ax4.scatter(bin_midpoints, prob_by_bin.values, 
                   s=150, color='#e74c3c', alpha=0.8, edgecolors='black', linewidth=1)
        ax4.plot(bin_midpoints, prob_by_bin.values, 
                color='#e74c3c', alpha=0.5, linewidth=2)
        
        ax4.set_xlabel(feature_col, fontsize=12)
        ax4.set_ylabel('Probability of Rain', fontsize=12)
        ax4.set_title(f'{feature_col} vs Target Probability', fontsize=14, fontweight='bold')
        
        # Add annotations
        for i, (x, y) in enumerate(zip(bin_midpoints, prob_by_bin.values)):
            ax4.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    except Exception:
        ax4.text(0.5, 0.5, 'Unable to create bins', 
                transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    return fig


def plot_feature_exploration_2panels(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    figsize: Tuple[int, int] = (14, 6),
    n_bins: int = 10
) -> plt.Figure:
    """
    Create 2-panel exploration plot for a feature.
    
    Args:
        df: DataFrame containing data
        feature_col: Name of feature to explore
        target_col: Name of target column
        figsize: Figure size
        n_bins: Number of bins for quantile analysis
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Box plot by target
    df_viz = df.copy()
    df_viz['Target_Category'] = df_viz[target_col].map({0: 'No Rain', 1: 'Rain'})
    sns.boxplot(data=df_viz, x='Target_Category', y=feature_col, 
                order=['No Rain', 'Rain'], ax=ax1, palette=['#3498db', '#e74c3c'])
    ax1.set_xlabel('Target', fontsize=12)
    ax1.set_ylabel(feature_col, fontsize=12)
    ax1.set_title(f'{feature_col} Distribution by Target', fontsize=14, fontweight='bold')
    
    # 2. Binned feature vs rain probability
    try:
        feature_bins = pd.qcut(df[feature_col], q=n_bins, duplicates='drop')
        prob_by_bin = df.groupby(feature_bins)[target_col].mean()
        
        # Get bin midpoints
        bin_midpoints = [interval.mid for interval in prob_by_bin.index]
        
        # Plot
        ax2.scatter(bin_midpoints, prob_by_bin.values, 
                   s=150, color='#e74c3c', alpha=0.8, edgecolors='black', linewidth=1)
        ax2.plot(bin_midpoints, prob_by_bin.values, 
                color='#e74c3c', alpha=0.5, linewidth=2)
        
        ax2.set_xlabel(feature_col, fontsize=12)
        ax2.set_ylabel('Probability of Rain', fontsize=12)
        ax2.set_title(f'{feature_col} vs Rain Probability', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations
        for i, (x, y) in enumerate(zip(bin_midpoints, prob_by_bin.values)):
            ax2.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
    except Exception:
        ax2.text(0.5, 0.5, 'Unable to create bins', 
                transform=ax2.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 12),
    mask_upper: bool = True
) -> plt.Figure:
    """
    Plot correlation heatmap with optional upper triangle mask.
    
    Args:
        df: DataFrame containing data
        numeric_cols: List of numeric columns to include
        exclude_cols: List of columns to exclude
        figsize: Figure size
        mask_upper: Whether to mask upper triangle
        
    Returns:
        matplotlib Figure object
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8},
                ax=ax)
    
    ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    scores_df: pd.DataFrame,
    score_col: str = 'score',
    feature_col: str = 'feature',
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot horizontal bar chart of feature importance scores.
    
    Args:
        scores_df: DataFrame with feature names and scores
        score_col: Name of score column
        feature_col: Name of feature column
        top_n: Number of top features to show
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Sort and get top features
    top_features = scores_df.nlargest(top_n, score_col)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create barplot
    sns.barplot(data=top_features, x=score_col, y=feature_col, 
                palette='viridis', ax=ax)
    
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    if title is None:
        title = f'Top {top_n} Features by Importance'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_monthly_probabilities(
    df: pd.DataFrame,
    target_col: str,
    month_col: str = 'month',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot monthly probability distribution with average line.
    
    Args:
        df: DataFrame containing data
        target_col: Name of target column
        month_col: Name of month column
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate monthly probabilities
    monthly_prob = df.groupby(month_col)[target_col].mean()
    
    # Define month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Reorder data
    monthly_prob = monthly_prob.reindex(month_order)
    
    # Create bar plot
    sns.barplot(x=month_order, y=monthly_prob.values, 
                palette='viridis', alpha=0.8, ax=ax)
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Probability of Rain', fontsize=12)
    ax.set_title('Monthly Rain Probability', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    for i, (month, prob) in enumerate(zip(month_order, monthly_prob.values)):
        ax.text(i, prob + 0.01, f'{prob:.1%}', 
                ha='center', va='bottom', fontsize=10)
    
    # Add overall average line
    overall_avg = df[target_col].mean()
    ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.5, 
               label=f'Overall Average: {overall_avg:.1%}')
    ax.legend()
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (15, 12),
    model_name: str = ""
) -> plt.Figure:
    """
    Create 4-panel model performance visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        threshold: Classification threshold used
        figsize: Figure size
        model_name: Name of model for title
        
    Returns:
        matplotlib Figure object
    """
    # Calculate metrics
    test_auc = roc_auc_score(y_true, y_proba)
    test_f1 = f1_score(y_true, y_pred, pos_label=1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['No Rain', 'Rain'], 
                yticklabels=['No Rain', 'Rain'],
                annot_kws={'size': 14})
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_xlabel('Predicted', fontsize=12)
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm.sum() * 100
            ax1.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='darkblue')
    
    # 2. ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax2.plot(fpr, tpr, color='#e74c3c', linewidth=3, label=f'ROC (AUC = {test_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax2.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = axes[1, 0]
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_true, y_proba)
    ax3.plot(rec_curve, prec_curve, color='#16a085', linewidth=3, 
             label=f'PR Curve (F1 = {test_f1:.3f})')
    ax3.axhline(y=y_true.mean(), color='k', linestyle='--', linewidth=2, 
                label=f'Baseline (Positive Rate = {y_true.mean():.3f})')
    ax3.fill_between(rec_curve, prec_curve, alpha=0.3, color='#16a085')
    
    # Mark operating point
    operating_idx = np.argmin(np.abs(thresholds_pr - threshold))
    ax3.plot(rec_curve[operating_idx], prec_curve[operating_idx], 'ro', markersize=10, 
             label=f'Operating Point (threshold={threshold:.2f})')
    
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Probability Distribution by Class
    ax4 = axes[1, 1]
    no_rain_probs = y_proba[y_true == 0]
    rain_probs = y_proba[y_true == 1]
    
    ax4.hist(no_rain_probs, bins=30, alpha=0.6, label='No Rain (Actual)', 
             color='#3498db', density=True, edgecolor='black')
    ax4.hist(rain_probs, bins=30, alpha=0.6, label='Rain (Actual)', 
             color='#e74c3c', density=True, edgecolor='black')
    ax4.axvline(x=threshold, color='green', linestyle='--', linewidth=3, 
                label=f'Threshold = {threshold:.3f}')
    ax4.set_xlabel('Predicted Probability', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    if model_name:
        fig.suptitle(f'{model_name} Performance', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_regression_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Create comprehensive regression performance visualization with 4 panels.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if model_name:
        fig.suptitle(f'{model_name} Regression Performance', fontsize=16, y=0.98)
    
    # 1. Actual vs Predicted scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax1.plot(sorted(y_true), p(sorted(y_true)), "b-", alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax1.set_xlabel('Actual Values (mm)')
    ax1.set_ylabel('Predicted Values (mm)')
    ax1.set_title('Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax2 = axes[0, 1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    
    # Add ±1 std bands
    std_residual = residuals.std()
    ax2.axhline(y=std_residual, color='orange', linestyle='--', alpha=0.7, label=f'±1 STD ({std_residual:.2f})')
    ax2.axhline(y=-std_residual, color='orange', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Predicted Values (mm)')
    ax2.set_ylabel('Residuals (mm)')
    ax2.set_title('Residual Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Overlay normal distribution
    from scipy import stats
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax3.plot(x, p, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
    
    ax3.set_xlabel('Residuals (mm)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Normal Q-Q Plot')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_precipitation_distribution(
    df: pd.DataFrame,
    target_col: str = 'precipitation_3day',
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot distribution of precipitation values with and without zeros.
    
    Args:
        df: DataFrame with precipitation data
        target_col: Column name for precipitation values
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: All values including zeros
    ax1.hist(df[target_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('3-Day Precipitation (mm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of {target_col}\n(All Values)')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.text(0.95, 0.95, f'Mean: {df[target_col].mean():.2f} mm\n'
                        f'Median: {df[target_col].median():.2f} mm\n'
                        f'Std: {df[target_col].std():.2f} mm\n'
                        f'Zero days: {(df[target_col] == 0).sum()} ({(df[target_col] == 0).mean():.1%})',
             transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right plot: Non-zero values only
    non_zero = df[df[target_col] > 0][target_col]
    if len(non_zero) > 0:
        ax2.hist(non_zero, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('3-Day Precipitation (mm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Distribution of {target_col}\n(Non-Zero Values Only)')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        ax2.text(0.95, 0.95, f'Mean: {non_zero.mean():.2f} mm\n'
                            f'Median: {non_zero.median():.2f} mm\n'
                            f'Std: {non_zero.std():.2f} mm\n'
                            f'Count: {len(non_zero)}',
                 transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_feature_vs_target_regression(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot relationship between a feature and regression target.
    
    Args:
        df: DataFrame with feature and target
        feature_col: Feature column name
        target_col: Target column name
        n_bins: Number of bins for grouping
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bins for the feature
    df_copy = df[[feature_col, target_col]].copy()
    df_copy['bin'] = pd.qcut(df_copy[feature_col], q=n_bins, duplicates='drop')
    
    # Calculate statistics per bin
    bin_stats = df_copy.groupby('bin')[target_col].agg(['mean', 'std', 'count'])
    bin_centers = df_copy.groupby('bin')[feature_col].mean()
    
    # Plot with error bars
    ax.errorbar(bin_centers, bin_stats['mean'], yerr=bin_stats['std'], 
                fmt='o-', capsize=5, capthick=2, markersize=8)
    
    ax.set_xlabel(f'{feature_col}')
    ax.set_ylabel(f'Mean {target_col} (mm)')
    ax.set_title(f'{target_col} vs {feature_col}')
    ax.grid(True, alpha=0.3)
    
    # Add count information
    for i, (x, y, count) in enumerate(zip(bin_centers, bin_stats['mean'], bin_stats['count'])):
        ax.annotate(f'n={count}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    return fig