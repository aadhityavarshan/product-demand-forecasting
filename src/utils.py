"""Utility functions for data loading, preprocessing, and evaluation."""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import os
import joblib
from pathlib import Path


def load_and_preprocess(data_path, date_col='Date', demand_col='Order_Demand'):
    """
    Load and preprocess the demand data.
    
    Args:
        data_path: Path to the CSV file
        date_col: Name of the date column
        demand_col: Name of the demand column
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(data_path)
    df.dropna(subset=[date_col, demand_col], inplace=True)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')
    df.dropna(subset=[demand_col], inplace=True)
    return df


def aggregate_demand(df, date_col='Date', demand_col='Order_Demand', freq='M'):
    """
    Aggregate demand data by specified frequency.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        demand_col: Name of the demand column
        freq: Aggregation frequency ('D' for daily, 'M' for monthly, etc.)
        
    Returns:
        Aggregated Series indexed by date
    """
    df_copy = df.set_index(date_col)
    return df_copy[demand_col].resample(freq).sum()


def train_test_split_timeseries(series, train_ratio=0.8):
    """
    Split time series data into train and test sets.
    
    Args:
        series: Time series data
        train_ratio: Ratio of training data (0 to 1)
        
    Returns:
        train, test: Split time series data
    """
    train_size = int(len(series) * train_ratio)
    return series[:train_size], series[train_size:]


def calculate_metrics(actual, predicted):
    """
    Calculate evaluation metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with RMSE, MAE, and R²
    """
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def save_model(model, filename, models_dir='models'):
    """
    Save a trained model using joblib.
    
    Args:
        model: Trained model object
        filename: Name of the file (without directory)
        models_dir: Directory to save the model
    """
    Path(models_dir).mkdir(exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filename, models_dir='models'):
    """
    Load a trained model using joblib.
    
    Args:
        filename: Name of the file (without directory)
        models_dir: Directory where the model is saved
        
    Returns:
        Loaded model object
    """
    filepath = os.path.join(models_dir, filename)
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        print(f"Model file not found: {filepath}")
        return None


def print_metrics(model_name, metrics):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary with evaluation metrics
    """
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")
