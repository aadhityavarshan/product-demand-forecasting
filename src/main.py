"""
Main orchestration script for demand forecasting.
Loads data, trains models, compares performance, and generates future forecasts.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import config
from utils import (
    load_and_preprocess, 
    aggregate_demand,
    train_test_split_timeseries,
    calculate_metrics,
    save_model,
    print_metrics
)


def train_arima_model(series, order=config.ARIMA_ORDER):
    """Train ARIMA model."""
    print("\n" + "="*60)
    print("Training ARIMA Model...")
    print("="*60)
    
    train, test = train_test_split_timeseries(series, config.TRAIN_TEST_SPLIT)
    
    model = ARIMA(train, order=order)
    fitted = model.fit()
    
    # Forecast on test set
    forecast = fitted.forecast(steps=len(test))
    metrics = calculate_metrics(test.values, forecast.values)
    print_metrics("ARIMA", metrics)
    
    save_model(fitted, 'arima_model.pkl')
    
    return fitted, train, test, forecast, metrics


def train_xgboost_model(series):
    """Train XGBoost model."""
    print("\n" + "="*60)
    print("Training XGBoost Model...")
    print("="*60)
    
    # Feature engineering
    df_features = pd.DataFrame({
        'Date': series.index,
        'Demand': series.values
    })
    
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Month'] = df_features['Date'].dt.month
    df_features['MonthsSinceStart'] = np.arange(len(df_features))
    
    # Lag features
    for lag in [1, 3, 6, 12]:
        if lag < len(df_features):
            df_features[f'Lag_{lag}'] = df_features['Demand'].shift(lag)
    
    df_features = df_features.dropna()
    
    feature_cols = ['Year', 'Month', 'MonthsSinceStart', 'Lag_1', 'Lag_3', 'Lag_6', 'Lag_12']
    X = df_features[feature_cols]
    y = df_features['Demand']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.COMBINED_MODEL_TEST_SIZE, 
        random_state=config.RANDOM_STATE, shuffle=False
    )
    
    # Train model
    model = XGBRegressor(**config.XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test.values, y_pred)
    print_metrics("XGBoost", metrics)
    
    save_model(model, 'xgboost_model.pkl')
    
    return model, X, y, X_test, y_test, y_pred, metrics


def train_combined_model(series):
    """Train combined ARIMA + XGBoost model."""
    print("\n" + "="*60)
    print("Training Combined Model (ARIMA + XGBoost)...")
    print("="*60)
    
    # ARIMA component
    arima_model = ARIMA(series, order=config.ARIMA_ORDER)
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.predict(start=0, end=len(series)-1)
    residuals = series - arima_pred
    
    # XGBoost component (predict residuals)
    df_features = pd.DataFrame({
        'Date': series.index,
        'Residuals': residuals.values
    })
    
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Month'] = df_features['Date'].dt.month
    df_features['MonthsSinceStart'] = np.arange(len(df_features))
    
    feature_cols = ['Year', 'Month', 'MonthsSinceStart']
    X = df_features[feature_cols]
    y = df_features['Residuals']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.COMBINED_MODEL_TEST_SIZE,
        random_state=config.RANDOM_STATE, shuffle=False
    )
    
    xgb_model = XGBRegressor(**config.XGBOOST_PARAMS)
    xgb_model.fit(X_train, y_train)
    
    # Generate ARIMA forecast for test period
    test_start_idx = len(series) - len(X_test)
    arima_test_pred = arima_fit.forecast(steps=len(X_test))
    
    xgb_pred_residuals = xgb_model.predict(X_test)
    
    # Combine predictions
    combined_pred = arima_test_pred.values + xgb_pred_residuals
    y_actual = arima_pred.iloc[-len(X_test):].values + y_test.values
    
    metrics = calculate_metrics(y_actual, combined_pred)
    print_metrics("Combined Model", metrics)
    
    save_model(arima_fit, 'combined_arima_model.pkl')
    save_model(xgb_model, 'combined_xgboost_model.pkl')
    
    return arima_fit, xgb_model, metrics


def forecast_future(series, periods=config.FORECAST_PERIODS):
    """Generate future forecast using ARIMA."""
    print("\n" + "="*60)
    print(f"Forecasting {periods} periods into the future...")
    print("="*60)
    
    model = ARIMA(series, order=config.ARIMA_ORDER)
    fitted = model.fit()
    
    forecast = fitted.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()
    forecast_df['index'] = pd.date_range(
        start=series.index[-1] + pd.DateOffset(months=1),
        periods=periods,
        freq='MS'
    )
    
    print("\nFuture Demand Forecast:")
    print(forecast_df[['index', 'mean', 'mean_ci_lower', 'mean_ci_upper']])
    
    return forecast_df


def compare_models(arima_metrics, xgboost_metrics, combined_metrics):
    """Compare all models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'ARIMA': arima_metrics,
        'XGBoost': xgboost_metrics,
        'Combined': combined_metrics
    })
    
    print("\n", comparison_df)
    
    # Find best model
    best_model = comparison_df.loc['RMSE'].idxmin()
    print(f"\n✓ Best Model (lowest RMSE): {best_model}")
    
    return comparison_df


def plot_results(series, arima_fit, arima_train, arima_test, arima_forecast, xgb_pred, y_test_indices):
    """Plot model results."""
    fig, axes = plt.subplots(2, 2, figsize=config.PLOT_FIGSIZE)
    
    # Full series
    axes[0, 0].plot(series.index, series.values, label='Actual', linewidth=2, color='blue')
    axes[0, 0].set_title('Full Time Series', fontsize=12)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Demand')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # ARIMA - Full predictions
    arima_pred_full = arima_fit.predict(start=0, end=len(series)-1)
    axes[0, 1].plot(series.index, series.values, label='Actual', linewidth=2, color='blue')
    axes[0, 1].plot(series.index, arima_pred_full, label='ARIMA Prediction', linestyle='--', alpha=0.7, color='orange')
    axes[0, 1].axvline(x=arima_train.index[-1], color='red', linestyle=':', alpha=0.5, label='Train/Test Split')
    axes[0, 1].set_title('ARIMA Forecast on Full Data', fontsize=12)
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Demand')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # XGBoost
    test_start_idx = len(series) - len(y_test_indices)
    test_indices_dates = series.index[test_start_idx:]
    axes[1, 0].scatter(test_indices_dates, y_test_indices, label='Actual', s=50, color='blue')
    axes[1, 0].plot(test_indices_dates, xgb_pred, label='XGBoost Prediction', color='orange', linewidth=2)
    axes[1, 0].set_title('XGBoost Predictions vs Actual', fontsize=12)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Demand')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Residuals
    residuals = series.values - arima_pred_full.values
    axes[1, 1].plot(series.index, residuals, label='Residuals', color='red', linewidth=1.5)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('ARIMA Residuals', fontsize=12)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(config.PROJECT_ROOT, 'demand_forecast_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Results saved to '{output_file}'")
    plt.close()


def main():
    """Main execution flow."""
    print("="*60)
    print("PRODUCT DEMAND FORECASTING")
    print("="*60)
    
    # Create models directory
    Path(config.MODELS_DIR).mkdir(exist_ok=True)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = load_and_preprocess(config.DATA_PATH, config.DATE_COLUMN, config.DEMAND_COLUMN)
    
    # Aggregate demand
    print("Aggregating demand data...")
    series = aggregate_demand(df, config.DATE_COLUMN, config.DEMAND_COLUMN, config.AGGREGATION_FREQ)
    print(f"Data range: {series.index[0].date()} to {series.index[-1].date()}")
    print(f"Number of periods: {len(series)}")
    
    # Train models
    arima_fit, arima_train, arima_test, arima_forecast, arima_metrics = train_arima_model(series)
    
    xgb_model, X, y, X_test, y_test, xgb_pred, xgb_metrics = train_xgboost_model(series)
    
    arima_combined, xgb_residual, combined_metrics = train_combined_model(series)
    
    # Compare models
    compare_models(arima_metrics, xgb_metrics, combined_metrics)
    
    # Future forecast
    forecast_df = forecast_future(series, config.FORECAST_PERIODS)
    
    # Generate plots
    plot_results(series, arima_fit, arima_train, arima_test, arima_forecast, xgb_pred, y_test.values)
    
    # Generate dashboard
    try:
        from dashboard import generate_dashboard_html
        generate_dashboard_html()
    except ImportError:
        print("\n⚠ Dashboard requires plotly. Install with: pip install plotly kaleido")
    
    print("\n" + "="*60)
    print("✓ Forecasting Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
