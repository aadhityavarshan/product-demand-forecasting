"""Combined ARIMA + XGBoost hybrid forecasting model."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import config
from utils import (
    load_and_preprocess,
    aggregate_demand,
    calculate_metrics,
    save_model,
    print_metrics
)


def main():
    """Train and evaluate combined ARIMA + XGBoost model."""
    # Load and preprocess data
    df = load_and_preprocess(config.DATA_PATH, config.DATE_COLUMN, config.DEMAND_COLUMN)
    
    # Aggregate data monthly
    series = aggregate_demand(df, config.DATE_COLUMN, config.DEMAND_COLUMN, config.AGGREGATION_FREQ)
    
    # Fit ARIMA to capture trends and seasonality
    arima_model = ARIMA(series, order=config.ARIMA_ORDER)
    arima_fit = arima_model.fit()
    
    # ARIMA predictions on full data
    arima_pred_full = arima_fit.predict(start=0, end=len(series)-1)
    residuals = series - arima_pred_full
    
    # Feature engineering for XGBoost (to model residuals)
    df_features = pd.DataFrame({'Date': series.index})
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Month'] = df_features['Date'].dt.month
    df_features['MonthsSinceStart'] = np.arange(len(df_features))
    df_features['Residuals'] = residuals.values
    
    # Prepare features and target
    feature_cols = ['Year', 'Month', 'MonthsSinceStart']
    X = df_features[feature_cols]
    y = df_features['Residuals']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.COMBINED_MODEL_TEST_SIZE,
        random_state=config.RANDOM_STATE, shuffle=False
    )
    
    # Train XGBoost to predict residuals
    xgb_model = XGBRegressor(**config.XGBOOST_PARAMS)
    xgb_model.fit(X_train, y_train)
    
    # XGBoost predictions for residuals on test set
    xgb_pred_residuals = xgb_model.predict(X_test)
    
    # ARIMA forecast for test period
    arima_test_pred = arima_fit.forecast(steps=len(X_test))
    
    # Combine: ARIMA trend + XGBoost residuals
    final_pred = arima_test_pred.values + xgb_pred_residuals
    
    # Get actual values for test period
    test_indices = X_test.index
    y_actual_test = series.iloc[len(series) - len(X_test):].values
    
    # Evaluate the combined model
    metrics = calculate_metrics(y_actual_test, final_pred)
    print_metrics("Combined Model (ARIMA + XGBoost)", metrics)
    
    # Save models
    save_model(arima_fit, 'combined_arima_model.pkl')
    save_model(xgb_model, 'combined_xgboost_model.pkl')
    
    # Plot the results
    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.plot(series.index, series, label='Actual', linewidth=2)
    plt.plot(series.index, arima_pred_full, label='ARIMA Prediction', linestyle='--', alpha=0.7)
    
    # Plot combined predictions on test set
    test_start_idx = len(series) - len(X_test)
    test_indices_plot = range(test_start_idx, len(series))
    plt.scatter(range(test_start_idx, len(series)), final_pred, color='orange', 
               label='Combined Prediction', alpha=0.8, s=50)
    
    plt.legend(fontsize=11)
    plt.title('Combined ARIMA + XGBoost Predictions', fontsize=14)
    plt.xlabel('Time Index')
    plt.ylabel('Order Demand')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

