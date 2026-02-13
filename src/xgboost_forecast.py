"""XGBoost gradient boosting forecasting model."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    """Train and evaluate XGBoost model."""
    # Load and preprocess data
    df = load_and_preprocess(config.DATA_PATH, config.DATE_COLUMN, config.DEMAND_COLUMN)
    
    # Aggregate data monthly
    sales_trend = aggregate_demand(df, config.DATE_COLUMN, config.DEMAND_COLUMN, config.AGGREGATION_FREQ)
    
    # Feature engineering
    df_features = pd.DataFrame({
        'Date': sales_trend.index,
        'Demand': sales_trend.values
    })
    
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Month'] = df_features['Date'].dt.month
    df_features['MonthsSinceStart'] = np.arange(len(df_features))
    
    # Lag features
    for lag in [1, 3, 6, 12]:
        if lag < len(df_features):
            df_features[f'Lag_{lag}'] = df_features['Demand'].shift(lag)
    
    df_features = df_features.dropna()
    
    # Prepare features and target
    feature_cols = ['Year', 'Month', 'MonthsSinceStart', 'Lag_1', 'Lag_3', 'Lag_6', 'Lag_12']
    X = df_features[feature_cols]
    y = df_features['Demand']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, shuffle=False
    )
    
    # Train XGBoost model
    xgb_model = XGBRegressor(**config.XGBOOST_PARAMS)
    xgb_model.fit(X_train, y_train)
    
    # Predict and evaluate
    xgb_pred = xgb_model.predict(X_test)
    metrics = calculate_metrics(y_test.values, xgb_pred)
    print_metrics("XGBoost", metrics)
    
    # Save model
    save_model(xgb_model, 'xgboost_model.pkl')
    
    # Plot actual vs predicted
    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.scatter(y_test, xgb_pred, alpha=0.6, edgecolors='w', s=100)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('XGBoost Model - Actual vs Predicted', fontsize=14)
    plt.xlabel('Actual Order Demand')
    plt.ylabel('Predicted Order Demand')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
