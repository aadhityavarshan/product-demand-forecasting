"""Configuration settings for demand forecasting models."""

import os

# Data settings - resolve path relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "Historical Product Demand.csv")
DATE_COLUMN = 'Date'
DEMAND_COLUMN = 'Order_Demand'
AGGREGATION_FREQ = 'ME'  # Monthly end aggregation (changed from 'M' for pandas 2.0+)

# Train-test split
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# ARIMA parameters
ARIMA_ORDER = (2, 1, 2)  # (p, d, q)

# XGBoost parameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
}

# Combined model parameters
COMBINED_MODEL_TEST_SIZE = 0.2

# Forecasting horizon (number of periods to forecast into the future)
FORECAST_PERIODS = 12  # 12 months ahead

# Models directory
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Plotting settings
PLOT_FIGSIZE = (14, 8)
