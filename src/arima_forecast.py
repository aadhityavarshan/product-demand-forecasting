"""ARIMA time series forecasting model."""

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import config
from utils import (
    load_and_preprocess,
    aggregate_demand,
    train_test_split_timeseries,
    calculate_metrics,
    save_model,
    print_metrics
)


def main():
    """Train and evaluate ARIMA model."""
    # Load and preprocess data
    df = load_and_preprocess(config.DATA_PATH, config.DATE_COLUMN, config.DEMAND_COLUMN)
    
    # Aggregate data monthly
    sales_trend = aggregate_demand(df, config.DATE_COLUMN, config.DEMAND_COLUMN, config.AGGREGATION_FREQ)
    
    # Train-test split
    train, test = train_test_split_timeseries(sales_trend, config.TRAIN_TEST_SPLIT)
    
    # Fit ARIMA model
    arima_model = ARIMA(train, order=config.ARIMA_ORDER)
    arima_fit = arima_model.fit()
    
    # Forecast
    forecast = arima_fit.forecast(steps=len(test))
    
    # Calculate metrics
    metrics = calculate_metrics(test.values, forecast.values)
    print_metrics("ARIMA", metrics)
    
    # Save model
    save_model(arima_fit, 'arima_model.pkl')
    
    # Plot actual vs forecast
    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.plot(test.index, test, label='Actual', linewidth=2)
    plt.plot(test.index, forecast, label='Forecast', color='orange', linewidth=2)
    plt.title('ARIMA Model Forecast', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Order Demand')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
