"""Exploratory demand analysis and visualization."""

import pandas as pd
import matplotlib.pyplot as plt

import config
from utils import load_and_preprocess, aggregate_demand


def main():
    """Analyze and visualize demand trends."""
    # Load and preprocess data
    df = load_and_preprocess(config.DATA_PATH, config.DATE_COLUMN, config.DEMAND_COLUMN)
    
    # Filter data for the date range from 2012 to 2016 (if available)
    df = df.loc[(df[config.DATE_COLUMN] >= '2012-01-01') & (df[config.DATE_COLUMN] <= '2016-12-31')]
    
    # Group by Date to ensure uniqueness
    df = df.groupby(config.DATE_COLUMN, as_index=False).agg({config.DEMAND_COLUMN: 'sum'})
    
    # Reindex to ensure all days are included
    full_index = pd.date_range(
        start=df[config.DATE_COLUMN].min(),
        end=df[config.DATE_COLUMN].max(),
        freq='D'
    )
    df = df.set_index(config.DATE_COLUMN).reindex(full_index).reset_index()
    df.rename(columns={'index': config.DATE_COLUMN}, inplace=True)
    
    # Fill missing values through interpolation
    df[config.DEMAND_COLUMN] = df[config.DEMAND_COLUMN].interpolate()
    
    # Aggregate data by month with unique days
    monthly_aggregates = df.groupby(pd.Grouper(key=config.DATE_COLUMN, freq='ME')).agg(
        total_sales=(config.DEMAND_COLUMN, 'sum'),
        unique_days=(config.DATE_COLUMN, 'nunique')
    )
    
    # Calculate average daily demand
    monthly_aggregates['average_daily_demand'] = (
        monthly_aggregates['total_sales'] / monthly_aggregates['unique_days']
    )
    
    # Use average daily demand for plotting
    sales_trend = monthly_aggregates['average_daily_demand']
    
    # Calculate a rolling average
    sales_trend_rolling = sales_trend.rolling(window=3).mean()
    
    # Plot
    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.plot(sales_trend, label='Monthly Sales', color='blue', linewidth=1.5, alpha=0.7)
    plt.plot(sales_trend_rolling, label='3-Month Rolling Average', color='red', linestyle='--', linewidth=2)
    
    # Add titles and labels
    plt.title('Monthly Sales Trends Over Time', fontsize=16)
    plt.xlabel('Date (Months)', fontsize=14)
    plt.ylabel('Average Demand Per Day (Units)', fontsize=14)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)
    
    # Add grid and legend
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    
    # Display
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()