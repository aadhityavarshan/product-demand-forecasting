import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math


# Load and preprocess data
df = pd.read_csv("dataset/Historical Product Demand.csv")
df.dropna(subset=['Date', 'Order_Demand'], inplace=True) # Removes rows where either Data or Order_Demand is missing
df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Converts Date to Datetime Format
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce') # Converts Order_Demand to Numeric
df.dropna(subset=['Order_Demand'], inplace=True) # Removes remaining NaN Values

# Aggregate data monthly
sales_trend = df.groupby(pd.Grouper(key='Date', freq='M'))['Order_Demand'].sum() # Sums up the Order_Demand for each month to calculate total montly sales

# Calculate a rolling average
sales_trend_rolling = sales_trend.rolling(window=3).mean()

# Plot
plt.figure(figsize=(14, 8))
plt.plot(sales_trend, label='Monthly Sales', color='blue', linewidth=1.5, alpha=0.7)
plt.plot(sales_trend_rolling, label='3-Month Rolling Average', color='red', linestyle='--', linewidth=2)

# Add titles and labels
plt.title('Monthly Sales Trends Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Order Demand', fontsize=14)

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Add a grid
plt.grid(alpha=0.3)

# Add a legend
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()