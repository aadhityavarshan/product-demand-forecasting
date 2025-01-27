import pandas as pd
import matplotlib.pyplot as plt
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

# Train-test split
train_size = int(len(sales_trend) * 0.8)
train, test = sales_trend[:train_size], sales_trend[train_size:]

# Fit ARIMA model
arima_model = ARIMA(train, order=(2, 1, 2))  # Adjust p, d, q based on grid search or AIC/BIC analysis
arima_fit = arima_model.fit()

# Forecast
forecast = arima_fit.forecast(steps=len(test))

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(test, forecast))
print(f"ARIMA RMSE: {rmse}")

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='orange')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Order Demand')
plt.legend()
plt.show()
