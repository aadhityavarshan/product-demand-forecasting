import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import math


# Load and preprocess data
df = pd.read_csv("dataset/Historical Product Demand.csv")
df.dropna(subset=['Date', 'Order_Demand'], inplace=True) # Removes rows where either Data or Order_Demand is missing
df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Converts Date to Datetime Format
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce') # Converts Order_Demand to Numeric
df.dropna(subset=['Order_Demand'], inplace=True) # Removes remaining NaN Values
df = df.set_index('Date')
df = df['Order_Demand'].resample('M').sum()  # Aggregate monthly sales

# Fit ARIMA to capture trends and seasonality
arima_model = ARIMA(df, order=(2, 1, 2))  # Adjust parameters based on AIC/BIC optimization
arima_fit = arima_model.fit()

# ARIMA predictions
arima_pred = arima_fit.predict(start=df.index[0], end=df.index[-1])
residuals = df - arima_pred

# Feature engineering for XGBoost
df_features = pd.DataFrame({'Date': df.index})
df_features['Year'] = df_features['Date'].dt.year
df_features['Month'] = df_features['Date'].dt.month
df_features['Residuals'] = residuals.values

# Prepare features and target
X = df_features[['Year', 'Month']]
y = df_features['Residuals']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# XGBoost predictions for residuals
xgb_pred_residuals = xgb_model.predict(X_test)

# Final predictions

# Ensure ARIMA forecast matches the length of X_test
arima_test_pred = arima_fit.forecast(steps=len(X_test))

# Convert predictions to NumPy arrays
arima_test_pred = np.array(arima_test_pred)
xgb_pred_residuals = np.array(xgb_pred_residuals)

# Ensure shapes match before combining
if arima_test_pred.shape == xgb_pred_residuals.shape:
    final_pred = arima_test_pred + xgb_pred_residuals
else:
    print("Error: Shapes do not match. Check forecast length.")
    print(f"ARIMA Predictions Shape: {arima_test_pred.shape}")
    print(f"XGBoost Residuals Shape: {xgb_pred_residuals.shape}")


# Evaluate the combined model
rmse = math.sqrt(mean_squared_error(y_test + arima_test_pred, final_pred))
mae = mean_absolute_error(y_test + arima_test_pred, final_pred)
print(f"Combined Model RMSE: {rmse:.2f}")
print(f"Combined Model MAE: {mae:.2f}")

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(df.index, df, label='Actual')
plt.plot(df.index, arima_pred, label='ARIMA Prediction', linestyle='--', alpha=0.7)
plt.scatter(X_test.index, final_pred, color='orange', label='Combined Prediction', alpha=0.8)
plt.legend()
plt.title('Combined ARIMA + XGBoost Predictions')
plt.xlabel('Date')
plt.ylabel('Order Demand')
plt.grid(alpha=0.3)
plt.show()

