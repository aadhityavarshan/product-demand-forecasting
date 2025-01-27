import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
df = pd.read_csv("dataset/Historical Product Demand.csv")
df.dropna(subset=['Date', 'Order_Demand'], inplace=True) # Removes rows where either Data or Order_Demand is missing
df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Converts Date to Datetime Format
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce') # Converts Order_Demand to Numeric
df.dropna(subset=['Order_Demand'], inplace=True) # Removes remaining NaN Values

# Feature engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Aggregate data
df_monthly = df.groupby(['Year', 'Month'])['Order_Demand'].sum().reset_index()

# Prepare features and target
X = df_monthly[['Year', 'Month']]
y = df_monthly['Order_Demand']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
xgb_pred = xgb_model.predict(X_test)
xgb_rmse = mean_absolute_error(y_test, xgb_pred)
print(f"XGBoost RMSE: {xgb_rmse}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.scatter(y_test, xgb_pred, alpha=0.6, edgecolors='w')
plt.title('XGBoost Model - Actual vs Predicted')
plt.xlabel('Actual Order Demand')
plt.ylabel('Predicted Order Demand')
plt.grid(alpha=0.3)
plt.show()
