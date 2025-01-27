import pandas as pd
import matplotlib.pyplot as plt


# Load and preprocess data
df = pd.read_csv("dataset/Historical Product Demand.csv")
df.dropna(subset=['Date', 'Order_Demand'], inplace=True) # Removes rows where either Data or Order_Demand is missing
df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Converts Date to Datetime Format
df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce') # Converts Order_Demand to Numeric
df.dropna(subset=['Order_Demand'], inplace=True) # Removes remaining NaN Values

# Filter data for the date range from 2012 to 2016
df = df.loc[(df['Date'] >= '2012-01-01') & (df['Date'] <= '2016-12-31')]

# Group by Date to ensure uniqueness
df = df.groupby('Date', as_index=False).agg({'Order_Demand': 'sum'})  # Aggregates Order_Demand by summing for each date

# Reindex to ensure all days are included
full_index = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')  # Create a daily date range
df = df.set_index('Date').reindex(full_index).reset_index()  # Reindex and reset index
df.rename(columns={'index': 'Date'}, inplace=True)  # Rename the index column back to 'Date'

# Fill missing values (e.g., using interpolation)
df['Order_Demand'] = df['Order_Demand'].interpolate()  # Interpolates missing values

'''
# Aggregate data by month with counts
monthly_aggregates = df.groupby(pd.Grouper(key='Date', freq='M')).agg(
    total_sales=('Order_Demand', 'sum'),
    data_points=('Order_Demand', 'count')
)

# Calculate weighted sales (example: dividing by count to normalize)
monthly_aggregates['weighted_sales'] = monthly_aggregates['total_sales'] / monthly_aggregates['data_points']

# Use weighted sales for plotting
sales_trend = monthly_aggregates['weighted_sales']
'''



# Aggregate data by month with unique days
monthly_aggregates = df.groupby(pd.Grouper(key='Date', freq='ME')).agg(
    total_sales=('Order_Demand', 'sum'),     # Total sales for the month
    unique_days=('Date', 'nunique')          # Number of unique days in the month
)

# Calculate average daily demand
monthly_aggregates['average_daily_demand'] = monthly_aggregates['total_sales'] / monthly_aggregates['unique_days']

# Use average daily demand for plotting
sales_trend = monthly_aggregates['average_daily_demand']

# Calculate a rolling average
sales_trend_rolling = sales_trend.rolling(window=3).mean()

# Plot
plt.figure(figsize=(14, 8))
plt.plot(sales_trend, label='Monthly Sales', color='blue', linewidth=1.5, alpha=0.7)
plt.plot(sales_trend_rolling, label='3-Month Rolling Average', color='red', linestyle='--', linewidth=2)

# Add titles and labels
plt.title('Monthly Sales Trends Over Time', fontsize=16)
plt.xlabel('Date (Months)', fontsize=14)  # X-axis label
plt.ylabel('Average Demand Per Day (Units)', fontsize=14)  # Y-axis label

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Add a grid
plt.grid(alpha=0.3)

# Add a legend
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()