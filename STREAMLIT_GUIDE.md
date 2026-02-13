# Streamlit Dashboard Quick Start Guide

This guide explains how to use the interactive Streamlit dashboard for product demand forecasting.

## Installation

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

### Option 1: Using the launcher script (Recommended)
```bash
python src/run_dashboard.py
```

### Option 2: Direct Streamlit command
```bash
streamlit run src/streamlit_app.py
```

### Option 3: From the src directory
```bash
cd src
streamlit run streamlit_app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## Dashboard Sections

The dashboard has 5 main views accessible from the left sidebar:

### 1️⃣ Overview
- **Best Model Badge**: Shows which model has the lowest RMSE
- **Key Metrics**: Displays the top performer's RMSE, MAE, and R² score
- **Comparison Charts**: Side-by-side RMSE/MAE and R² score visualizations

### 2️⃣ Model Comparison
- **Metrics Table**: Detailed table showing all models' performance metrics
- **Individual Cards**: Separate sections for ARIMA, XGBoost, and Combined models
- **Quick Metrics**: RMSE, MAE, and R² scores for each model

### 3️⃣ Forecasts
- **Full Series Chart**: Shows actual vs ARIMA predictions across the entire dataset
- **Test Period Detail**: Zoomed view of the test period with all model predictions
- **Test Statistics**: Average demand values for actual and predicted data

### 4️⃣ Feature Analysis
- **Feature Importance**: Bar chart showing XGBoost's top 10 important features
- **Feature Descriptions**: Explanation of each feature used in the model
- **Feature Types**:
  - Temporal: Year, Month, MonthsSinceStart
  - Lag Features: Lag_1, Lag_3, Lag_6, Lag_12 (seasonal patterns)

### 5️⃣ Diagnostics
- **Residuals Over Time**: Line chart showing ARIMA forecast errors
- **Residuals Distribution**: Histogram showing error frequency distribution
- **Residual Statistics**: Mean, Standard Deviation, Min, and Max values

## Features

- ✅ **Real-time Model Comparison**: Compare all three models side-by-side
- ✅ **Interactive Charts**: Zoom, pan, and hover for detailed information
- ✅ **Responsive Design**: Works on desktop and tablet browsers
- ✅ **Fast Loading**: Models are cached for quick interactions
- ✅ **Professional Interface**: Clean, modern Streamlit design

## Tips

1. **Sidebar Navigation**: Use the sidebar on the left to switch between views
2. **Data Info**: Check the sidebar for quick data statistics
3. **Hover Data**: Hover over charts to see exact values
4. **Responsive Charts**: Charts adjust to your browser size
5. **Stop Server**: Press `Ctrl+C` in the terminal to stop the dashboard

## Troubleshooting

### Dashboard won't start
Make sure Streamlit is installed:
```bash
pip install streamlit
```

### Port already in use
Run on a different port:
```bash
streamlit run src/streamlit_app.py --server.port 8502
```

### Charts not displaying
Try clearing your browser cache or using an incognito window

### Slow performance
Restart the dashboard - it caches models in memory. Use:
```bash
streamlit run src/streamlit_app.py --logger.level=warning
```

## Advanced Usage

To customize the dashboard, edit `src/streamlit_app.py`:
- Modify colors by changing hex codes in Plotly charts
- Add new metrics or visualizations
- Change the sidebar layout
- Customize data preprocessing

For more Streamlit documentation, visit: https://docs.streamlit.io/

## First Time Setup

If this is your first time running the dashboard:

1. Ensure you have run `python src/main.py` at least once to train models
2. Verify `dataset/Historical Product Demand.csv` exists
3. Check that `models/` directory was created
4. Run the dashboard with: `streamlit run src/streamlit_app.py`

The dashboard will take a moment to load the first time as it trains models. Subsequent runs will be faster due to caching.

---

**Need help?** Check the main README.md for more project information.
