# Product Demand Forecasting

A comprehensive demand forecasting system that combines ARIMA and XGBoost machine learning models to predict product demand. This project includes individual model implementations, a hybrid approach, and a complete pipeline for model training, evaluation, and forecasting.

## Features

- **ARIMA Model**: Time series forecasting using AutoRegressive Integrated Moving Average
- **XGBoost Model**: Gradient boosting with engineered features for demand prediction
- **Combined Model**: Hybrid approach leveraging both ARIMA trend and XGBoost residual learning
- **Model Comparison**: Automated evaluation and comparison of all models
- **Interactive Dashboard**: Comprehensive visual dashboard comparing models side-by-side
- **Future Forecasting**: Generate predictions for future periods with confidence intervals
- **Model Persistence**: Save and load trained models for reuse
- **Visualization**: Comprehensive plots for result analysis

## Project Structure

```
product-demand-forecasting/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── dataset/
│   └── Historical Product Demand.csv  # Training data
├── models/                            # Trained model storage
└── src/
    ├── config.py                      # Configuration parameters
    ├── utils.py                       # Shared utility functions
    ├── main.py                        # Main orchestration script
    ├── dashboard.py                   # Plotly dashboard generator
    ├── generate_dashboard.py           # Standalone dashboard script
    ├── streamlit_app.py               # Interactive Streamlit app
    ├── arima_forecast.py              # ARIMA model implementation
    ├── xgboost_forecast.py            # XGBoost model implementation
    ├── combinedmodel.py               # Combined model implementation
    └── demandforecastingmodel.py      # Exploratory analysis
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/product-demand-forecasting.git
   cd product-demand-forecasting
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start - Run All Models

```bash
cd src
python main.py
```

This will:
- Load and preprocess the historical demand data
- Train ARIMA, XGBoost, and Combined models
- Compare model performance
- Generate future demand forecasts
- Create visualization plots
- Generate an interactive dashboard
- Save trained models to the `models/` directory

### Interactive Streamlit Dashboard

For an interactive web-based dashboard with multiple views and real-time interactions:

```bash
streamlit run src/streamlit_app.py
```

This launches a browser-based dashboard with five sections:
- **Overview**: Best model selection and key performance metrics
- **Model Comparison**: Detailed metrics table and side-by-side comparison
- **Forecasts**: Full series and test period forecast visualizations
- **Feature Analysis**: XGBoost feature importance rankings
- **Diagnostics**: ARIMA residuals analysis and statistics

### Generate Plotly Dashboard

To generate an interactive HTML dashboard (after models have been trained):

```bash
python src/generate_dashboard.py
```

This creates `demand_forecasting_dashboard.html` with:
- **Performance Metrics Comparison**: RMSE and MAE side-by-side
- **R² Score Comparison**: Model accuracy visualization
- **Forecast Comparison**: Overlay of actual vs predicted values
- **Feature Importance**: XGBoost feature importance chart
- **Residuals Analysis**: ARIMA residual patterns over time
- **Detailed Metrics Table**: Complete numerical comparison

### Run Individual Models

**ARIMA Forecast:**
```bash
python src/arima_forecast.py
```

**XGBoost Forecast:**
```bash
python src/xgboost_forecast.py
```

**Combined Model:**
```bash
python src/combinedmodel.py
```

**Exploratory Analysis:**
```bash
python src/demandforecastingmodel.py
```

## Configuration

Edit `src/config.py` to customize:
- Data path and column names
- Train-test split ratio
- ARIMA parameters (p, d, q)
- XGBoost hyperparameters
- Forecast horizon (number of periods)
- Plotting settings

## Model Performance

All models are evaluated using:
- **RMSE** (Root Mean Squared Error) - Penalizes larger errors more
- **MAE** (Mean Absolute Error) - Average absolute prediction error
- **R²** (Coefficient of Determination) - Variance explained by the model

The system compares all models and identifies the best performer.

## Data Format

The input CSV file should contain:
- `Date`: Date column (will be converted to datetime)
- `Order_Demand`: Demand quantity (numeric)

Additional columns are ignored. Missing values are automatically handled through interpolation or removal.

## Output

- **demand_forecast_results.png**: Static visualization of all model results (4-panel plot)
- **demand_forecasting_dashboard.html**: Interactive Plotly HTML dashboard for detailed analysis
- **Streamlit Web App**: Interactive web interface (run with `streamlit run src/streamlit_app.py`)
- **models/** directory: Saved trained models (ARIMA, XGBoost, Combined)

## Future Improvements

- [ ] Hyperparameter tuning (grid/random search)
- [ ] Ensemble methods (weighted combination of predictions)
- [ ] Seasonality detection and adjustment
- [ ] External regressor support (holidays, promotions, etc.)
- [ ] Web API for real-time predictions
- [ ] Unit tests and integration tests
- [ ] Automated model retraining pipeline
- [ ] Forecast accuracy tracking over time

## Requirements

See `requirements.txt` for all dependencies. Main packages:
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: ML utilities and metrics
- statsmodels: ARIMA implementation
- xgboost: XGBoost algorithms
- matplotlib: Visualization
- joblib: Model serialization

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
