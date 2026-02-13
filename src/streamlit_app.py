"""
Interactive Streamlit Dashboard for Product Demand Forecasting Models
Run with: streamlit run src/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import load_and_preprocess, aggregate_demand, calculate_metrics


# Page configuration
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .best-model {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_data_and_train_models():
    """Load data and train all models."""
    # Load and preprocess
    df = load_and_preprocess(config.DATA_PATH, config.DATE_COLUMN, config.DEMAND_COLUMN)
    series = aggregate_demand(df, config.DATE_COLUMN, config.DEMAND_COLUMN, config.AGGREGATION_FREQ)
    
    # ARIMA
    arima_model = ARIMA(series, order=config.ARIMA_ORDER)
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.predict(start=0, end=len(series)-1)
    
    # XGBoost
    df_features = pd.DataFrame({
        'Date': series.index,
        'Demand': series.values
    })
    
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Month'] = df_features['Date'].dt.month
    df_features['MonthsSinceStart'] = np.arange(len(df_features))
    
    for lag in [1, 3, 6, 12]:
        if lag < len(df_features):
            df_features[f'Lag_{lag}'] = df_features['Demand'].shift(lag)
    
    df_features = df_features.dropna()
    
    feature_cols = ['Year', 'Month', 'MonthsSinceStart', 'Lag_1', 'Lag_3', 'Lag_6', 'Lag_12']
    X = df_features[feature_cols]
    y = df_features['Demand']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, shuffle=False
    )
    
    xgb_model = XGBRegressor(**config.XGBOOST_PARAMS)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    # Combined model
    residuals = series - arima_pred
    df_features_combined = pd.DataFrame({'Date': series.index})
    df_features_combined['Year'] = df_features_combined['Date'].dt.year
    df_features_combined['Month'] = df_features_combined['Date'].dt.month
    df_features_combined['MonthsSinceStart'] = np.arange(len(df_features_combined))
    df_features_combined['Residuals'] = residuals.values
    
    X_combined = df_features_combined[['Year', 'Month', 'MonthsSinceStart']]
    y_combined = df_features_combined['Residuals']
    
    X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=config.RANDOM_STATE, shuffle=False
    )
    
    xgb_combined = XGBRegressor(**config.XGBOOST_PARAMS)
    xgb_combined.fit(X_combined_train, y_combined_train)
    
    arima_test_pred = arima_fit.forecast(steps=len(X_combined_test))
    xgb_combined_pred = xgb_combined.predict(X_combined_test)
    combined_final_pred = arima_test_pred.values + xgb_combined_pred
    
    y_combined_actual = series.iloc[-len(X_combined_test):].values
    
    # Calculate metrics
    arima_metrics = calculate_metrics(series.values, arima_pred.values)
    xgb_metrics = calculate_metrics(y_test.values, xgb_pred)
    combined_metrics = calculate_metrics(y_combined_actual, combined_final_pred)
    
    return {
        'series': series,
        'arima_fit': arima_fit,
        'arima_pred': arima_pred,
        'arima_metrics': arima_metrics,
        'xgb_model': xgb_model,
        'X_test': X_test,
        'y_test': y_test,
        'xgb_pred': xgb_pred,
        'xgb_metrics': xgb_metrics,
        'combined_metrics': combined_metrics,
        'feature_cols': feature_cols
    }


def plot_metrics_comparison(metrics_dict):
    """Create metrics comparison chart."""
    models = list(metrics_dict.keys())
    rmse_values = [metrics_dict[m]['RMSE'] for m in models]
    mae_values = [metrics_dict[m]['MAE'] for m in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE Comparison', 'MAE Comparison'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='#EF553B',
               text=[f'{x:,.0f}' for x in rmse_values], textposition='auto'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=mae_values, name='MAE', marker_color='#00CC96',
               text=[f'{x:,.0f}' for x in mae_values], textposition='auto'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, template='plotly_white')
    return fig


def plot_r2_scores(metrics_dict):
    """Create R² score comparison."""
    models = list(metrics_dict.keys())
    r2_scores = [metrics_dict[m]['R2'] for m in models]
    colors = ['#00CC96' if r2 > 0 else '#EF553B' for r2 in r2_scores]
    
    fig = go.Figure(data=[go.Bar(
        x=models,
        y=r2_scores,
        marker_color=colors,
        text=[f'{r2:.4f}' for r2 in r2_scores],
        textposition='auto'
    )])
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title='R² Score Comparison (Higher is Better)',
        xaxis_title='Model',
        yaxis_title='R² Score',
        height=400,
        template='plotly_white'
    )
    return fig


def plot_forecast_comparison(series, arima_pred, X_test, xgb_pred):
    """Create forecast comparison visualization."""
    test_start = len(series) - len(X_test)
    test_indices = series.index[test_start:]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Full Series Forecast', 'Test Period Detail'),
        vertical_spacing=0.12
    )
    
    # Full series
    fig.add_trace(
        go.Scatter(x=series.index, y=series.values, name='Actual',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=series.index, y=arima_pred, name='ARIMA',
                  line=dict(color='orange', dash='dash')),
        row=1, col=1
    )
    
    # Test period detail
    fig.add_trace(
        go.Scatter(x=test_indices, y=series.values[test_start:], name='Actual (Test)',
                  mode='lines+markers', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_indices, y=arima_pred.iloc[test_start:], name='ARIMA (Test)',
                  line=dict(color='orange', dash='dash')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_indices, y=xgb_pred, name='XGBoost (Test)',
                  mode='markers', marker=dict(color='green', size=8)),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Demand", row=1, col=1)
    fig.update_yaxes(title_text="Demand", row=2, col=1)
    
    fig.update_layout(height=600, template='plotly_white', hovermode='x unified')
    return fig


def plot_feature_importance(xgb_model, feature_cols):
    """Create feature importance chart."""
    importance = xgb_model.feature_importances_
    indices = np.argsort(importance)[::-1][:10]  # Top 10
    
    fig = go.Figure(data=[go.Bar(
        y=[feature_cols[i] for i in indices],
        x=importance[indices],
        orientation='h',
        marker_color='#636EFA'
    )])
    
    fig.update_layout(
        title='Top 10 XGBoost Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400,
        template='plotly_white'
    )
    return fig


def plot_residuals(series, arima_pred):
    """Create residuals analysis chart."""
    residuals = series.values - arima_pred.values
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals Over Time', 'Distribution'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=series.index, y=residuals, mode='lines',
                  line=dict(color='red'), name='Residuals'),
        row=1, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=25, marker_color='lightblue', 
                    name='Distribution', showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Residual", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_layout(height=400, template='plotly_white')
    return fig


def main():
    # Title
    st.markdown("# 📊 Product Demand Forecasting Dashboard")
    st.markdown("Compare ARIMA, XGBoost, and Combined forecasting models")
    
    # Load data and train models
    with st.spinner('Loading data and training models...'):
        data = load_data_and_train_models()
    
    series = data['series']
    arima_fit = data['arima_fit']
    arima_pred = data['arima_pred']
    arima_metrics = data['arima_metrics']
    xgb_model = data['xgb_model']
    X_test = data['X_test']
    y_test = data['y_test']
    xgb_pred = data['xgb_pred']
    xgb_metrics = data['xgb_metrics']
    combined_metrics = data['combined_metrics']
    feature_cols = data['feature_cols']
    
    metrics_dict = {
        'ARIMA': arima_metrics,
        'XGBoost': xgb_metrics,
        'Combined': combined_metrics
    }
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Navigation")
    page = st.sidebar.radio(
        "Select View:",
        ["Overview", "Model Comparison", "Forecasts", "Feature Analysis", "Diagnostics"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Data Information")
    st.sidebar.metric("Data Points", len(series))
    st.sidebar.metric("Date Range", f"{series.index[0].date()} to {series.index[-1].date()}")
    st.sidebar.metric("Train/Test Split", "80/20")
    
    # Page: Overview
    if page == "Overview":
        st.markdown("## Model Performance Overview")
        
        col1, col2, col3 = st.columns(3)
        
        # Best model
        best_model = min(metrics_dict, key=lambda x: metrics_dict[x]['RMSE'])
        with col1:
            st.markdown(f"### 🏆 Best Model: **{best_model}**")
            st.metric("RMSE", f"{metrics_dict[best_model]['RMSE']:,.0f}")
        
        with col2:
            st.metric("MAE", f"{metrics_dict[best_model]['MAE']:,.0f}")
        
        with col3:
            st.metric("R² Score", f"{metrics_dict[best_model]['R2']:.4f}")
        
        st.markdown("---")
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_metrics_comparison(metrics_dict), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_r2_scores(metrics_dict), use_container_width=True)
    
    # Page: Model Comparison
    elif page == "Model Comparison":
        st.markdown("## Detailed Model Comparison")
        
        # Metrics table
        st.markdown("### Performance Metrics Table")
        df_metrics = pd.DataFrame(metrics_dict).T
        df_metrics = df_metrics.round(4)
        st.dataframe(df_metrics, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### ARIMA")
            st.metric("RMSE", f"{arima_metrics['RMSE']:,.0f}")
            st.metric("MAE", f"{arima_metrics['MAE']:,.0f}")
            st.metric("R²", f"{arima_metrics['R2']:.4f}")
        
        with col2:
            st.markdown("### XGBoost")
            st.metric("RMSE", f"{xgb_metrics['RMSE']:,.0f}")
            st.metric("MAE", f"{xgb_metrics['MAE']:,.0f}")
            st.metric("R²", f"{xgb_metrics['R2']:.4f}")
        
        with col3:
            st.markdown("### Combined")
            st.metric("RMSE", f"{combined_metrics['RMSE']:,.0f}")
            st.metric("MAE", f"{combined_metrics['MAE']:,.0f}")
            st.metric("R²", f"{combined_metrics['R2']:.4f}")
    
    # Page: Forecasts
    elif page == "Forecasts":
        st.markdown("## Forecast Comparison")
        
        st.plotly_chart(
            plot_forecast_comparison(series, arima_pred, X_test, xgb_pred),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Forecast statistics
        st.markdown("### Test Period Statistics")
        test_start = len(series) - len(X_test)
        test_actual = series.values[test_start:]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Actual Demand", f"{test_actual.mean():,.0f}")
        
        with col2:
            st.metric("Average ARIMA Forecast", f"{arima_pred.iloc[test_start:].mean():,.0f}")
        
        with col3:
            st.metric("Average XGBoost Forecast", f"{xgb_pred.mean():,.0f}")
    
    # Page: Feature Analysis
    elif page == "Feature Analysis":
        st.markdown("## XGBoost Feature Analysis")
        
        st.plotly_chart(
            plot_feature_importance(xgb_model, feature_cols),
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### Feature Descriptions")
        feature_descriptions = {
            'Year': 'Year of the observation',
            'Month': 'Month of the year (1-12)',
            'MonthsSinceStart': 'Number of months since the beginning of the dataset',
            'Lag_1': 'Demand value from 1 month ago',
            'Lag_3': 'Demand value from 3 months ago',
            'Lag_6': 'Demand value from 6 months ago',
            'Lag_12': 'Demand value from 12 months ago (seasonal lag)'
        }
        
        for feature, description in feature_descriptions.items():
            st.write(f"**{feature}**: {description}")
    
    # Page: Diagnostics
    elif page == "Diagnostics":
        st.markdown("## Model Diagnostics")
        
        st.markdown("### ARIMA Residuals Analysis")
        st.plotly_chart(
            plot_residuals(series, arima_pred),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Residuals statistics
        residuals = series.values - arima_pred.values
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{residuals.mean():,.0f}")
        
        with col2:
            st.metric("Std Dev", f"{residuals.std():,.0f}")
        
        with col3:
            st.metric("Min", f"{residuals.min():,.0f}")
        
        with col4:
            st.metric("Max", f"{residuals.max():,.0f}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Product Demand Forecasting Dashboard | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
