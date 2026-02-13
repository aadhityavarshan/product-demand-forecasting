"""
Interactive dashboard for comparing demand forecasting models.
Generates an HTML dashboard with comprehensive model comparisons.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import config
from utils import load_and_preprocess, aggregate_demand, calculate_metrics, load_model


def create_metrics_comparison_chart(metrics_dict):
    """Create a comparison chart for model metrics."""
    df_metrics = pd.DataFrame(metrics_dict).T
    df_metrics = df_metrics.reset_index()
    df_metrics.columns = ['Model', 'RMSE', 'MAE', 'R²']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_metrics['Model'],
        y=df_metrics['RMSE'],
        name='RMSE',
        marker_color='#EF553B'
    ))
    
    fig.add_trace(go.Bar(
        x=df_metrics['Model'],
        y=df_metrics['MAE'],
        name='MAE',
        marker_color='#00CC96'
    ))
    
    fig.update_layout(
        title='Model Performance Metrics Comparison',
        xaxis_title='Model',
        yaxis_title='Error Value',
        barmode='group',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_r2_comparison(metrics_dict):
    """Create R² score comparison."""
    models = list(metrics_dict.keys())
    r2_scores = [metrics_dict[m]['R2'] for m in models]
    
    colors = ['#00CC96' if r2 > 0 else '#EF553B' for r2 in r2_scores]
    
    fig = go.Figure(data=[go.Bar(
        x=models,
        y=r2_scores,
        marker_color=colors,
        text=[f'{r2:.4f}' for r2 in r2_scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>R² Score: %{y:.4f}<extra></extra>'
    )])
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Baseline (R²=0)")
    
    fig.update_layout(
        title='R² Score Comparison (Higher is Better)',
        xaxis_title='Model',
        yaxis_title='R² Score',
        height=450,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_forecast_comparison(series, arima_fit, xgb_model, X_all):
    """Create forecast comparison across all models."""
    # Prepare predictions
    arima_pred = arima_fit.predict(start=0, end=len(series)-1)
    
    feature_cols = ['Year', 'Month', 'MonthsSinceStart', 'Lag_1', 'Lag_3', 'Lag_6', 'Lag_12']
    xgb_full_pred = xgb_model.predict(X_all)
    
    # Get test period
    test_size = int(len(series) * 0.2)
    test_start = len(series) - test_size
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Full Series Comparison', 'Test Period Detail'),
        specs=[[{'secondary_y': False}], [{'secondary_y': False}]]
    )
    
    # Full series
    fig.add_trace(
        go.Scatter(x=series.index, y=series.values, name='Actual', 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=series.index, y=arima_pred, name='ARIMA',
                  line=dict(color='orange', width=1.5, dash='dash')),
        row=1, col=1
    )
    
    # Test period detail
    test_indices = series.index[test_start:]
    fig.add_trace(
        go.Scatter(x=test_indices, y=series.values[test_start:], name='Actual (Test)',
                  mode='lines+markers', line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_indices, y=arima_pred.iloc[test_start:], name='ARIMA (Test)',
                  line=dict(color='orange', width=1.5, dash='dash')),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Order Demand", row=1, col=1)
    fig.update_yaxes(title_text="Order Demand", row=2, col=1)
    
    fig.update_layout(
        title_text='Forecast Comparison Across Models',
        height=700,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_xgboost_feature_importance(xgb_model, feature_names):
    """Create feature importance chart for XGBoost."""
    importance = xgb_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    fig = go.Figure(data=[go.Bar(
        x=importance[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker_color='#636EFA',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    )])
    
    fig.update_layout(
        title='XGBoost Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_residuals_analysis(series, arima_fit):
    """Create residuals analysis chart."""
    arima_pred = arima_fit.predict(start=0, end=len(series)-1)
    residuals = series.values - arima_pred.values
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals Over Time', 'Residuals Distribution'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Residuals over time
    fig.add_trace(
        go.Scatter(x=series.index, y=residuals, name='Residuals',
                  mode='lines', line=dict(color='red', width=1.5)),
        row=1, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=residuals, name='Distribution', nbinsx=30,
                    marker_color='lightblue', showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Residual Value", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_layout(
        title_text='ARIMA Residuals Analysis',
        height=500,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def create_metrics_table(metrics_dict):
    """Create a detailed metrics table."""
    data = []
    for model_name, metrics in metrics_dict.items():
        data.append([
            model_name,
            f"{metrics['RMSE']:,.2f}",
            f"{metrics['MAE']:,.2f}",
            f"{metrics['R2']:.4f}"
        ])
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Model</b>', '<b>RMSE</b>', '<b>MAE</b>', '<b>R² Score</b>'],
            fill_color='#636EFA',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[[row[0] for row in data],
                   [row[1] for row in data],
                   [row[2] for row in data],
                   [row[3] for row in data]],
            fill_color='lavender',
            align='center',
            font=dict(size=11),
            height=30
        )
    )])
    
    fig.update_layout(
        title='Model Performance Metrics Table',
        height=350,
        template='plotly_white'
    )
    
    return fig


def create_dashboard(metrics_dict, series, arima_fit, xgb_model, X_all, feature_names):
    """Create the complete interactive dashboard."""
    
    # Create individual charts
    metrics_chart = create_metrics_comparison_chart(metrics_dict)
    r2_chart = create_r2_comparison(metrics_dict)
    forecast_chart = create_forecast_comparison(series, arima_fit, xgb_model, X_all)
    importance_chart = create_xgboost_feature_importance(xgb_model, feature_names)
    residuals_chart = create_residuals_analysis(series, arima_fit)
    table_chart = create_metrics_table(metrics_dict)
    
    # Create main dashboard
    dashboard = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Performance Metrics (RMSE & MAE)', 'R² Score Comparison',
            'Full Series Forecast', 'XGBoost Feature Importance',
            'ARIMA Residuals', 'Metrics Summary',
            'Model Rankings', 'Data Statistics'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'scatter', 'colspan': 2}, None],
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'table', 'colspan': 2}, None]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Add traces from individual charts
    for trace in metrics_chart.data:
        dashboard.add_trace(trace, row=1, col=1)
    
    for trace in r2_chart.data:
        dashboard.add_trace(trace, row=1, col=2)
    
    # Get test period for forecast chart
    test_size = int(len(series) * 0.2)
    test_start = len(series) - test_size
    arima_pred = arima_fit.predict(start=0, end=len(series)-1)
    
    dashboard.add_trace(
        go.Scatter(x=series.index, y=series.values, name='Actual',
                  line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    dashboard.add_trace(
        go.Scatter(x=series.index, y=arima_pred, name='ARIMA',
                  line=dict(color='orange', dash='dash')),
        row=2, col=1
    )
    
    # Add feature importance
    importance = xgb_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    dashboard.add_trace(
        go.Bar(x=importance[indices],
              y=[feature_names[i] for i in indices],
              orientation='h',
              name='Importance',
              marker_color='#636EFA'),
        row=3, col=2
    )
    
    # Add residuals
    residuals = series.values - arima_pred.values
    dashboard.add_trace(
        go.Scatter(x=series.index, y=residuals, name='Residuals',
                  mode='lines', line=dict(color='red')),
        row=3, col=1
    )
    dashboard.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # Add table
    data = []
    for model_name, metrics in metrics_dict.items():
        data.append([
            model_name,
            f"{metrics['RMSE']:,.0f}",
            f"{metrics['MAE']:,.0f}",
            f"{metrics['R2']:.4f}"
        ])
    
    dashboard.add_trace(
        go.Table(
            header=dict(values=['<b>Model</b>', '<b>RMSE</b>', '<b>MAE</b>', '<b>R²</b>'],
                       fill_color='#636EFA',
                       align='center',
                       font=dict(color='white')),
            cells=dict(values=[[row[0] for row in data],
                             [row[1] for row in data],
                             [row[2] for row in data],
                             [row[3] for row in data]],
                      fill_color='lavender',
                      align='center')),
        row=4, col=1
    )
    
    # Update layout
    dashboard.update_xaxes(title_text="Model", row=1, col=1)
    dashboard.update_xaxes(title_text="Model", row=1, col=2)
    dashboard.update_xaxes(title_text="Feature", row=3, col=2)
    dashboard.update_xaxes(title_text="Date", row=2, col=1)
    dashboard.update_xaxes(title_text="Date", row=3, col=1)
    
    dashboard.update_yaxes(title_text="Error", row=1, col=1)
    dashboard.update_yaxes(title_text="Score", row=1, col=2)
    dashboard.update_yaxes(title_text="Demand", row=2, col=1)
    dashboard.update_yaxes(title_text="Importance", row=3, col=2)
    dashboard.update_yaxes(title_text="Residual", row=3, col=1)
    
    dashboard.update_layout(
        title_text='<b>Product Demand Forecasting - Model Comparison Dashboard</b>',
        showlegend=True,
        height=1400,
        width=1600,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return dashboard


def generate_dashboard_html():
    """Generate the complete dashboard HTML file."""
    print("\n" + "="*60)
    print("Generating Interactive Dashboard...")
    print("="*60)
    
    # Load data
    df = load_and_preprocess(config.DATA_PATH, config.DATE_COLUMN, config.DEMAND_COLUMN)
    series = aggregate_demand(df, config.DATE_COLUMN, config.DEMAND_COLUMN, config.AGGREGATION_FREQ)
    
    # Train models to get necessary data
    print("Loading models and training data...")
    
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
    
    # Calculate metrics
    arima_metrics = calculate_metrics(series.values, arima_pred.values)
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = calculate_metrics(y_test.values, xgb_pred)
    
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
    combined_metrics = calculate_metrics(y_combined_actual, combined_final_pred)
    
    metrics_dict = {
        'ARIMA': arima_metrics,
        'XGBoost': xgb_metrics,
        'Combined': combined_metrics
    }
    
    # Create dashboard
    print("Creating dashboard visualizations...")
    dashboard = create_dashboard(metrics_dict, series, arima_fit, xgb_model, X, feature_cols)
    
    # Save dashboard
    output_path = os.path.join(config.PROJECT_ROOT, 'demand_forecasting_dashboard.html')
    dashboard.write_html(output_path)
    
    print(f"✓ Interactive Dashboard saved to: {output_path}")
    print(f"✓ Open in browser to view detailed model comparisons")
    
    return dashboard


if __name__ == "__main__":
    generate_dashboard_html()
