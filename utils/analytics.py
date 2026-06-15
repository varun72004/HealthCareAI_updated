"""
Analytics Dashboard Utilities
Provides visualization and analysis functions
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def calculate_bmi(weight_kg, height_m):
    """Calculate BMI"""
    if weight_kg > 0 and height_m > 0:
        return weight_kg / (height_m ** 2)
    return None

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi is None:
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def create_bmi_chart(bmi_history):
    """Create BMI trend chart"""
    if not bmi_history or len(bmi_history) < 2:
        return None
    
    df = pd.DataFrame(bmi_history)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['bmi'],
        mode='lines+markers',
        name='BMI',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=10)
    ))
    
    # Add category lines
    fig.add_hline(y=18.5, line_dash="dash", line_color="orange", annotation_text="Underweight")
    fig.add_hline(y=25, line_dash="dash", line_color="green", annotation_text="Normal")
    fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Obese")
    
    fig.update_layout(
        title={'text': "BMI Trend Over Time", 'font': {'size': 18}},
        xaxis_title="Date",
        yaxis_title="BMI",
        template="plotly_white",
        height=400,
        font=dict(size=12),
        margin=dict(l=60, r=30, t=60, b=60)
    )
    
    return fig

def create_temperature_chart(temperature_history):
    """Create temperature trend chart"""
    if not temperature_history or len(temperature_history) < 2:
        return None
    
    df = pd.DataFrame(temperature_history)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['temperature'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10)
    ))
    
    # Add normal range
    fig.add_hline(y=37.0, line_dash="dash", line_color="green", annotation_text="Normal (37°C)")
    fig.add_hline(y=38.0, line_dash="dash", line_color="orange", annotation_text="Fever (38°C)")
    
    fig.update_layout(
        title={'text': "Body Temperature Trend", 'font': {'size': 18}},
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=400,
        font=dict(size=12),
        margin=dict(l=60, r=30, t=60, b=60)
    )
    
    return fig

def create_symptom_frequency_chart(symptom_data):
    """Create symptom frequency chart"""
    if not symptom_data:
        return None
    
    df = pd.DataFrame(symptom_data)
    symptom_counts = df['symptom'].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Bar(
            x=symptom_counts.values,
            y=symptom_counts.index,
            orientation='h',
            marker=dict(color='#4ECDC4')
        )
    ])
    
    fig.update_layout(
        title={'text': "Top 10 Most Frequent Symptoms", 'font': {'size': 18}},
        xaxis_title="Frequency",
        yaxis_title="Symptom",
        template="plotly_white",
        height=400,
        font=dict(size=12),
        margin=dict(l=80, r=30, t=60, b=60)
    )
    
    return fig

def create_disease_risk_chart(disease_risks):
    """Create disease risk comparison chart"""
    if not disease_risks:
        return None
    
    df = pd.DataFrame(disease_risks)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['disease'],
            y=df['risk_score'],
            marker=dict(
                color=df['risk_score'],
                colorscale='RdYlGn_r',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title={'text': "Disease Risk Scores", 'font': {'size': 18}},
        xaxis_title="Disease",
        yaxis_title="Risk Score",
        template="plotly_white",
        height=400,
        xaxis_tickangle=-45,
        font=dict(size=12),
        margin=dict(l=60, r=30, t=60, b=80)
    )
    
    return fig

def create_health_metrics_dashboard(bmi, temperature, symptoms_count, risk_score):
    """Create comprehensive health metrics dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('BMI', 'Temperature', 'Symptoms', 'Risk'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # BMI Gauge
    bmi_category = get_bmi_category(bmi)
    bmi_color = 'green' if bmi_category == 'Normal' else 'orange' if bmi_category in ['Underweight', 'Overweight'] else 'red'
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=bmi if bmi else 0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"BMI\n{bmi_category}"},
            gauge={
                'axis': {'range': [None, 40]},
                'bar': {'color': bmi_color},
                'steps': [
                    {'range': [0, 18.5], 'color': "lightgray"},
                    {'range': [18.5, 25], 'color': "gray"},
                    {'range': [25, 30], 'color': "lightgray"},
                    {'range': [30, 40], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ),
        row=1, col=1
    )
    
    # Temperature Gauge
    temp_color = 'green' if temperature and 36.1 <= temperature <= 37.2 else 'orange' if temperature and 37.3 <= temperature <= 38.0 else 'red'
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=temperature if temperature else 0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Temp (°C)"},
            gauge={
                'axis': {'range': [None, 42]},
                'bar': {'color': temp_color},
                'steps': [
                    {'range': [0, 36], 'color': "lightgray"},
                    {'range': [36, 37.2], 'color': "gray"},
                    {'range': [37.3, 38], 'color': "lightgray"},
                    {'range': [38, 42], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 38
                }
            }
        ),
        row=1, col=2
    )
    
    # Symptom Count
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=symptoms_count,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Active Symptoms"}
        ),
        row=2, col=1
    )
    
    # Risk Score
    risk_color = 'green' if risk_score < 0.4 else 'orange' if risk_score < 0.7 else 'red'
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=risk_score * 100 if risk_score else 0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title={'text': "Health Metrics Overview", 'font': {'size': 18}},
        height=700,
        template="plotly_white",
        margin=dict(l=60, r=30, t=80, b=60),
        font=dict(size=12)
    )
    
    return fig

def create_trend_analysis(data_history):
    """Create trend analysis over time"""
    if not data_history or len(data_history) < 2:
        return None
    
    df = pd.DataFrame(data_history)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('BMI Trend', 'Temperature Trend'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    if 'bmi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['bmi'],
                mode='lines+markers',
                name='BMI',
                line=dict(color='#2E86AB')
            ),
            row=1, col=1
        )
    
    if 'temperature' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#FF6B6B')
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="BMI", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    
    fig.update_layout(
        title={'text': "Health Trends Over Time", 'font': {'size': 18}},
        height=700,
        template="plotly_white",
        margin=dict(l=60, r=30, t=80, b=60),
        font=dict(size=12)
    )
    
    return fig

