
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_processor import DataProcessor
from src.forecaster import EnergyForecaster
from src.optimizer import SetpointOptimizer
from src.utils import load_sample_data, export_schedule

def main():
    st.set_page_config(
        page_title="Building Energy Efficiency Optimizer",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Building Energy Efficiency Optimization")
    st.markdown("### Optimize building energy consumption while maintaining comfort")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Data Upload & Processing", "Forecasting", "Optimization", "Results & Export"]
    )
    
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'optimization_done' not in st.session_state:
        st.session_state.optimization_done = False
    
    if page == "Data Upload & Processing":
        data_upload_page()
    elif page == "Forecasting":
        forecasting_page()
    elif page == "Optimization":
        optimization_page()
    elif page == "Results & Export":
        results_page()

def data_upload_page():
    st.header("Data Upload & Processing")
    
    data_option = st.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload CSV File"]
    )
    
    if data_option == "Use Sample Data":
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                data = load_sample_data()
                st.session_state.raw_data = data
                st.success("Sample data loaded successfully!")
    
    else:
        uploaded_file = st.file_uploader(
            "Upload building telemetry data (CSV)",
            type=['csv'],
            help="CSV should contain: datetime, zone_temps, humidity, occupancy, kWh"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.raw_data = data
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    if 'raw_data' in st.session_state:
        st.subheader("Data Overview")
        data = st.session_state.raw_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(data))
            st.metric("Date Range", f"{len(data)} records")
        
        with col2:
            st.metric("Columns", len(data.columns))
            st.metric("Missing Values", data.isnull().sum().sum())
        
        st.subheader("Data Sample")
        st.dataframe(data.head(10))
        
        # Data preprocessing
        st.subheader("Data Preprocessing")
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                processor = DataProcessor()
                processed_data = processor.process_data(data)
                st.session_state.processed_data = processed_data
                st.session_state.data_processed = True
                st.success("Data processed successfully!")
                
                st.subheader("Processing Results")
                st.write("Missing values handled")
                st.write("Features engineered")
                st.write("Data resampled and aligned")
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                axes[0,0].plot(processed_data.index, processed_data['energy_kwh'])
                axes[0,0].set_title('Energy Consumption Over Time')
                axes[0,0].set_ylabel('kWh')
                
                if 'temperature' in processed_data.columns:
                    axes[0,1].hist(processed_data['temperature'].dropna(), bins=30)
                    axes[0,1].set_title('Temperature Distribution')
                    axes[0,1].set_xlabel('Temperature (°C)')
                
                if 'occupancy' in processed_data.columns:
                    axes[1,0].plot(processed_data.index, processed_data['occupancy'])
                    axes[1,0].set_title('Occupancy Pattern')
                    axes[1,0].set_ylabel('Occupancy')
                
                if 'temperature' in processed_data.columns:
                    axes[1,1].scatter(processed_data['temperature'], processed_data['energy_kwh'], alpha=0.5)
                    axes[1,1].set_title('Energy vs Temperature')
                    axes[1,1].set_xlabel('Temperature (°C)')
                    axes[1,1].set_ylabel('Energy (kWh)')
                
                plt.tight_layout()
                st.pyplot(fig)

def forecasting_page():
    st.header("Energy Demand Forecasting")
    
    if not st.session_state.get('data_processed', False):
        st.warning("Please process data first in the Data Upload & Processing page.")
        return
    
    data = st.session_state.processed_data
    
    st.subheader("Model Configuration")
    model_type = st.selectbox(
        "Select forecasting model:",
        ["SARIMA", "XGBoost", "Ensemble"]
    )
    
    forecast_horizon = st.slider(
        "Forecast horizon (hours):",
        min_value=1,
        max_value=168,
        value=24
    )
    
    if st.button("Train Forecasting Model"):
        with st.spinner("Training model..."):
            forecaster = EnergyForecaster(model_type=model_type)
            forecaster.train(data)
            
            forecast = forecaster.forecast(horizon=forecast_horizon)
            
            st.session_state.forecaster = forecaster
            st.session_state.forecast = forecast
            st.session_state.model_trained = True
            
            st.success(f"{model_type} model trained successfully!")
            
            st.subheader("Model Performance")
            metrics = forecaster.get_metrics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAPE (%)", f"{metrics.get('mape', 0):.2f}")
            with col2:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
            with col3:
                st.metric("R²", f"{metrics.get('r2', 0):.3f}")
            
            st.subheader("Forecast Visualization")
            
            fig = go.Figure()
            
            recent_data = data.tail(forecast_horizon * 2)
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['energy_kwh'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast
            forecast_index = pd.date_range(
                start=data.index[-1] + pd.Timedelta(hours=1),
                periods=forecast_horizon,
                freq='H'
            )
            
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=forecast['prediction'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence intervals
            if 'upper_bound' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=forecast['upper_bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='red', width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=forecast['lower_bound'],
                    mode='lines',
                    name='Confidence Interval',
                    line=dict(color='red', width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
            
            fig.update_layout(
                title='Energy Demand Forecast',
                xaxis_title='Time',
                yaxis_title='Energy (kWh)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def optimization_page():
    st.header("Setpoint Optimization")
    
    if not st.session_state.get('model_trained', False):
        st.warning("Please train a forecasting model first.")
        return
    
    # Optimization parameters
    st.subheader("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temp_min = st.slider("Minimum Temperature (°C)", 18, 25, 21)
        temp_max = st.slider("Maximum Temperature (°C)", 22, 28, 25)
        comfort_weight = st.slider("Comfort Weight (λ)", 0.1, 2.0, 0.5, 0.1)
    
    with col2:
        optimization_horizon = st.slider("Optimization Horizon (hours)", 1, 48, 24)
        energy_price = st.number_input("Energy Price ($/kWh)", 0.05, 0.30, 0.12, 0.01)
        
    # Comfort constraints
    st.subheader("Comfort Constraints")
    constraint_type = st.selectbox(
        "Constraint Type:",
        ["ASHRAE Standard", "Custom Range", "Adaptive Comfort"]
    )
    
    # Run optimization
    if st.button("Optimize Setpoints"):
        with st.spinner("Running optimization..."):
            optimizer = SetpointOptimizer(
                temp_range=(temp_min, temp_max),
                comfort_weight=comfort_weight,
                energy_price=energy_price
            )
            
            forecast = st.session_state.forecast
            optimal_schedule = optimizer.optimize(
                forecast=forecast,
                horizon=optimization_horizon,
                constraint_type=constraint_type
            )
            
            st.session_state.optimal_schedule = optimal_schedule
            st.session_state.optimization_done = True
            
            st.success("Optimization completed!")
            
            # Results summary
            st.subheader("Optimization Results")
            
            baseline_energy = forecast['prediction'].sum()
            optimized_energy = optimal_schedule['predicted_energy'].sum()
            energy_savings = baseline_energy - optimized_energy
            savings_percent = (energy_savings / baseline_energy) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Baseline Energy (kWh)", f"{baseline_energy:.1f}")
            with col2:
                st.metric("Optimized Energy (kWh)", f"{optimized_energy:.1f}")
            with col3:
                st.metric("Energy Savings (kWh)", f"{energy_savings:.1f}")
            with col4:
                st.metric("Savings Percentage", f"{savings_percent:.1f}%")
            
            # Visualization
            st.subheader("Optimization Visualization")
            
            fig = go.Figure()
            
            # Baseline vs optimized energy
            time_index = pd.date_range(
                start=datetime.now(),
                periods=len(optimal_schedule),
                freq='H'
            )
            
            fig.add_trace(go.Scatter(
                x=time_index,
                y=forecast['prediction'][:len(optimal_schedule)],
                mode='lines',
                name='Baseline Energy',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=time_index,
                y=optimal_schedule['predicted_energy'],
                mode='lines',
                name='Optimized Energy',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title='Energy Consumption: Baseline vs Optimized',
                xaxis_title='Time',
                yaxis_title='Energy (kWh)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Setpoint schedule
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=time_index,
                y=optimal_schedule['temperature_setpoint'],
                mode='lines+markers',
                name='Temperature Setpoint',
                line=dict(color='blue')
            ))
            
            # Comfort bounds
            fig2.add_hline(y=temp_min, line_dash="dash", line_color="orange", 
                          annotation_text="Min Comfort")
            fig2.add_hline(y=temp_max, line_dash="dash", line_color="orange", 
                          annotation_text="Max Comfort")
            
            fig2.update_layout(
                title='Optimal Temperature Setpoint Schedule',
                xaxis_title='Time',
                yaxis_title='Temperature (°C)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig2, use_container_width=True)

def results_page():
    st.header("Results & Export")
    
    if not st.session_state.get('optimization_done', False):
        st.warning("Please complete optimization first.")
        return
    
    optimal_schedule = st.session_state.optimal_schedule
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    total_energy = optimal_schedule['predicted_energy'].sum()
    avg_setpoint = optimal_schedule['temperature_setpoint'].mean()
    max_setpoint = optimal_schedule['temperature_setpoint'].max()
    min_setpoint = optimal_schedule['temperature_setpoint'].min()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Energy (kWh)", f"{total_energy:.1f}")
    with col2:
        st.metric("Avg Setpoint (°C)", f"{avg_setpoint:.1f}")
    with col3:
        st.metric("Max Setpoint (°C)", f"{max_setpoint:.1f}")
    with col4:
        st.metric("Min Setpoint (°C)", f"{min_setpoint:.1f}")
    
    # Detailed schedule table
    st.subheader("Detailed Schedule")
    st.dataframe(optimal_schedule)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export CSV"):
            csv_data = optimal_schedule.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"optimal_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export iCal"):
            ical_data = export_schedule.to_ical(optimal_schedule)
            st.download_button(
                label="Download iCal",
                data=ical_data,
                file_name=f"optimal_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.ics",
                mime="text/calendar"
            )
    
    with col3:
        if st.button("Export Report"):
            # Generate PDF report (placeholder)
            st.info("PDF report export feature coming soon!")
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    if 'forecaster' in st.session_state:
        metrics = st.session_state.forecaster.get_metrics()
        
        metrics_df = pd.DataFrame({
            'Metric': ['MAPE (%)', 'RMSE', 'MAE', 'R²'],
            'Value': [
                f"{metrics.get('mape', 0):.2f}",
                f"{metrics.get('rmse', 0):.2f}",
                f"{metrics.get('mae', 0):.2f}",
                f"{metrics.get('r2', 0):.3f}"
            ]
        })
        
        st.table(metrics_df)

if __name__ == "__main__":
    main()
