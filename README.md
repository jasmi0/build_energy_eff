# Building Energy Efficiency Optimization

A comprehensive machine learning system for optimizing building energy consumption while maintaining occupant comfort through intelligent HVAC control and scheduling.

## Project Overview

This project combines time-series forecasting with optimization algorithms to reduce building energy consumption by 10-30% while ensuring occupant comfort remains within ASHRAE standards. The system provides:

- **Energy Demand Forecasting**: Uses SARIMA and XGBoost models to predict energy consumption
- **Setpoint Optimization**: Finds optimal HVAC schedules using constrained optimization
- **Interactive Dashboard**: Streamlit-based interface for monitoring and control
- **Export Capabilities**: Generate schedules in CSV and iCalendar formats

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/alerman/projects/build_energy_eff
   ```
### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## System Workflow

1. **Data Upload**: Upload CSV files with building telemetry data or use sample data
2. **Data Processing**: Automatic cleaning, feature engineering, and validation
3. **Model Training**: Train forecasting models (SARIMA, XGBoost, or Ensemble)
4. **Optimization**: Find optimal setpoint schedules with comfort constraints
5. **Export Results**: Download schedules and performance reports

## Project Structure

```
build_energy_eff/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Core modules
│   ├── __init__.py
│   ├── data_processor.py  # Data preprocessing and feature engineering
│   ├── forecaster.py      # Energy demand forecasting models
│   ├── optimizer.py       # Setpoint optimization algorithms
│   └── utils.py          # Utility functions and export tools
├── data/                  # Data storage (CSV files)
├── models/               # Saved model files
├── notebooks/            # Jupyter notebooks for analysis
└── tests/               # Unit tests
```

## Data Format

The system expects CSV files with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `datetime` | Timestamp | 2023-01-01 00:00:00 |
| `energy_kwh` | Energy consumption (kWh) | 45.2 |
| `temperature` | Zone temperature (°C) | 22.5 |
| `humidity` | Relative humidity (%) | 45.0 |
| `occupancy` | Occupancy status (0/1) | 1 |

**Optional columns**: `zone_temps`, `weather_temp`, `solar_irradiance`

## Configuration Options

### Forecasting Models
- **SARIMA**: Statistical time-series model for seasonal patterns
- **XGBoost**: Machine learning model with weather and occupancy features  
- **Ensemble**: Combination of multiple models for improved accuracy

### Optimization Parameters
- **Temperature Range**: Min/max setpoint bounds (18-28°C)
- **Comfort Weight**: Penalty for comfort violations (0.1-2.0)
- **Energy Price**: Cost per kWh for economic optimization ($0.05-0.30)
- **Constraint Type**: ASHRAE Standard, Custom Range, or Adaptive Comfort

## Performance Metrics

The system provides comprehensive performance metrics:

- **Forecasting Accuracy**: MAPE, RMSE, R² scores
- **Energy Savings**: Absolute and percentage reduction
- **Comfort Compliance**: Violation frequency and severity
- **Economic Impact**: Cost savings and payback analysis

## Advanced Usage

### Custom Data Processing
```python
from src.data_processor import DataProcessor

processor = DataProcessor()
processed_data = processor.process_data(raw_data)
features = processor.get_feature_columns(processed_data)
```

### Model Training
```python
from src.forecaster import EnergyForecaster

forecaster = EnergyForecaster(model_type='XGBoost')
forecaster.train(processed_data)
forecast = forecaster.forecast(horizon=24)
```

### Optimization
```python
from src.optimizer import SetpointOptimizer

optimizer = SetpointOptimizer(
    temp_range=(20, 26),
    comfort_weight=0.5,
    energy_price=0.12
)
schedule = optimizer.optimize(forecast, horizon=24)
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Example Results

Typical optimization results show:
- **10-30% energy savings** compared to baseline schedules
- **<5% comfort violations** when properly configured
- **2-5 minute processing time** for 24-hour optimization horizons
- **<10% MAPE** for next-day energy forecasting

## Future Enhancements

- **Real-time Integration**: Connect to building management systems (BMS)
- **Multi-zone Control**: Optimize multiple thermal zones simultaneously
- **Predictive Maintenance**: Detect HVAC equipment anomalies
- **Occupant Feedback**: Adaptive comfort learning from user preferences
- **Carbon Optimization**: Minimize CO₂ emissions alongside energy costs

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Email: contact@energy-optimization.com
- Documentation: [Wiki Pages](../../wiki)

## Acknowledgments

- ASHRAE for comfort standards and guidelines
- Open-source communities for excellent ML and optimization libraries
- Building industry professionals for domain expertise and validation

---

**Built for a more energy-efficient future**
