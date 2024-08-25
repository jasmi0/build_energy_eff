import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def load_sample_data():
    """Generate sample building telemetry data for demonstration."""
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=30)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    data = {
        'datetime': date_range,
        'energy_kwh': [],
        'temperature': [],
        'humidity': [],
        'occupancy': []
    }
    
    for i, dt in enumerate(date_range):
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
        
        base_energy = 40
        
        if 6 <= hour <= 22:
            daily_factor = 1.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 16)
        else:
            daily_factor = 0.8
        
        # Weekly pattern (lower on weekends)
        weekly_factor = 0.7 if day_of_week >= 5 else 1.0
        
        # Seasonal pattern (higher in summer/winter)
        seasonal_factor = 1 + 0.4 * abs(np.sin(2 * np.pi * (month - 3) / 12))
        
        # Random variation
        noise = np.random.normal(0, 5)
        
        energy = max(10, base_energy * daily_factor * weekly_factor * seasonal_factor + noise)
        data['energy_kwh'].append(energy)
        
        # Temperature (outdoor temperature affecting building load)
        base_temp = 20 + 8 * np.sin(2 * np.pi * (month - 3) / 12)  # Seasonal variation
        daily_temp_var = 5 * np.sin(2 * np.pi * (hour - 6) / 24)   # Daily variation
        temp_noise = np.random.normal(0, 2)
        temperature = base_temp + daily_temp_var + temp_noise
        data['temperature'].append(temperature)
        
        # Humidity (inversely correlated with temperature)
        humidity = max(20, min(80, 60 - 0.5 * (temperature - 20) + np.random.normal(0, 5)))
        data['humidity'].append(humidity)
        
        # Occupancy (binary, higher during business hours)
        if day_of_week < 5 and 8 <= hour <= 18:  # Weekday business hours
            occupancy = 1 if np.random.random() > 0.2 else 0
        else:
            occupancy = 1 if np.random.random() > 0.8 else 0
        data['occupancy'].append(occupancy)
    
    return pd.DataFrame(data)

def export_schedule_to_csv(schedule, filename=None):
    """
    Export optimization schedule to CSV format.
    
    Args:
        schedule (pd.DataFrame): Optimization schedule
        filename (str): Output filename (optional)
        
    Returns:
        str: CSV data as string
    """
    if filename is None:
        filename = f"optimal_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    export_data = schedule.copy()
    export_data['timestamp'] = export_data.index.strftime('%Y-%m-%d %H:%M:%S')
    
    columns = ['timestamp', 'temperature_setpoint', 'predicted_energy', 'comfort_violation']
    export_data = export_data[columns]
    
    csv_data = export_data.to_csv(index=False)
    
    return csv_data

def export_schedule_to_ical(schedule):
    """
    Export optimization schedule to iCalendar format.
    
    Args:
        schedule (pd.DataFrame): Optimization schedule
        
    Returns:
        str: iCalendar data as string
    """
    ical_lines = []
    ical_lines.append("BEGIN:VCALENDAR")
    ical_lines.append("VERSION:2.0")
    ical_lines.append("PRODID:-//Building Energy Optimizer//EN")
    ical_lines.append("CALSCALE:GREGORIAN")
    ical_lines.append("METHOD:PUBLISH")
    
    for i, (timestamp, row) in enumerate(schedule.iterrows()):
        # Create event for each hour
        event_start = timestamp.strftime('%Y%m%dT%H%M%S')
        event_end = (timestamp + timedelta(hours=1)).strftime('%Y%m%dT%H%M%S')
        
        ical_lines.append("BEGIN:VEVENT")
        ical_lines.append(f"UID:setpoint-{i}@building-energy-optimizer")
        ical_lines.append(f"DTSTART:{event_start}")
        ical_lines.append(f"DTEND:{event_end}")
        ical_lines.append(f"SUMMARY:Temp Setpoint: {row['temperature_setpoint']:.1f}°C")
        ical_lines.append(f"DESCRIPTION:Energy: {row['predicted_energy']:.1f} kWh")
        ical_lines.append("END:VEVENT")
    
    ical_lines.append("END:VCALENDAR")
    
    return '\n'.join(ical_lines)

def calculate_energy_savings(baseline_schedule, optimized_schedule):
    """
    Calculate energy savings between baseline and optimized schedules.
    
    Args:
        baseline_schedule (pd.DataFrame): Baseline schedule
        optimized_schedule (pd.DataFrame): Optimized schedule
        
    Returns:
        dict: Savings metrics
    """
    baseline_energy = baseline_schedule['predicted_energy'].sum()
    optimized_energy = optimized_schedule['predicted_energy'].sum()
    
    absolute_savings = baseline_energy - optimized_energy
    percent_savings = (absolute_savings / baseline_energy) * 100 if baseline_energy > 0 else 0
    
    return {
        'baseline_energy_kwh': baseline_energy,
        'optimized_energy_kwh': optimized_energy,
        'absolute_savings_kwh': absolute_savings,
        'percent_savings': percent_savings
    }

def validate_data_quality(data):
    """
    Validate quality of input building data.
    
    Args:
        data (pd.DataFrame): Building telemetry data
        
    Returns:
        dict: Data quality report
    """
    report = {
        'total_records': len(data),
        'missing_values': {},
        'data_ranges': {},
        'quality_score': 0,
        'issues': []
    }
    
    for column in data.columns:
        missing_count = data[column].isnull().sum()
        missing_percent = (missing_count / len(data)) * 100
        report['missing_values'][column] = {
            'count': missing_count,
            'percent': missing_percent
        }
        
        if missing_percent > 20:
            report['issues'].append(f"High missing values in {column}: {missing_percent:.1f}%")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column in data.columns:
            col_data = data[column].dropna()
            if len(col_data) > 0:
                report['data_ranges'][column] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std': col_data.std()
                }
                
                if column == 'temperature' and (col_data.min() < -20 or col_data.max() > 50):
                    report['issues'].append(f"Unrealistic temperature values: {col_data.min():.1f} to {col_data.max():.1f}°C")
                
                if column == 'energy_kwh' and (col_data.min() < 0 or col_data.max() > 1000):
                    report['issues'].append(f"Unrealistic energy values: {col_data.min():.1f} to {col_data.max():.1f} kWh")
                
                if column == 'humidity' and (col_data.min() < 0 or col_data.max() > 100):
                    report['issues'].append(f"Invalid humidity values: {col_data.min():.1f} to {col_data.max():.1f}%")
    
    missing_score = max(0, 100 - sum(v['percent'] for v in report['missing_values'].values()) / len(report['missing_values']))
    issues_score = max(0, 100 - len(report['issues']) * 10)
    
    report['quality_score'] = (missing_score + issues_score) / 2
    
    return report

def create_comfort_bounds(temperature_data, method='adaptive'):
    """
    Create comfort bounds based on different standards.
    
    Args:
        temperature_data (pd.Series): Temperature data
        method (str): Method to use ('adaptive', 'ashrae', 'fixed')
        
    Returns:
        dict: Comfort bounds
    """
    if method == 'adaptive':
        mean_temp = temperature_data.mean()
        comfort_temp = 17.8 + 0.31 * mean_temp  # Simplified adaptive model
        
        bounds = {
            'lower': comfort_temp - 2.5,
            'upper': comfort_temp + 2.5,
            'optimal': comfort_temp
        }
    
    elif method == 'ashrae':
        bounds = {
            'lower': 20,
            'upper': 26,
            'optimal': 23
        }
    
    elif method == 'fixed':
        bounds = {
            'lower': 21,
            'upper': 25,
            'optimal': 22
        }
    
    return bounds

def generate_report_summary(optimization_results):
    """
    Generate a summary report of optimization results.
    
    Args:
        optimization_results (dict): Results from optimization
        
    Returns:
        str: Formatted report summary
    """
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("BUILDING ENERGY OPTIMIZATION REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    if 'savings' in optimization_results:
        savings = optimization_results['savings']
        report_lines.append("ENERGY SAVINGS SUMMARY:")
        report_lines.append(f"  Baseline Energy: {savings['baseline_energy_kwh']:.1f} kWh")
        report_lines.append(f"  Optimized Energy: {savings['optimized_energy_kwh']:.1f} kWh")
        report_lines.append(f"  Absolute Savings: {savings['absolute_savings_kwh']:.1f} kWh")
        report_lines.append(f"  Percent Savings: {savings['percent_savings']:.1f}%")
        report_lines.append("")
    
    if 'metrics' in optimization_results:
        metrics = optimization_results['metrics']
        report_lines.append("PERFORMANCE METRICS:")
        for key, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        report_lines.append("")
    
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("  1. Implement the optimized setpoint schedule")
    report_lines.append("  2. Monitor actual vs. predicted energy consumption")
    report_lines.append("  3. Adjust comfort parameters based on occupant feedback")
    report_lines.append("  4. Re-run optimization weekly with updated data")
    
    return '\n'.join(report_lines)

# Export utility class for better organization
class ExportUtility:
    """Utility class for exporting schedules and reports."""
    
    @staticmethod
    def to_csv(schedule, filename=None):
        """Export schedule to CSV."""
        return export_schedule_to_csv(schedule, filename)
    
    @staticmethod
    def to_ical(schedule):
        """Export schedule to iCalendar."""
        return export_schedule_to_ical(schedule)
    
    @staticmethod
    def to_json(schedule, metadata=None):
        """Export schedule to JSON format."""
        export_data = {
            'schedule': schedule.to_dict('records'),
            'metadata': metadata or {},
            'generated_at': datetime.now().isoformat()
        }
        return json.dumps(export_data, indent=2)

# Create a module-level instance for easy access
export_schedule = ExportUtility()
