
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools
from datetime import datetime, timedelta

class SetpointOptimizer:    
    def __init__(self, temp_range=(20, 26), comfort_weight=0.5, energy_price=0.12):
        self.temp_min, self.temp_max = temp_range
        self.comfort_weight = comfort_weight
        self.energy_price = energy_price
        self.comfort_constraints = {}
        
    def optimize(self, forecast, horizon=24, constraint_type='ASHRAE Standard'):
        self._setup_constraints(constraint_type)
        
        optimal_schedule = self._grid_search_optimization(forecast, horizon)
        
        return optimal_schedule
    
    def _setup_constraints(self, constraint_type):
        if constraint_type == 'ASHRAE Standard':
            self.comfort_constraints = {
                'temp_min': 20,
                'temp_max': 26,
                'humidity_min': 30,
                'humidity_max': 70,
                'air_speed_max': 0.8
            }
        elif constraint_type == 'Custom Range':
            self.comfort_constraints = {
                'temp_min': self.temp_min,
                'temp_max': self.temp_max,
                'humidity_min': 30,
                'humidity_max': 70
            }
        elif constraint_type == 'Adaptive Comfort':
            self.comfort_constraints = {
                'temp_min': 18,
                'temp_max': 28,
                'adaptive': True
            }
    
    def _grid_search_optimization(self, forecast, horizon):
        temp_candidates = np.arange(self.temp_min, self.temp_max + 0.5, 0.5)
        
        periods_per_day = 3
        total_periods = max(1, horizon // (24 // periods_per_day))
        
        best_cost = float('inf')
        best_schedule = None
        
        setpoint_combinations = list(itertools.product(temp_candidates, repeat=total_periods))
        
        if len(setpoint_combinations) > 1000:
            import random
            setpoint_combinations = random.sample(setpoint_combinations, 1000)
        
        for setpoints in setpoint_combinations:
            # Create schedule from setpoints
            schedule = self._create_schedule_from_setpoints(setpoints, horizon)
            
            # Calculate cost
            cost = self._calculate_total_cost(schedule, forecast)
            
            if cost < best_cost:
                best_cost = cost
                best_schedule = schedule
        
        return best_schedule
    
    def _create_schedule_from_setpoints(self, setpoints, horizon):
        time_index = pd.date_range(
            start=datetime.now(),
            periods=horizon,
            freq='H'
        )
        periods_per_day = 3
        hours_per_period = 24 // periods_per_day
        schedule_data = []
        for hour in range(horizon):
            day_hour = hour % 24
            period_idx = min(day_hour // hours_per_period, len(setpoints) - 1)
            period_idx = period_idx % len(setpoints)
            temp_setpoint = setpoints[period_idx]
            predicted_energy = self._predict_energy_for_setpoint(
                temp_setpoint, hour, time_index[hour]
            )
            schedule_data.append({
                'hour': hour,
                'temperature_setpoint': temp_setpoint,
                'predicted_energy': predicted_energy,
                'comfort_violation': self._calculate_comfort_violation(temp_setpoint),
                'period': period_idx
            })
        schedule_df = pd.DataFrame(schedule_data)
        schedule_df.index = time_index
        return schedule_df
    
    def _predict_energy_for_setpoint(self, temp_setpoint, hour, timestamp):
        hour_of_day = hour % 24
        base_load = 40 + 30 * (0.5 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24))
        temp_deviation = abs(temp_setpoint - 22)
        temp_penalty = temp_deviation * 2
        month = timestamp.month
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (month - 3) / 12)
        is_weekend = timestamp.weekday() >= 5
        weekend_factor = 0.7 if is_weekend else 1.0
        total_energy = (base_load + temp_penalty) * seasonal_factor * weekend_factor
        noise = np.random.normal(0, 2)
        return max(0, total_energy + noise)
    
    def _calculate_comfort_violation(self, temp_setpoint):
        comfort_min = self.comfort_constraints.get('temp_min', 20)
        comfort_max = self.comfort_constraints.get('temp_max', 26)
        if temp_setpoint < comfort_min:
            return comfort_min - temp_setpoint
        elif temp_setpoint > comfort_max:
            return temp_setpoint - comfort_max
        else:
            return 0
    
    def _calculate_total_cost(self, schedule, forecast):
        energy_cost = schedule['predicted_energy'].sum() * self.energy_price
        comfort_penalty = schedule['comfort_violation'].sum() * self.comfort_weight * 10
        setpoint_changes = schedule['temperature_setpoint'].diff().abs().sum()
        change_penalty = setpoint_changes * 0.1
        total_cost = energy_cost + comfort_penalty + change_penalty
        return total_cost
    
    def optimize_with_scipy(self, forecast, horizon=24):
        n_variables = horizon
        bounds = [(self.temp_min, self.temp_max) for _ in range(n_variables)]
        x0 = [22.0] * n_variables
        def objective(setpoints):
            schedule = self._create_schedule_from_setpoints(setpoints, horizon)
            return self._calculate_total_cost(schedule, forecast)
        constraints = self._create_optimization_constraints(horizon)
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            optimal_setpoints = result.x
            optimal_schedule = self._create_schedule_from_setpoints(optimal_setpoints, horizon)
            return optimal_schedule
        else:
            return self._grid_search_optimization(forecast, horizon)
    
    def _create_optimization_constraints(self, horizon):
        constraints = []
        max_temp_change = 2.0
        def temp_change_constraint(setpoints):
            changes = np.abs(np.diff(setpoints))
            return max_temp_change - np.max(changes)
        constraints.append({
            'type': 'ineq',
            'fun': temp_change_constraint
        })
        return constraints
    
    def evaluate_schedule(self, schedule, actual_data=None):
        total_energy = schedule['predicted_energy'].sum()
        total_comfort_violations = schedule['comfort_violation'].sum()
        avg_setpoint = schedule['temperature_setpoint'].mean()
        setpoint_variability = schedule['temperature_setpoint'].std()
        metrics = {
            'total_energy_kwh': total_energy,
            'total_comfort_violations': total_comfort_violations,
            'average_setpoint': avg_setpoint,
            'setpoint_variability': setpoint_variability,
            'total_cost': self._calculate_total_cost(schedule, None)
        }
        if actual_data is not None:
            actual_energy = actual_data['energy_kwh'].sum()
            energy_error = abs(total_energy - actual_energy) / actual_energy * 100
            metrics['energy_prediction_error_percent'] = energy_error
        return metrics
    
    def generate_baseline_schedule(self, horizon=24, constant_temp=22):
        setpoints = [constant_temp] * horizon
        return self._create_schedule_from_setpoints(setpoints, horizon)
