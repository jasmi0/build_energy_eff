import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:    
    def __init__(self):
        self.feature_columns = []
        self.target_column = 'energy_kwh'
    
    def process_data(self, data):
        df = data.copy()
        
        df = self._clean_data(df)
        df = self._process_datetime(df)
        df = self._handle_missing_values(df)
        df = self._engineer_features(df)
        df = self._resample_data(df)
        
        return df
    
    def _clean_data(self, df):
        column_mapping = {
            'datetime': 'datetime',
            'timestamp': 'datetime',
            'time': 'datetime',
            'zone_temp': 'temperature',
            'zone_temps': 'temperature',
            'temp': 'temperature',
            'temperature': 'temperature',
            'humidity': 'humidity',
            'rh': 'humidity',
            'relative_humidity': 'humidity',
            'occupancy': 'occupancy',
            'occ': 'occupancy',
            'kwh': 'energy_kwh',
            'energy': 'energy_kwh',
            'power': 'energy_kwh',
            'energy_consumption': 'energy_kwh'
        }
        
        df.columns = df.columns.str.lower()
        df = df.rename(columns=column_mapping)
        
        return df
    
    def _process_datetime(self, df):
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        else:
            df.index = pd.date_range(
                start='2023-01-01',
                periods=len(df),
                freq='H'
            )
        
        df = df.sort_index()
        
        return df
    
    def _handle_missing_values(self, df):
        df = df.fillna(method='ffill', limit=3)
        df = df.fillna(method='bfill', limit=3)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
        critical_columns = ['energy_kwh']
        critical_existing = [col for col in critical_columns if col in df.columns]
        if critical_existing:
            df = df.dropna(subset=critical_existing)
        return df
    
    def _engineer_features(self, df):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_business_hour'] = ((df.index.hour >= 8) & (df.index.hour <= 18)).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        if 'energy_kwh' in df.columns:
            for lag in [1, 2, 3, 24, 168]:
                df[f'energy_lag_{lag}'] = df['energy_kwh'].shift(lag)
        if 'energy_kwh' in df.columns:
            df['energy_rolling_24h'] = df['energy_kwh'].rolling(window=24).mean()
            df['energy_rolling_24h_std'] = df['energy_kwh'].rolling(window=24).std()
        if 'temperature' in df.columns:
            df['temp_rolling_24h'] = df['temperature'].rolling(window=24).mean()
            df['temp_rolling_24h_std'] = df['temperature'].rolling(window=24).std()
            df['temp_deviation'] = np.abs(df['temperature'] - 22)
            df['hdd'] = np.maximum(18 - df['temperature'], 0)
            df['cdd'] = np.maximum(df['temperature'] - 25, 0)
        if 'occupancy' in df.columns:
            df['occ_rolling_24h'] = df['occupancy'].rolling(window=24).mean()
            df['occ_change'] = df['occupancy'].diff()
        if 'humidity' in df.columns:
            df['humidity_rolling_24h'] = df['humidity'].rolling(window=24).mean()
            if 'temperature' in df.columns:
                df['comfort_index'] = self._calculate_comfort_index(
                    df['temperature'], df['humidity']
                )
        if 'energy_kwh' in df.columns and 'temperature' in df.columns:
            df['energy_per_degree'] = df['energy_kwh'] / (df['temp_deviation'] + 1e-6)
        return df
    
    def _calculate_comfort_index(self, temperature, humidity):
        temp_comfort = 1 - np.abs(temperature - 22) / 10
        humid_comfort = 1 - np.abs(humidity - 50) / 50
        comfort = (temp_comfort + humid_comfort) / 2
        return np.clip(comfort, 0, 1)
    
    def _resample_data(self, df, freq='H'):
        df_resampled = df.resample(freq).mean()
        df_resampled = df_resampled.interpolate(method='linear')
        return df_resampled
    
    def get_feature_columns(self, df):
        exclude_columns = [
            'energy_kwh',
        ]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        missing_threshold = 0.5
        for col in feature_columns.copy():
            if df[col].isnull().sum() / len(df) > missing_threshold:
                feature_columns.remove(col)
        self.feature_columns = feature_columns
        return feature_columns
    
    def prepare_modeling_data(self, df, test_size=0.2):
        feature_columns = self.get_feature_columns(df)
        X = df[feature_columns].copy()
        y = df[self.target_column].copy()
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test
