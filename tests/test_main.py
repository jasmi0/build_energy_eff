import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import DataProcessor
from src.utils import load_sample_data, validate_data_quality

class TestDataProcessor:
    """Test data processing functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.processor = DataProcessor()
        self.sample_data = load_sample_data()
    
    def test_load_sample_data(self):
        """Test sample data generation."""
        data = load_sample_data()
        
        assert len(data) > 0
        assert 'datetime' in data.columns
        assert 'energy_kwh' in data.columns
        assert 'temperature' in data.columns
        assert data['energy_kwh'].min() >= 0
        assert data['temperature'].count() > 0
    
    def test_process_data(self):
        """Test data processing pipeline."""
        processed = self.processor.process_data(self.sample_data)
        
        assert len(processed) > 0
        assert processed.index.dtype == 'datetime64[ns]'
        assert 'hour' in processed.columns
        assert 'day_of_week' in processed.columns
        assert processed['energy_kwh'].isnull().sum() == 0
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        processed = self.processor.process_data(self.sample_data)
        
        # Check cyclical features
        assert 'hour_sin' in processed.columns
        assert 'hour_cos' in processed.columns
        
        # Check lag features
        lag_columns = [col for col in processed.columns if 'lag' in col]
        assert len(lag_columns) > 0
        
        # Check rolling features
        rolling_columns = [col for col in processed.columns if 'rolling' in col]
        assert len(rolling_columns) > 0
    
    def test_prepare_modeling_data(self):
        """Test data preparation for modeling."""
        processed = self.processor.process_data(self.sample_data)
        X_train, X_test, y_train, y_test = self.processor.prepare_modeling_data(processed)
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert X_train.isnull().sum().sum() == 0

class TestUtils:
    """Test utility functions."""
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        data = load_sample_data()
        report = validate_data_quality(data)
        
        assert 'total_records' in report
        assert 'missing_values' in report
        assert 'quality_score' in report
        assert report['total_records'] > 0
        assert 0 <= report['quality_score'] <= 100
    
    def test_data_quality_with_issues(self):
        """Test data quality with problematic data."""
        data = load_sample_data()
        
        # Introduce some issues
        data.loc[:50, 'energy_kwh'] = np.nan  # Missing values
        data.loc[100:110, 'temperature'] = 100  # Unrealistic temperature
        
        report = validate_data_quality(data)
        
        assert len(report['issues']) > 0
        assert report['quality_score'] < 90  # Should be lower due to issues

if __name__ == "__main__":
    pytest.main([__file__])
