import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer.data_analyzer import Data_Analyzer, XGBOOST_AVAILABLE, PROPHET_AVAILABLE

def test_new_features():
    print("--- Starting Advanced Algorithms Verification ---")
    print(f"XGBoost Available: {XGBOOST_AVAILABLE}")
    print(f"Prophet Available: {PROPHET_AVAILABLE}")
    
    analyzer = Data_Analyzer()

    # 1. Test XGBoost (Regression)
    print("\n[Test 1] XGBoost Regression")
    # Create regression data
    rows = 100
    df_reg = pd.DataFrame({
        'feature1': np.random.rand(rows),
        'feature2': np.random.normal(0, 1, rows),
        'target': np.random.rand(rows) * 100
    })
    analyzer.df = df_reg
    try:
        analyzer.train_model('target')
        # Check if XGBoost was considered (would appear in logs, but here we check no crash)
        print("SUCCESS: Regression Training completed.")
        if hasattr(analyzer.model_pipeline, 'named_steps'):
            model_name = analyzer.model_pipeline.named_steps['model'].__class__.__name__
            print(f"Selected Model: {model_name}")
    except Exception as e:
        print(f"FAILURE: Regression Training Failed. {e}")

    # 2. Test Prophet (Forecasting)
    print("\n[Test 2] Prophet Forecasting")
    if PROPHET_AVAILABLE:
        # Create time series data
        dates = [datetime.now() + timedelta(days=x) for x in range(rows)]
        df_ts = pd.DataFrame({
            'date_col': dates,
            'value': np.sin(np.linspace(0, 10, rows)) * 10 + np.random.normal(0, 1, rows)
        })
        analyzer.df = df_ts
        try:
            results = analyzer.run_forecasting('value', 'date_col', periods=10)
            print("SUCCESS: Forecasting completed.")
            print(f"Forecast Tail:\n{results['forecast_tail']}")
        except Exception as e:
            print(f"FAILURE: Forecasting Failed. {e}")
    else:
        print("SKIPPING: Prophet not installed.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_new_features()
