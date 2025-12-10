import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer.data_analyzer import Data_Analyzer

def test_optimization_flow():
    print("--- Starting Optimization Verification ---")
    
    # 1. Create Dummy Data
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.randint(0, 100, 100),
        'target': np.random.rand(100) * 10
    }
    df = pd.DataFrame(data)
    csv_path = "test_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Created dummy CSV: {csv_path}")

    analyzer = Data_Analyzer()

    # 2. Test Data Loading (PyArrow)
    try:
        print("Testing Data Loading (PyArrow)...")
        analyzer.load_data(csv_path)
        print("SUCCESS: Data Loaded.")
    except Exception as e:
        print(f"FAILURE: Data Loading Failed. {e}")
        return

    # 3. Test Training (Refactoring)
    try:
        print("Testing Model Training (Refactoring)...")
        analyzer.train_model("target")
        print("SUCCESS: Model Trained.")
    except Exception as e:
        print(f"FAILURE: Model Training Failed. {e}")
        return

    # 4. Test Model Saving (Compression)
    model_path = "test_model_compressed.joblib"
    try:
        print("Testing Model Saving (Compression)...")
        analyzer.save_model(model_path)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"SUCCESS: Model Saved. Size: {size} bytes")
        else:
            print("FAILURE: Model file not found.")
    except Exception as e:
        print(f"FAILURE: Model Saving Failed. {e}")
        return

    # Cleanup
    if os.path.exists(csv_path): os.remove(csv_path)
    if os.path.exists(model_path): os.remove(model_path)
    print("--- Verification Complete ---")

if __name__ == "__main__":
    test_optimization_flow()
