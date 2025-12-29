import sys
import os
import inspect

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer.data_analyzer import DataAnalyzer

def test_structure():
    print("Testing DataAnalyzer structure...")
    analyzer = DataAnalyzer()
    
    # Check attributes
    assert hasattr(analyzer, 'unsupervised_model'), "Missing 'unsupervised_model' attribute"
    assert hasattr(analyzer, 'unsupervised_type'), "Missing 'unsupervised_type' attribute"
    assert hasattr(analyzer, 'rl_model_path'), "Missing 'rl_model_path' attribute"
    
    # Check method signatures
    sig_save = inspect.signature(analyzer.save_model)
    assert 'file_path' in sig_save.parameters, "save_model missing 'file_path'"
    
    sig_rl = inspect.signature(analyzer.train_rl_agent)
    assert 'save_path' in sig_rl.parameters, "train_rl_agent missing 'save_path'"
    
    print("Structure verification passed!")

if __name__ == "__main__":
    test_structure()
