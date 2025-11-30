import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_process_data():
    """
    Loads Iris dataset, adds a synthetic 'location' attribute,
    and splits into train/test sets.
    """
    # Load Iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Introduce "location" attribute (0 and 1 randomly)
    np.random.seed(42)  # Fixed seed for reproducibility
    X['location'] = np.random.randint(0, 2, size=len(X))
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("✅ Data loaded and split successfully.")
    print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns
