import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def load_and_clean_data(filepath: str):
    """Loads dataset, handles target encoding, drops unnecessary columns."""
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns if present (e.g., 'id' or unnamed trailing columns)
    cols_to_drop = [col for col in ['id', 'Unnamed: 32'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Map target: M = 1 (Malignant), B = 0 (Benign)
    if df['diagnosis'].dtype == 'object':
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        
    return df

def build_preprocessing_pipeline():
    """Builds scaling and preprocessing pipeline."""
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    return pipeline

def run_preprocessing(data_path: str, output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_and_clean_data(data_path)
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    
    # Train / Validation / Test split (80/20 train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit preprocessing pipeline on training data
    preprocessor = build_preprocessing_pipeline()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save preprocessor artifact
    joblib.dump(preprocessor, os.path.join(output_dir, 'pipeline.pkl'))
    
    print(f"Data Split Successfully: Train shape {X_train.shape}, Test shape {X_test.shape}")
    print(f"Class Distribution: {np.bincount(y_train)} (0=Benign, 1=Malignant)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    run_preprocessing("data/data.csv")