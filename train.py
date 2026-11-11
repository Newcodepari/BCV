import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.preprocess import run_preprocessing

def train_and_evaluate(data_path: str, artifact_dir: str = "artifacts"):
    X_train, X_test, y_train, y_test = run_preprocessing(data_path, artifact_dir)
    
    # Candidate Algorithms
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        'SVM': {
            'model': SVC(class_weight='balanced', probability=True, random_state=42),
            'params': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
        }
    }
    
    best_overall_model = None
    best_overall_score = 0.0
    best_model_name = ""
    
    print("\n" + "="*50)
    print(" MODEL TRAINING & EVALUATION REPORT ")
    print("="*50)
    
    for name, config in models.items():
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='recall', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n--- Model: {name} ---")
        print(f"Best Params: {grid.best_params_}")
        print(f"Accuracy : {acc:.4f} | Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f} | F1-Score : {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Priority selection based on target recall
        if rec > best_overall_score:
            best_overall_score = rec
            best_overall_model = best_model
            best_model_name = name
            
    print(f"\nSelected Model for Deployment: {best_model_name} (Recall: {best_overall_score:.4f})")
    
    # Save selected model
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(best_overall_model, os.path.join(artifact_dir, 'model.pkl'))
    print("Model saved to artifacts/model.pkl")

if __name__ == "__main__":
    train_and_evaluate("data/data.csv")