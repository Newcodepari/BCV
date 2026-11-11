import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from preprocess import run_preprocessing

def train_and_evaluate(data_path: str, artifact_dir: str = 'artifacts'):
    X_train, X_test, y_train, y_test = run_preprocessing(data_path, artifact_dir)
    
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        'SVM': {
            'model': SVC(class_weight='balanced', probability=True, random_state=42),
            'params': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        }
    }
    
    best_overall_model = None
    best_overall_score = 0
    best_model_name = ''
    
    for name, config in models.items():
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='recall', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        rec = recall_score(y_test, y_pred, pos_label=1)
        print(f'Model: {name} | Recall: {rec:.4f}')
        if rec > best_overall_score:
            best_overall_score = rec
            best_overall_model = best_model
            best_model_name = name
            
    print(f'Selected Model: {best_model_name} (Recall: {best_overall_score:.4f})')
    joblib.dump(best_overall_model, os.path.join(artifact_dir, 'model.pkl'))
    print('Model saved to artifacts/model.pkl')

if __name__ == '__main__':
    train_and_evaluate('data/data.csv')
