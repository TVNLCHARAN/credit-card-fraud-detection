import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  roc_auc_score
from imblearn.over_sampling import SMOTE
from model import build_isolation_forest
from preprocess import preprocess_data
import warnings

warnings.filterwarnings(action='ignore')

def train_model():
    try:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/creditcard.csv'))
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    df = df.sample(frac = 0.1, random_state=42)

    df_cleaned = preprocess_data(df)

    X = df_cleaned.drop('Class', axis=1)
    y = df_cleaned['Class']

    X_train = pd.DataFrame(columns=X.columns)
    y_train = pd.Series(dtype=y.dtype)
    X_test = pd.DataFrame(columns=X.columns)
    y_test = pd.Series(dtype=y.dtype)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ind, (train_idx, _) in enumerate(skf.split(X, y)):
        if ind >= 3:
            X_test = pd.concat([X_test, X.iloc[train_idx]], axis=0)
            y_test = pd.concat([y_test, y.iloc[train_idx]], axis=0)
            # print(y_test.value_counts())
        else:
            X_train = pd.concat([X_train, X.iloc[train_idx]], axis=0)
            y_train = pd.concat([y_train, y.iloc[train_idx]], axis=0)
            # print(X_train.shape, y_train.shape)
            # print(y_train.value_counts())

    # smote = SMOTE(random_state=42)
    # X_train_balanced, y_train_balanced = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trained_model, y_pred_mapped = build_isolation_forest(X_train_scaled, X_test_scaled, y_test)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_mapped):.4f}")

    return trained_model

if __name__ == '__main__':
    trained_model = train_model()