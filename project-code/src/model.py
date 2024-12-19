from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

def build_isolation_forest(X_train, X_test, y_test, contamination=0.01, random_state=42):
    model = IsolationForest(n_estimators=100,contamination=contamination, random_state=random_state)

    model.fit(X_train)
    scores_prediction = model.decision_function(X_train)
    y_pred = model.predict(X_test)

    y_pred_mapped = np.where(y_pred == -1, 1, 0)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mapped))
    print("Classification Report:\n", classification_report(y_test, y_pred_mapped))

    return model, y_pred_mapped