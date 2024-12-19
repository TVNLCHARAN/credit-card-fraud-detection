# Credit Card Fraud Detection

This project demonstrates detecting credit card fraud using Kaggle's credit card fraud detection dataset. The primary focus is data preprocessing, exploratory data analysis (EDA), and applying the Isolation Forest algorithm for anomaly detection.

## Dataset
The dataset is sourced from Kaggle and contains anonymized transaction details. It includes features such as time, amount, and engineered principal components (V1-V28). The target class indicates if a transaction is fraudulent (1) or legitimate (0).

## Steps Followed

### 1. Data Preprocessing
- Removed outliers using Z-score and Interquartile Range (IQR) methods.
- Normalized the 'Amount' and 'Time' features.

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions of features.
- Analyzed class imbalance and feature correlations.

### 3. Sampling
- Used `StratifiedKFold` to take only 10% of the original dataset while preserving the class distribution.

### 4. Model Training
- Applied the **Isolation Forest** algorithm, designed to detect anomalies efficiently. It isolates outliers using random partitioning of features.
- Model evaluation metrics were generated based on the test set.

### How Isolation Forest Works
The Isolation Forest algorithm works by:
- Creating random splits in feature space.
- Isolating data points that require fewer splits, marking them as anomalies.
- Aggregating results across trees to determine anomaly scores.

### Results
**Confusion Matrix:**
```
 [[39305   339]
 [   13    41]]
```

**Classification Report:**
```
               precision    recall  f1-score   support

           0       1.00      0.99      1.00     39644
           1       0.11      0.76      0.19        54

    accuracy                           0.99     39698
   macro avg       0.55      0.88      0.59     39698
weighted avg       1.00      0.99      0.99     39698
```

**ROC-AUC Score:** `0.8754`

## Conclusion
The model effectively detected fraud while balancing recall and precision. The Isolation Forest algorithm demonstrated its potential in handling imbalanced datasets in fraud detection tasks.

## How to Run
1. Clone the repository.
2. Download creditcard.csv data and place it in data folder.
2. Install required packages: `pip install -r requirements.txt`
3. Run `train.py` to train the model.

Feel free to contribute by raising issues or submitting pull requests! 🚀
