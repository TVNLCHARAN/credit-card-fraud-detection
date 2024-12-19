import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def preprocess_data(df):

    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']

    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    df_no_outliers_z_score = df[(np.abs(stats.zscore(df[['scaled_amount', 'scaled_time']])) < 3).all(axis=1)]

    df_cleaned = remove_outliers_iqr(df_no_outliers_z_score, 'scaled_amount')
    df_cleaned = remove_outliers_iqr(df_cleaned, 'scaled_time')

    # corr_features = ['V3', 'V12', 'V14', 'V18']
    # for feature in corr_features:
    #     df_cleaned = remove_outliers_iqr(df_cleaned, feature)

    iso = IsolationForest(contamination=0.01, random_state=42)
    outliers = iso.fit_predict(df_cleaned[['scaled_amount', 'scaled_time']])

    df_filtered = df_cleaned[outliers == 1]
    
    return df_filtered
