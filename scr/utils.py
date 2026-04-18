import joblib
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """
    Loads the data and performs necessary treatments
    """
    df = pd.read_csv(path)

    # Correcting the mapping of SeniorCitizen
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 0: 'Yes'})

    # Converting the target variable to binary
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Handling missing values and filling problems
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    return df

def split_data(X, y, test_size = 0.20, random_state = 42):
    """
    Splits the data into train and test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def load_model(path):
    """
    Loads the trained model
    """
    return joblib.load(path)