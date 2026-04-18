import pandas as pd
import numpy as np
from predict import load_model
from utils import split_data, load_data
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluation(model, X_test, y_test):
    """
    Evaluates the model's performance on the test set
    """
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    return pd.Series(results)

def cross_validation(model, X, y):
    """
    Performs cross-validation using F1 Score
    """
    scoring = make_scorer(f1_score)
    cv = StratifiedKFold(n_splits = 5)

    scores = cross_val_score(model, X, y, cv = cv, scoring = scoring)
    print("Cross Validation")
    print(f"{'-' * 25}")
    print(f"Mean of F1 Scores: {scores.mean()}")
    print(f"Standard Deviation of F1 Scores: {scores.std()}")

    return scores.mean(), scores.std(), scores

if __name__=="__main__":
    # Load the data and the trained model
    data = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model = load_model("models/classifier.pkl")

    # Separate features and target
    X = data.drop(columns = ['Churn', 'customerID'], axis = 1)
    y = data['Churn']

    # Split the data into train and test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # perform cross-validation
    mean_f1, std_f1, all_scores = cross_validation(model, X_train, y_train)

    # Evaluate the model on the test set
    print("\nEvaluation metrics")
    print(f"{'-' * 25}")
    print(evaluation(model, X_test, y_test))
   