import pandas as pd
import numpy as np
from utils import load_model

def make_predictions(model, data):
    """
    Generates predictions and probabilities with a trained model
    """
    probabilities = model.predict_proba(data)[:,1]
    predictions = (probabilities > 0.40).astype(int)
    
    return predictions, probabilities

if __name__=="__main__":
    # Load the test data
    data = pd.read_csv("data/processed/test.csv")

    # Load the trained model
    model = load_model("models/classifier.pkl")

    # Make predictions
    predictions, probabilities = make_predictions(model, data)

    # Add predictions to the DataFrame
    data['predicted'] = predictions
    data['pred_probability'] = probabilities

    # Select only the necessary columns
    data = data[['Churn', 'predicted', 'pred_probability']]

    # Save predictions to an .xlsx file
    data.to_excel("data/processed/predictions.xlsx", index=False)
    print(f'\nPredictions saved to "data/processed/predictions.xlsx"')