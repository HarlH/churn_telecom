import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from category_encoders import TargetEncoder

def get_preprocessor():
    """
    Returns a preprocessor to transform categorical and numerical variables
    - Categorical variables: Imputation using the most frequent value and encoding using Target Encoding
    - Numerical variables: Imputation using the median
    """
    cat_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Pipeline for categorical variables
    cat_transformer = Pipeline([
        ('cat_imput', CategoricalImputer(imputation_method = 'frequent')),
        ('cat_encoding', TargetEncoder())
    ])

    # Pipeline for numerical variables
    num_transformer = Pipeline([
        ('num_imput', MeanMedianImputer(imputation_method = 'median'))
    ])

    # Applying the transformers
    preprocessor = ColumnTransformer(
        transformers = [
            ('cat', cat_transformer, cat_features),
            ('num', num_transformer, num_features)
        ],
        remainder = 'passthrough'
    )

    return preprocessor