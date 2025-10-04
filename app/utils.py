import os

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from .constants import Constant


def load_model(filename: str):
    model_path = os.path.join(Constant.MODEL_DIR, filename)
    model = joblib.load(model_path)
    return model


def load_cumulative(datafile: str):
    COLUMN_FOR_CLASSIFICATION = 'koi_pdisposition'

    path = os.path.join(Constant.DATA_DIR, datafile)
    df: pd.DataFrame = pd.read_csv(path)

    # Encode target labels
    label_encoder = LabelEncoder()
    df[COLUMN_FOR_CLASSIFICATION] = label_encoder.fit_transform(
        df[COLUMN_FOR_CLASSIFICATION]
    )

    # Preprocess data
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis=1)
    df = df.drop(columns=[COLUMN_FOR_CLASSIFICATION, 'rowid', 'kepid'], axis=1)

    imputer = SimpleImputer(strategy='mean')
    imputed_features = imputer.fit_transform(df)
    X = pd.DataFrame(imputed_features, columns=df.columns)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)
    X = pd.DataFrame(scaled_features, columns=X.columns)

    # Load model
    model = load_model('rf_cumulative.pkl')

    return model, label_encoder.classes_, imputer, scaler
