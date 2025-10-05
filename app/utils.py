import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .constants import Constant


def load_model(filename: str):
    model_path = os.path.join(Constant.MODEL_DIR, filename)
    model = joblib.load(model_path)
    return model


def predict(model, classes, imputer, scaler, input_data):
    input_data = scaler.transform(input_data)
    input_data = imputer.transform(input_data)

    # Make prediction
    prediction = model.predict_proba(input_data)
    pred_idx = int(np.argmax(prediction, axis=1)[0])

    # Resolve predicted class label
    predicted_class = (
        classes[pred_idx] if classes is not None else model.classes_[pred_idx]
    )
    probability_pct = float(prediction[0, pred_idx] * 100)
    return {
        'predicted_class': predicted_class,
        'probability': f'{probability_pct:.2f}%',
    }


def load_cumulative(datafile: str):
    COLUMN_FOR_CLASSIFICATION = 'koi_pdisposition'

    path = os.path.join(Constant.DATA_DIR, datafile)
    df: pd.DataFrame = pd.read_csv(path)

    # Encode target labels
    label_encoder = LabelEncoder()
    df[COLUMN_FOR_CLASSIFICATION] = label_encoder.fit_transform(
        df[COLUMN_FOR_CLASSIFICATION]
    )

    # Load model
    model = load_model('rf_cumulative.pkl')
    imputer = load_model('imputer_cumulative.pkl')
    scaler = load_model('scaler_cumulative.pkl')

    return model, label_encoder.classes_, imputer, scaler


def load_k2(datafile: str):
    COLUMN_FOR_CLASSIFICATION = 'disposition'

    path = os.path.join(Constant.DATA_DIR, datafile)
    df: pd.DataFrame = pd.read_csv(path)

    # Encode target labels
    label_encoder = LabelEncoder()
    df[COLUMN_FOR_CLASSIFICATION] = label_encoder.fit_transform(
        df[COLUMN_FOR_CLASSIFICATION]
    )

    # Load model
    model = load_model('rf_k2.pkl')
    imputer = load_model('imputer_k2.pkl')
    scaler = load_model('scaler_k2.pkl')

    return model, label_encoder.classes_, imputer, scaler


def load_toi(datafile: str):
    COLUMN_FOR_CLASSIFICATION = 'tfopwg_disp'

    path = os.path.join(Constant.DATA_DIR, datafile)
    df: pd.DataFrame = pd.read_csv(path)

    # Encode target labels
    label_encoder = LabelEncoder()
    df[COLUMN_FOR_CLASSIFICATION] = label_encoder.fit_transform(
        df[COLUMN_FOR_CLASSIFICATION]
    )

    # Load model
    model = load_model('rf_toi.pkl')
    imputer = load_model('imputer_toi.pkl')
    scaler = load_model('scaler_toi.pkl')

    return model, label_encoder.classes_, imputer, scaler
