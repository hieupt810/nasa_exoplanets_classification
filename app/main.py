from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer

from app.utils import load_cumulative

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models

    # Load cumulative model
    model, classes, imputer, scaler = load_cumulative(
        'cumulative_2025.10.03_12.03.43.csv'
    )
    models['cumulative'] = {
        'model': model,
        'classes': classes,
        'imputer': imputer,
        'scaler': scaler,
    }

    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


class ResponseModel(BaseModel):
    result: str


@app.get('/cumulative')
def predict_cumulative(
    koi_fpflag_nt: Optional[int] = None,
    koi_fpflag_ss: Optional[int] = None,
    koi_fpflag_co: Optional[int] = None,
    koi_fpflag_ec: Optional[int] = None,
    koi_period: Optional[float] = None,
    koi_time0bk: Optional[float] = None,
    koi_duration: Optional[float] = None,
    koi_count: Optional[int] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
) -> ResponseModel:
    model_info = models['cumulative']
    model = model_info['model']
    classes = model_info['classes']
    imputer: SimpleImputer = model_info['imputer']
    scaler: StandardScaler = model_info['scaler']

    input_data = [
        [
            0,
            koi_fpflag_nt,
            koi_fpflag_ss,
            koi_fpflag_co,
            koi_fpflag_ec,
            koi_period,
            koi_time0bk,
            koi_duration,
            koi_count,
            ra,
            dec,
        ]
    ]

    # Scale features
    input_data = scaler.transform(input_data)

    # Impute missing values
    input_data = imputer.transform(input_data)

    prediction = model.predict(np.array(input_data[0][:-1]).reshape(1, -1))
    predicted_class = classes[prediction[0]]

    return ResponseModel(result=predicted_class)
