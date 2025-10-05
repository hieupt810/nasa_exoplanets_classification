import warnings
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer

from .utils import load_cumulative, load_k2, load_toi, predict

warnings.filterwarnings(action='ignore')

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

    # Load k2 model
    model, classes, imputer, scaler = load_k2('k2pandc_2025.10.03_11.39.15.csv')
    models['k2'] = {
        'model': model,
        'classes': classes,
        'imputer': imputer,
        'scaler': scaler,
    }

    # Load TOI model
    model, classes, imputer, scaler = load_toi('TOI_2025.10.03_12.03.31.csv')
    models['toi'] = {
        'model': model,
        'classes': classes,
        'imputer': imputer,
        'scaler': scaler,
    }

    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


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
):
    model_info = models['cumulative']
    model = model_info['model']
    classes = model_info['classes']
    imputer: SimpleImputer = model_info['imputer']
    scaler: StandardScaler = model_info['scaler']

    input_data = [
        [
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

    return predict(model, classes, imputer, scaler, input_data)


@app.get('/k2')
def predict_k2(
    default_flag: Optional[int] = None,
    sy_pnum: Optional[int] = None,
    disc_year: Optional[int] = None,
    rv_flag: Optional[int] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    glat: Optional[float] = None,
    glon: Optional[float] = None,
    elat: Optional[float] = None,
    pl_nnotes: Optional[int] = None,
):
    model_info = models['k2']
    model = model_info['model']
    classes = model_info['classes']
    imputer: SimpleImputer = model_info['imputer']
    scaler: StandardScaler = model_info['scaler']

    input_data = [
        [
            default_flag,
            sy_pnum,
            disc_year,
            rv_flag,
            ra,
            dec,
            glat,
            glon,
            elat,
            pl_nnotes,
        ]
    ]

    return predict(model, classes, imputer, scaler, input_data)


@app.get('/toi')
def predict_toi(
    toi: Optional[float] = None,
    tid: Optional[int] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    pl_tranmid: Optional[float] = None,
    pl_trandurh: Optional[float] = None,
    pl_trandep: Optional[float] = None,
    st_tmag: Optional[float] = None,
    st_tmagerr1: Optional[float] = None,
):
    model_info = models['toi']
    model = model_info['model']
    classes = model_info['classes']
    imputer: SimpleImputer = model_info['imputer']
    scaler: StandardScaler = model_info['scaler']

    input_data = [
        [
            toi,
            tid,
            ra,
            dec,
            pl_tranmid,
            pl_trandurh,
            pl_trandep,
            st_tmag,
            st_tmagerr1,
        ]
    ]

    return predict(model, classes, imputer, scaler, input_data)
