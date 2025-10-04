import os
from dataclasses import dataclass


@dataclass
class Constant:
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'jupyter', 'models')
