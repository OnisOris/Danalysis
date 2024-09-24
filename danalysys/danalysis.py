import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def poly_reg(x: np.ndarray, t: np.ndarray, degree: int=25):
    t_r = t.reshape(-1, 1)    
    
    # Заготовка под будущий модуль
