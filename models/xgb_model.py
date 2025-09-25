"""
XGBoost helpers for Stocks On V2.0
- save_model / load_model / predict_from_df
"""
import xgboost as xgb
import pandas as pd
import numpy as np

def save_model(bst: xgb.Booster, path: str):
    bst.save_model(path)

def load_model(path: str):
    bst = xgb.Booster(); bst.load_model(path); return bst

def predict_from_df(bst, X):
    dmat = xgb.DMatrix(X.values)
    return bst.predict(dmat)
