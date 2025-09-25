"""
Walk-forward backtesting utilities (simple expanding-window)
"""
import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from typing import Callable, Dict, Any

def walk_forward_evaluate(X: pd.DataFrame, y: pd.Series, n_splits: int,
                           train_model_fn: Callable[[pd.DataFrame, pd.Series, Dict[str,Any]], Any],
                           model_kwargs: Dict[str,Any]=None):
    model_kwargs = model_kwargs or {}
    n = len(X)
    if n_splits < 2: raise ValueError("n_splits >=2 required")
    fold_size = n // (n_splits + 1)
    metrics=[]; models=[]
    for i in range(1, n_splits+1):
        train_end = fold_size * i; test_start = train_end; test_end = min(n, test_start + fold_size)
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        model = train_model_fn(X_train, y_train, model_kwargs)
        preds = predict_with_model(model, X_test)
        mape = mean_absolute_percentage_error(y_test.values, preds) * 100.0
        dir_acc = np.mean((np.sign(preds - X_test['Close'].values) == np.sign(y_test.values - X_test['Close'].values)).astype(int)) * 100.0
        metrics.append({"fold": i, "mape": mape, "directional_accuracy": dir_acc, "n_test": len(X_test)})
        models.append(model)
    df_metrics = pd.DataFrame(metrics)
    summary = {"median_mape": df_metrics["mape"].median(), "median_dir_acc": df_metrics["directional_accuracy"].median()}
    return df_metrics, summary, models

def predict_with_model(model, X_test):
    try:
        import xgboost as xgb
        if isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(X_test.values)
            preds = model.predict(dmat); return preds
    except Exception:
        pass
    if hasattr(model, "predict"):
        return model.predict(X_test)
    raise ValueError("Unknown model interface")
