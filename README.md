    # Stocks On V2.0


    Full Streamlit-ready app package for Stocks On V2.0.


    ## Files
    - stocks_on_v2_app.py : Main Streamlit app
    - models/xgb_model.py  : XGBoost helpers
    - explainer/shap_explain.py : SHAP explainability wrapper
    - backtest/walkforward.py : Walk-forward backtest helper
    - requirements.txt : dependencies

Usage:
1. (Optional) Pre-train XGBoost models offline and place the model files under models/ named 'xgb_global.model' or 'xgb_{TICKER}.model'.
2. Upload repository to GitHub (public) and deploy on Streamlit Cloud. Set the main file to `stocks_on_v2_app.py`.

Notes:
- If XGBoost or SHAP are not available on the host, the app will gracefully fall back to the RF+LR ensemble for predictions.
- For heavy training (LSTM/Transformers), train offline; this package focuses on inference and explainability in Streamlit.
