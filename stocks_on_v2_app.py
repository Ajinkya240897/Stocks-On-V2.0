# Stocks On V2.0 — Full Streamlit App (ready-to-upload)
# Combines V1 features + XGBoost inference + SHAP explainability + walk-forward backtest helpers.
# Notes:
# - For best results, pre-train XGBoost models offline and place them in models/ as 'xgb_global.model' or 'xgb_{TICKER}.model'.
# - If no XGBoost model found, the app falls back to RF+LR ensemble (like V1).
# - Keep heavy training offline; this app focuses on inference, diagnostics, and explainability.
# Educational only. Not financial advice.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests, time, os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import ta

# optional imports for V2
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from models.xgb_model import load_model as xgb_load_model, predict_from_df as xgb_predict_from_df if XGBOOST_AVAILABLE else (lambda *a, **k: None, lambda *a, **k: None)
from explainer.shap_explain import explain_tree_model if SHAP_AVAILABLE else (lambda *a, **k: (None, None, None))

# page config
st.set_page_config(page_title="Stocks On V2.0", layout="centered")

# CSS
st.markdown("""
<style>
body {background: linear-gradient(135deg, #071229 0%, #001219 100%); color: #E6EEF8}
.reportview-container .main .block-container{max-width:820px; padding:1rem}
.card {background: rgba(255,255,255,0.02); padding: 12px; border-radius: 10px;}
.small {font-size:0.90rem; color:#bcd9ff}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stocks On V2.0 — India Stock Predictor")
st.markdown("V2.0: XGBoost-ready inference + SHAP explainability + improved diagnostics. Educational only.")

with st.sidebar:
    st.markdown("### Inputs")
    ticker_raw = st.text_input("Ticker (without suffix)", value="RELIANCE")
    auto_ns = st.checkbox("Auto append .NS (NSE) if no suffix", value=True)
    interval = st.selectbox("Prediction interval", ["3-15 days", "1-3 months", "3-6 months", "1-3 years"])
    fmp_key = st.text_input("FMP API Key (optional)")
    use_xgb = st.checkbox("Prefer XGBoost model if available", value=True)
    show_shap = st.checkbox("Show SHAP explanation (if model available)", value=True)
    run_backtest = st.checkbox("Run quick backtest (may take time)", value=False)
    submit = st.button("Analyze")

# helper functions (similar to V1)
@st.cache_data(show_spinner=False)
def fetch_price_data(ticker, years=6):
    end = datetime.now(); start = end - timedelta(days=365*years)
    tk = yf.Ticker(ticker); df = tk.history(start=start, end=end, auto_adjust=False)
    if df is None or df.empty: return None
    df = df[['Open','High','Low','Close','Volume']].dropna(); return df

@st.cache_data(show_spinner=False)
def fetch_yf_info(ticker):
    try:
        info = yf.Ticker(ticker).info; return info if isinstance(info, dict) else None
    except Exception: return None

@st.cache_data(show_spinner=False)
def fetch_fmp_profile(ticker, apikey):
    if not apikey: return None
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={apikey}"
    try:
        r = requests.get(url, timeout=8)
        if r.ok:
            j = r.json()
            if isinstance(j, list) and len(j)>0: return j[0]
    except Exception: return None
    return None

def engineer_features(df):
    d = df.copy()
    d['return_1'] = d['Close'].pct_change(1)
    d['return_3'] = d['Close'].pct_change(3)
    d['ma_5'] = d['Close'].rolling(5).mean()
    d['ma_20'] = d['Close'].rolling(20).mean()
    d['vol_10'] = d['Close'].rolling(10).std()
    try: d['rsi'] = ta.momentum.rsi(d['Close'], window=14)
    except Exception: d['rsi'] = 50
    d = d.dropna(); return d

def prepare_data(df, horizon):
    df2 = df.copy(); df2['target'] = df2['Close'].shift(-horizon); df2 = df2.dropna()
    X = df2[['Close','return_1','return_3','ma_5','ma_20','vol_10','rsi']]; y = df2['target']; return X,y,df2

def compute_fundamentals_score(profile, yf_info=None):
    if profile is None and yf_info is None: return None
    score=40
    def safe_float(v):
        try:
            if v is None: return np.nan
            return float(v)
        except: return np.nan
    if profile:
        pe = safe_float(profile.get('pe') or profile.get('peRatio')); roe = safe_float(profile.get('returnOnEquity') or profile.get('roa') or profile.get('roe'))
        debt_to_equity = safe_float(profile.get('debtToEquity') or profile.get('debtEquity')); mcap = safe_float(profile.get('mktCap') or profile.get('marketCap'))
    else:
        pe = safe_float(yf_info.get('trailingPE') or yf_info.get('forwardPE') or yf_info.get('priceToEarnings') if yf_info else None)
        roe = safe_float(yf_info.get('returnOnEquity') or yf_info.get('roe') or np.nan if yf_info else None)
        debt_to_equity = safe_float(yf_info.get('debtToEquity') or yf_info.get('debtToEquityRatio') if yf_info else None)
        mcap = safe_float(yf_info.get('marketCap') or yf_info.get('mktCap') if yf_info else None)
    if not np.isnan(pe):
        if 0<pe<=12: score+=20
        elif pe<=25: score+=10
    if not np.isnan(roe):
        if roe>1: roe = roe/100.0
        if roe>=0.15: score+=20
        elif roe>=0.10: score+=10
    if not np.isnan(debt_to_equity):
        if debt_to_equity<0.5: score+=15
        elif debt_to_equity<1.0: score+=7
    if not np.isnan(mcap):
        if mcap>1e10: score+=5
    score = max(0, min(100, int(score))); return score

def momentum_label(df):
    try:
        cp = df['Close'].iloc[-1]; ret30 = (cp / df['Close'].iloc[-31] - 1) if len(df)>31 else np.nan
        ret90 = (cp / df['Close'].iloc[-91] - 1) if len(df)>91 else np.nan; rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
    except Exception:
        ret30 = np.nan; ret90 = np.nan; rsi = 50
    label = "Neutral / Sideways"
    if not np.isnan(ret30) and ret30>0.12 and rsi>60: label = "Strong Uptrend"
    elif not np.isnan(ret30) and ret30>0.04 and rsi>52: label = "Uptrend"
    elif not np.isnan(ret30) and ret30<-0.05 and rsi<40: label = "Downtrend / Weak"
    return label, ret30, ret90, rsi

def simple_recommendation(current_price, predicted_price, implied_return, momentum_label, fund_score, confidence, horizon_label):
    strong_buy_threshold = 12; buy_threshold = 6; sell_threshold = -5
    if implied_return >= strong_buy_threshold and (fund_score is None or fund_score>=60) and confidence>=65:
        decision='Strong BUY'
    elif implied_return >= buy_threshold and confidence>=55:
        decision='BUY'
    elif implied_return <= sell_threshold and confidence>=55:
        decision='SELL / AVOID'
    else:
        decision='HOLD / WATCH'
    # build text
    text = f\"Recommendation: {decision}\\n\\nCurrent price: ₹{current_price:,.2f}. Predicted: ₹{predicted_price:,.2f} (implied {implied_return:.2f}%).\\nMomentum: {momentum_label}. Confidence: {confidence:.1f}%.\")
    if decision in ['Strong BUY','BUY']:
        buy_price = current_price*0.995; take_profit = predicted_price*0.9
        text += f\"\\n\\nBeginner action: Consider buying near ₹{buy_price:,.2f} and consider partial profit taking near ₹{take_profit:,.2f}. Use stop-loss (6-10%).\"
    elif decision=='SELL / AVOID':
        text += \"\\n\\nBeginner action: If holding, consider trimming. Avoid new buys.\"
    else:
        text += \"\\n\\nBeginner action: Hold/watch and set alerts.\"
    return decision, text

# load optional xgboost model if exists
def load_xgb_model_for_ticker(ticker):
    # check for model files: models/xgb_{TICKER}.model or models/xgb_global.model
    model_paths = [f\"models/xgb_{ticker.replace('.NS','').upper()}.model\", \"models/xgb_global.model\", \"models/xgb.model\"]
    for p in model_paths:
        if os.path.exists(p):
            try:
                bst = xgb.Booster(); bst.load_model(p); return bst, p
            except Exception:
                try:
                    return xgb_load_model(p), p
                except Exception:
                    pass
    return None, None

# backtest helper (simple quick backtest using RF ensemble)
def quick_backtest(X, y):
    tscv = TimeSeriesSplit(n_splits=3)
    mape_list=[]; diracc_list=[]
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]; ytr, yte = y.iloc[tr], y.iloc[te]
        rf = RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42); lr = LinearRegression()
        rf.fit(Xtr, ytr); lr.fit(Xtr, ytr)
        preds = (rf.predict(Xte) + lr.predict(Xte))/2
        mape_list.append(mean_absolute_percentage_error(yte, preds)*100)
        diracc_list.append(np.mean((np.sign(preds - Xte['Close'].values) == np.sign(yte.values - Xte['Close'].values)).astype(int))*100)
    return np.median(mape_list), np.median(diracc_list)

# Main analyze flow
if submit:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.spinner('Fetching data and running models... (aim <40s)'):
        ticker = ticker_raw.strip().upper()
        yf_ticker = ticker + '.NS' if auto_ns and '.' not in ticker else ticker
        mapping = {'3-15 days':10, '1-3 months':60, '3-6 months':135, '1-3 years':550}
        horizon = mapping.get(interval, 10)
        price_df = fetch_price_data(yf_ticker, years=7 if horizon>365 else 5)
        if price_df is None or price_df.shape[0]<80:
            st.error('Not enough price data for ' + yf_ticker); st.stop()
        profile = fetch_fmp_profile(ticker if '.' not in ticker else ticker, fmp_key) if fmp_key else None
        yf_info = fetch_yf_info(yf_ticker)
        description = None; company_name = None
        if profile:
            description = profile.get('description') or profile.get('companyName') or None
            company_name = profile.get('companyName') or profile.get('name') or None
        if not description and yf_info:
            description = yf_info.get('longBusinessSummary') or yf_info.get('longName') or None
            company_name = yf_info.get('shortName') or yf_info.get('longName') or company_name
        feats = engineer_features(price_df)
        X, y, df_full = prepare_data(feats, horizon)
        current_price = float(feats['Close'].iloc[-1])
        # Try XGBoost if available and user prefers
        xgb_model = None; xgb_path = None; predicted_price = None; used_model = 'RF+LR ensemble'
        if XGBOOST_AVAILABLE and use_xgb:
            xgb_model, xgb_path = load_xgb_model_for_ticker(yf_ticker.replace('.NS','').upper())
            if xgb_model is not None:
                try:
                    preds_xgb = xgb_model.predict(xgb.DMatrix(X.values)) if hasattr(xgb_model, 'predict') else xgb_predict_from_df(xgb_model, X)
                    predicted_price = float(preds_xgb[-1]) if len(preds_xgb)>0 else float(preds_xgb)
                    used_model = f'XGBoost ({os.path.basename(xgb_path)})'
                except Exception:
                    xgb_model = None
        # Fallback to RF+LR
        if predicted_price is None:
            rf = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42); lr = LinearRegression()
            rf.fit(X, y); lr.fit(X, y)
            pred_rf = rf.predict(X.iloc[[-1]])[0]; pred_lr = lr.predict(X.iloc[[-1]])[0]
            predicted_price = float((pred_rf + pred_lr)/2)
            used_model = 'RF+LR ensemble'
        implied_return = (predicted_price/current_price-1)*100
        # backtest quick diagnostics
        mape, diracc = quick_backtest(X, y)
        # confidence heuristic (simple)
        confidence = float(max(0, min(100, (diracc/100*0.6 + (100 - min(mape,200))/100*0.3 + 0.1*1)*100)))
        # momentum and fundamentals
        mom_label, ret30, ret90, rsi = momentum_label(price_df)
        fund_score = compute_fundamentals_score(profile, yf_info=yf_info)
        decision, rec_text = simple_recommendation(current_price, predicted_price, implied_return, mom_label, fund_score if fund_score is not None else 50, confidence, interval)
        # SHAP explain (if requested and available)
        shap_fig = None; shap_values = None
        if SHAP_AVAILABLE and xgb_model is not None and show_shap:
            try:
                shap_fig, shap_values, explainer = explain_tree_model(xgb_model, X.tail(100), max_display=10, plot_type='bar')
            except Exception:
                shap_fig = None
        time.sleep(0.25)
    # Output
    col1, col2 = st.columns([1,1])
    with col1:
        st.metric("Current price (₹)", f"{current_price:,.2f}")
        st.metric("Predicted price (₹)", f"{predicted_price:,.2f}", delta=f"{implied_return:.2f}%")
    with col2:
        st.metric("Model / Source", used_model)
        st.metric("Model confidence", f"{confidence:.1f}%")
    st.markdown("### Company description")
    if company_name: st.markdown(f"**{company_name}**")
    if description: st.caption(description)
    else: st.caption("No description available from FMP or yfinance for this ticker.")
    st.markdown("### Momentum & Diagnostics")
    st.write(f"- Momentum: **{mom_label}** (30d: {ret30*100:.2f}%, 90d: {ret90*100:.2f}%, RSI: {rsi:.1f})")
    st.write(f"- Quick backtest median MAPE: **{mape:.2f}%** — Directional accuracy: **{diracc:.1f}%**")
    st.markdown("### Recommendation (Beginner-friendly)")
    st.info(rec_text)
    if shap_fig is not None:
        st.markdown("### SHAP feature importance (XGBoost)")
        st.pyplot(shap_fig)
    if run_backtest:
        st.markdown("### Full walk-forward backtest (sample)")
        st.write("Running a more detailed walk-forward backtest... this may take longer.")
        from backtest.walkforward import walk_forward_evaluate
        def train_func(Xt, yt, kwargs):
            # training using xgboost if available, else rf
            if XGBOOST_AVAILABLE:
                import xgboost as xgb
                dtrain = xgb.DMatrix(Xt.values, label=yt.values)
                params = {'objective':'reg:squarederror','eta':0.05,'max_depth':6,'subsample':0.8,'colsample_bytree':0.8}
                bst = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
                return bst
            else:
                rf = RandomForestRegressor(n_estimators=100, max_depth=6)
                rf.fit(Xt, yt); return rf
        df_metrics, summary, models = walk_forward_evaluate(X, y, n_splits=4, train_model_fn=train_func, model_kwargs={})
        st.write(df_metrics)
        st.write("Summary:", summary)
    st.markdown("---")
    st.caption("Stocks On V2.0 — Educational demo. Not financial advice.")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="card small">Enter inputs and tap Analyze. V2.0 supports optional XGBoost & SHAP. For heavy model training, run offline.</div>', unsafe_allow_html=True)
