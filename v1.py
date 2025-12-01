# app.py（最終無敵穩定版）
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import time

st.set_page_config(page_title="多股票趨勢監控 + Telegram 即時警報", layout="wide")
st.title("多股票趨勢監控 + Telegram 即時警報")

# ============================
# Telegram 防洗版
# ============================
ALERT_LOG = {}

def send_telegram(text):
    if "telegram_token" not in st.secrets:
        return
    token = st.secrets["telegram_token"]
    chat_id = st.secrets["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
    except:
        pass

def send_once(key: str, text: str, cooldown: int = 3600):
    now = time.time()
    if ALERT_LOG.get(key, 0) + cooldown < now:
        send_telegram(f"{text}")
        ALERT_LOG[key] = now

# ============================
# 超穩定 SuperTrend（已修）
# ============================
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    high, low, close = df['High'].values, df['Low'].values, df['Close'].values

    tr0 = abs(high - low)
    tr1 = abs(high - np.roll(close, 1))
    tr2 = abs(low - np.roll(close, 1))
    tr = pd.DataFrame({'tr0': tr0, 'tr1': tr1, 'tr2': tr2}).max(axis=1)
    tr.iloc[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(period).mean()

    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = upper.copy()
    final_lower = lower.copy()
    trend = np.zeros(len(df))
    st_line = np.full(len(df), np.nan)

    for i in range(period, len(df)):
        final_upper[i] = upper[i] if (upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = lower[i] if (lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]

        if close[i] > final_upper[i-1]:
            trend[i] = 1
        elif close[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1] if i > 0 else 0

        st_line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    df['ATR'] = atr
    df['SuperTrend'] = st_line
    df['ST_Direction'] = trend
    return df

# ============================
# 其他指標（MACD / RSI / VWAP）
# ============================
def add_macd(df):
    close = df["Close"]
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    return df

def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_vwap(df):
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df["PV"].cumsum()
    df["CumVol"] = df["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df

# ============================
# 終極穩定 ADX（重點！）
# ============================
def add_adx(df, period=14):
    df = df.copy()
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ])
    tr[0] = high[0] - low[0]
    tr = pd.Series(tr, index=df.index)
    atr = tr.rolling(period).mean()

    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    df['ADX'] = adx
    return df

# ============================
# 資料快取
# ============================
@st.cache_data(ttl=60, show_spinner=False)
def get_data(symbol: str, period: str, interval: str):
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=False)
        if df.empty:
            return None
        return df
    except:
        return None

# ============================
# 警報
# ============================
def trigger_alerts(df, symbol):
    if len(df) < 60: return
    try:
        # SuperTrend 翻轉
        if df["ST_Direction"].iloc[-2] == -1 and df["ST_Direction"].iloc[-1] == 1:
            send_once(f"{symbol}_st_up", f"{symbol}\nSuperTrend 翻多", 86400)
        if df["ST_Direction"].iloc[-2] == 1 and df["ST_Direction"].iloc[-1] == -1:
            send_once(f"{symbol}_st_down", f"{symbol}\nSuperTrend 翻空", 86400)

        # MACD Hist 三連升 + ADX強勢
        if (df["Hist"].iloc[-3] < df["Hist"].iloc[-2] < df["Hist"].iloc[-1] and df["ADX"].iloc[-1] > 25):
            send_once(f"{symbol}_macd3", f"{symbol}\nMACD Hist 連3升 + ADX強勢", 7200)
    except:
        pass

# ============================
# UI 與主迴圈
# ============================
col1, col2 = st.columns([3,1])
with col1:
    symbols_input = st.text_input("股票代號", "AAPL,TSLA,NVDA,2330.TW")
with col2:
    interval = st.selectbox("週期", ["5m","15m","1h","1d"], index=1)

period = st.selectbox("期間", ["5d","10d","1mo","3mo","1y"], index=2)

refresh = st.selectbox("刷新", ["關閉","30秒","1分鐘","2分鐘"], index=2)
if refresh != "關閉":
    time.sleep({"30秒":30,"1分鐘":60,"2分鐘":120}[refresh])
    st.experimental_rerun()

st.info(f"更新時間：{datetime.now().strftime('%H:%M:%S')}")

for symbol in [s.strip().upper() for s in symbols_input.split(",") if s.strip()]:
    with st.container():
        st.subheader(symbol)
        df = get_data(symbol, period, interval)
        if df is None or df.empty:
            st.error("無資料")
            continue

        df = add_macd(df)
        df = add_rsi(df)
        df = add_adx(df)
        df = add_vwap(df)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df = supertrend(df)

        direction = "上升" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "下降"
        strength = "強勢" if df["ADX"].iloc[-1] > 25 else "弱勢"
        st.write(f"趨勢：{direction} | 強度：{strength} | RSI {df['RSI'].iloc[-1]:.1f}")

        st.line_chart(df[["Close","MA20","MA50","VWAP","SuperTrend"]].tail(200))
        trigger_alerts(df, symbol)

st.success("監控完成")
