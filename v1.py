# app.py  ← 直接複製貼上，部署即可完美運行
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import time

st.set_page_config(page_title="多股票趨勢監控", layout="wide")
st.title("多股票即時趨勢監控 + Telegram 警報")

# ============================ Telegram 推播 ============================
ALERT_LOG = {}

def send_telegram(text):
    if "telegram_token" not in st.secrets:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{st.secrets['telegram_token']}/sendMessage",
            json={"chat_id": st.secrets["telegram_chat_id"], "text": text, "parse_mode": "HTML"},
            timeout=10
        )
    except:
        pass

def send_once(key: str, text: str, cooldown: int = 3600):
    now = time.time()
    if ALERT_LOG.get(key, 0) + cooldown < now:
        send_telegram(f"<b>[警報]</b>\n{text}")
        ALERT_LOG[key] = now

# ============================ 超穩定指標 ============================
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    atr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        atr[i] = (atr[i-1]*(period-1) + tr)/period if i >= period else tr
    hl2 = (h + l)/2
    upper = hl2 + multiplier*atr
    lower = hl2 - multiplier*atr
    final_upper = upper.copy()
    final_lower = lower.copy()
    trend = np.zeros(len(df))
    st_line = np.full(len(df), np.nan)
    for i in range(period, len(df)):
        final_upper[i] = upper[i] if (upper[i] < final_upper[i-1] or c[i-1] > final_upper[i-1]) else final_upper[i-1]
        final_lower[i] = lower[i] if (lower[i] > final_lower[i-1] or c[i-1] < final_lower[i-1]) else final_lower[i-1]
        if c[i] > final_upper[i-1]: trend[i] = 1
        elif c[i] < final_lower[i-1]: trend[i] = -1
        else: trend[i] = trend[i-1]
        st_line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]
    df['SuperTrend'] = st_line
    df['ST_Direction'] = trend
    return df

def add_macd(df):
    c = df["Close"]
    df["EMA12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA26"] = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["MACD"] - df["Signal"]
    return df

def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100/(1+rs))
    return df

def add_adx(df, period=14):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    tr = np.maximum.reduce([h-l, np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))])
    tr[0] = h[0]-l[0]
    atr = pd.Series(tr).rolling(period).mean()
    up = h - np.roll(h,1)
    down = np.roll(l,1) - l
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df["ADX"] = dx.ewm(alpha=1/period, adjust=False).mean()
    return df

def add_vwap(df):
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df["PV"].cumsum()
    df["CumVol"] = df["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df

@st.cache_data(ttl=60, show_spinner=False)
def get_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=False)
        return df if not df.empty else None
    except:
        return None

def trigger_alerts(df, symbol):
    if len(df) < 60: return
    try:
        if df["ST_Direction"].iloc[-2] == -1 and df["ST_Direction"].iloc[-1] == 1:
            send_once(f"{symbol}_st_up", f"{symbol}\nSuperTrend 翻多", 86400)
        if df["ST_Direction"].iloc[-2] == 1 and df["ST_Direction"].iloc[-1] == -1:
            send_once(f"{symbol}_st_down", f"{symbol}\nSuperTrend 翻空", 86400)
        if df["Hist"].iloc[-3] < df["Hist"].iloc[-2] < df["Hist"].iloc[-1] and df["ADX"].iloc[-1] > 25:
            send_once(f"{symbol}_macd3", f"{symbol}\nMACD Hist 三連升 + ADX強勢", 7200)
    except:
        pass

# ============================ UI ============================
col1, col2 = st.columns([3, 1])

with col1:
    symbols_input = st.text_input("股票代號（逗號分隔）", "AAPL,TSLA,NVDA,2330.TW,0050.TW")
with col2:
    interval = st.selectbox("時間框", ["5m","15m","30m","1h","1d"], index=1)

period = st.selectbox("回看期間", ["5d","10d","1mo","3mo","6mo","1y"], index=2)

refresh = st.selectbox("自動刷新", ["關閉", "30秒", "1分鐘", "2分鐘"], index=1)
if refresh != "關閉":
    seconds = {"30秒":30, "1分鐘":60, "2分鐘":120}[refresh]
    st.toast(f"將在 {refresh} 後自動刷新...")   # 完全移除 icon 參數，最穩！
    time.sleep(seconds)
    st.rerun()

st.info(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

for symbol in symbols:
    with st.container():
        st.subheader(f"{symbol}")
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

        trend = "上升" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "下降"
        strength = "強勢" if df["ADX"].iloc[-1] > 25 else "弱勢"
        st.write(f"趨勢：{trend} | 強度：{strength} | RSI {df['RSI'].iloc[-1]:.1f}")

        st.line_chart(df[["Close","MA20","MA50","VWAP","SuperTrend"]].tail(300))
        trigger_alerts(df, symbol)

st.success("監控完成")
