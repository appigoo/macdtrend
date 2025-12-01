# app.py  ← 完整進階版（已包含 成交量爆量 + 多時框共振）
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import time

st.set_page_config(page_title="多股票趨勢監控 Pro", layout="wide")
st.title("多股票趨勢監控 Pro（爆量 + 多時框共振 + Telegram）")

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
        send_telegram(f"<b>【強勢訊號】</b>\n{text}")
        ALERT_LOG[key] = now

# ============================ 超穩定指標函數（同前）========================
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
    df["RSI"] = 100 - (100 / (1 + rs))
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

# 新增：成交量爆量偵測
def detect_volume_spike(df, window=20, threshold=2.0):
    if len(df) < window + 1:
        return False
    avg_vol = df["Volume"].iloc[-window:-1].mean()
    current_vol = df["Volume"].iloc[-1]
    return current_vol > avg_vol * threshold

@st.cache_data(ttl=60, show_spinner=False)
def get_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True, threads=False)
        return df if not df.empty else None
    except:
        return None

# ============================ 進階警報（爆量 + 多時框共振）========================
def trigger_advanced_alerts(symbol):
    intervals = ["5m", "15m", "1h"]
    directions = []
    
    for interval in intervals:
        df = get_data(symbol, period="5d", interval=interval)
        if df is None or len(df) < 60:
            return
        df = add_macd(df)
        df = add_adx(df)
        df = supertrend(df)
        
        # 判斷當前趨勢
        current_dir = df["ST_Direction"].iloc[-1]
        prev_dir = df["ST_Direction"].iloc[-2]
        directions.append((current_dir, prev_dir, interval))
    
    # 1. 爆量警報（用 5 分鐘資料）
    df_5m = get_data(symbol, "5d", "5m")
    if df_5m is not None and len(df_5m) > 20:
        df_5m = add_macd(df_5m)
        df_5m = supertrend(df_5m)
        if detect_volume_spike(df_5m, window=20, threshold=2.5):
            if df_5m["ST_Direction"].iloc[-1] == 1:
                send_once(f"{symbol}_vol_up", f"{symbol}\n成交量爆量 2.5 倍以上\n同時 SuperTrend 多頭\n極強起漲訊號！", 86400)

    # 2. 多時框共振（三個時框同時翻多）
    up_count = sum(1 for cur, prev, intv in directions if cur == 1 and prev == -1)
    if up_count >= 2:  # 至少兩個時框同時翻多
        msg = f"{symbol}\n多時框共振啟動！\n"
        for cur, prev, intv in directions:
            if cur == 1 and prev == -1:
                msg += f"{intv} 翻多\n"
        send_once(f"{symbol}_multi_tf", msg + "強勢多頭共振", 86400*3)

# ============================ UI ============================
col1, col2 = st.columns([3, 1])
with col1:
    symbols_input = st.text_input("股票代號", "NIO,TSLA,NVDA,XPEV,GOOGL,META")
with col2:
    interval = st.selectbox("主要時間框", ["5m","15m","1h","1d"], index=1)

period = st.selectbox("期間", ["5d","10d","1mo","3mo"], index=0)

refresh = st.selectbox("自動刷新", ["關閉", "30秒", "1分鐘"], index=1)
if refresh != "關閉":
    sec = {"30秒":30, "1分鐘":60}[refresh]
    st.write(f"自動刷新倒數 {refresh}...")
    time.sleep(sec)
    st.rerun()

st.info(f"更新時間：{datetime.now().strftime('%H:%M:%S')}")

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

for symbol in symbols:
    with st.expander(f"{symbol}", expanded=True):
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
        vol_status = "爆量" if detect_volume_spike(df) else "正常"
        
        st.write(f"趨勢：{trend} | 強度：{strength} | 成交量：{vol_status} | RSI {df['RSI'].iloc[-1]:.1f}")

        st.line_chart(df[["Close","MA20","MA50","VWAP","SuperTrend"]].tail(200))

        # 觸發進階警報（每支股票獨立觸發）
        trigger_advanced_alerts(symbol)

st.success("Pro 版監控完成！爆量與共振訊號已啟動")
