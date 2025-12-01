# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import time

# ============================
# 頁面設定
# ============================
st.set_page_config(page_title="多股票趨勢監控 + Telegram 即時警報", layout="wide")
st.title("多股票趨勢監控 + Telegram 即時警報")

# ============================
# Telegram 推播 + 防洗版機制
# ============================
ALERT_LOG = {}   # key: "symbol_類型" → 最後發送時間

def send_telegram(text):
    if "telegram_token" not in st.secrets:
        return
    token = st.secrets["telegram_token"]
    chat_id = st.secrets["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def send_once(key: str, text: str, cooldown: int = 3600):
    """cooldown 秒內同樣訊號不重複發送"""
    now = time.time()
    if ALERT_LOG.get(key, 0) + cooldown < now:
        send_telegram(text)
        ALERT_LOG[key] = now

# ============================
# 穩定版 SuperTrend（永不爆炸）
# ============================
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    high, low, close = df['High'], df['Low'], df['Close']

    # 正確 ATR
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift(1))
    tr2 = abs(low - close.shift(1))
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # 基本上下軌
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    # 最終軌道（趨勢持續性）
    final_upper = upper.copy()
    final_lower = lower.copy()
    trend = pd.Series(0, index=df.index)
    st_line = pd.Series(np.nan, index=df.index)

    for i in range(period, len(df)):
        # Upper band 調整
        final_upper.iloc[i] = upper.iloc[i] if (upper.iloc[i] < final_upper.iloc[i-1] or 
                                                close.iloc[i-1] > final_upper.iloc[i-1]) else final_upper.iloc[i-1]
        # Lower band 調整
        final_lower.iloc[i] = lower.iloc[i] if (lower.iloc[i] > final_lower.iloc[i-1] or 
                                                close.iloc[i-1] < final_lower.iloc[i-1]) else final_lower.iloc[i-1]

        # 趨勢判斷
        if close.iloc[i] > final_upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

        # SuperTrend 線值
        st_line.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]

    df['ATR'] = atr
    df['SuperTrend'] = st_line
    df['ST_Direction'] = trend          # 1=多頭, -1=空頭, 0=初期無訊號
    return df

# ============================
# 其他指標
# ============================
def add_macd(df):
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
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
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df["ADX"] = dx.ewm(alpha=1/period).mean()
    return df

def add_vwap(df):
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df["PV"].cumsum()
    df["CumVol"] = df["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df

# ============================
# 資料快取
# ============================
@st.cache_data(ttl=60, show_spinner=False)
def get_data(symbol: str, period: str, interval: str):
    try:
        data = yf.download(symbol, period=period, interval=interval,
                           progress=False, auto_adjust=True)
        if data.empty:
            return None
        return data
    except:
        return None

# ============================
# 警報集合
# ============================
def trigger_alerts(df, symbol):
    if len(df) < 60:
        return

    # 1. MACD Hist 三連升 + ADX > 25
    if (df["Hist"].iloc[-3] < df["Hist"].iloc[-2] < df["Hist"].iloc[-1] and
        df["ADX"].iloc[-1] > 25):
        send_once(f"{symbol}_macd3", f"{symbol}\nMACD Hist 連3上升 + ADX強勢\n可能強勢啟動！", 7200)

    # 2. SuperTrend 翻多 / 翻空
    if df["ST_Direction"].iloc[-2] == -1 and df["ST_Direction"].iloc[-1] == 1:
        send_once(f"{symbol}_st_up", f"{symbol}\nSuperTrend 翻多（看漲）", 86400)
    if df["ST_Direction"].iloc[-2] == 1 and df["ST_Direction"].iloc[-1] == -1:
        send_once(f"{symbol}_st_down", f"{symbol}\nSuperTrend 翻空（看跌）", 86400)

    # 3. MA20/50 金叉死叉
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()
    if ma20.iloc[-2] <= ma50.iloc[-2] and ma20.iloc[-1] > ma50.iloc[-1]:
        send_once(f"{symbol}_golden", f"{symbol}\nMA20 / MA50 金叉", 86400*3)
    if ma20.iloc[-2] >= ma50.iloc[-2] and ma20.iloc[-1] < ma50.iloc[-1]:
        send_once(f"{symbol}_death", f"{symbol}\nMA20 / MA50 死叉", 86400*3)

    # 4. VWAP 突破
    if df["Close"].iloc[-2] <= df["VWAP"].iloc[-2] and df["Close"].iloc[-1] > df["VWAP"].iloc[-1]:
        send_once(f"{symbol}_vwap_up", f"{symbol}\n收盤價突破 VWAP → 看多", 3600)
    if df["Close"].iloc[-2] >= df["VWAP"].iloc[-2] and df["Close"].iloc[-1] < df["VWAP"].iloc[-1]:
        send_once(f"{symbol}_vwap_down", f"{symbol}\n收盤價跌破 VWAP → 看空", 3600)

# ============================
# UI 設定
# ============================
col1, col2 = st.columns([3, 1])
with col1:
    symbols_input = st.text_input("股票代號（逗號分隔）",
                                  value="AAPL,TSLA,NVDA,2330.TW,0050.TW")
with col2:
    interval = st.selectbox("K線週期", ["5m","15m","30m","1h","1d"], index=1)

period = st.selectbox("回看期間", ["5d","10d","1mo","3mo","6mo","1y","2y"], index=2)

refresh_opt = st.selectbox("自動刷新", ["關閉","30秒","1分鐘","2分鐘","5分鐘"])
refresh_map = {"30秒":30, "1分鐘":60, "2分鐘":120, "5分鐘":300}
if refresh_opt != "關閉":
    time.sleep(refresh_map[refresh_opt])
    st.experimental_rerun()

st.info(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# ============================
# 主迴圈
# ============================
for symbol in symbols:
    with st.container():
        st.subheader(f"{symbol}")

        df = get_data(symbol, period, interval)
        if df is None or df.empty:
            st.error(f"無法取得 {symbol} 資料")
            continue

        # 計算所有指標
        df = add_macd(df)
        df = add_rsi(df)
        df = add_adx(df)
        df = add_vwap(df)
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df = supertrend(df, period=10, multiplier=3)

        # 趨勢總覽
        direction = "上升" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "下降"
        strength = "強勢" if df["ADX"].iloc[-1] > 25 else "弱勢"
        rsi_val = df["RSI"].iloc[-1]
        st.write(f"**趨勢**：{direction} | **強度**：{strength} | RSI：{rsi_val:.1f}")

        # 圖表
        chart_df = df[["Close","MA20","MA50","VWAP","SuperTrend"]].copy()
        chart_df.rename(columns={"SuperTrend": "SuperTrend"}, inplace=True)
        st.line_chart(chart_df.tail(200))

        # 觸發警報
        trigger_alerts(df, symbol)

st.success("所有股票監控完成！")
