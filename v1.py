# app.py  ← 終極專業版（互動K線圖 + 買賣訊號標記 + 爆量 + 多時框共振）
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="專業級股票監控 Pro", layout="wide")
st.title("專業級多股票監控（互動K線 + 買賣訊號 + 爆量 + 多時框共振）")

# ============================ Telegram ============================
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
        send_telegram(f"<b>強勢訊號</b>\n{text}")
        ALERT_LOG[key] = now

# ============================ 指標函數（同前）========================
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

def add_vwap(df):
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df["PV"].cumsum()
    df["CumVol"] = df["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df

def detect_volume_spike(df, window=20, threshold=2.0):
    if len(df) < window + 1: return False
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

# ============================ 互動式專業K線圖 ============================
def plot_candlestick_chart(df, symbol):
    df = df.tail(200).copy()  # 只顯示最近200根
    
    # 買賣訊號標記
    df['Buy_Signal'] = (df['ST_Direction'].shift(1) == -1) & (df['ST_Direction'] == 1)
    df['Sell_Signal'] = (df['ST_Direction'].shift(1) == 1) & (df['ST_Direction'] == -1)
    
    fig = go.Figure()

    # K線
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="K線", increasing_line_color='red', decreasing_line_color='green'
    ))

    # SuperTrend
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SuperTrend'], mode='lines',
        name='SuperTrend', line=dict(width=2, color='purple')
    ))

    # VWAP
    fig.add_trace(go.Scatter(
        x=df.index, y=df['VWAP'], mode='lines',
        name='VWAP', line=dict(width=1.5, color='orange', dash='dot')
    ))

    # MA20 & MA50
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50", line=dict(color="gold")))

    # 買賣訊號
    buy_signals = df[df['Buy_Signal']]
    sell_signals = df[df['Sell_Signal']]
    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['Low'] * 0.985,
        mode='markers', name='買入訊號',
        marker=dict(symbol='triangle-up', size=16, color='lime', line=dict(width=2))
    ))
    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['High'] * 1.015,
        mode='markers', name='賣出訊號',
        marker=dict(symbol='triangle-down', size=16, color='red', line=dict(width=2))
    ))

    # 成交量
    colors = ['red' if o >= c else 'green' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors, yaxis='y2'))

    fig.update_layout(
        title=f"{symbol} 專業走勢圖",
        xaxis_title="時間", yaxis_title="價格",
        yaxis2=dict(title="成交量", overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=650,
        template="plotly_dark"
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig

# ============================ 進階警報 ============================
def trigger_advanced_alerts(symbol):
    df_5m = get_data(symbol, "5d", "5m")
    if df_5m is not None and len(df_5m) > 60:
        df_5m = add_macd(df_5m)
        df_5m = supertrend(df_5m)
        if detect_volume_spike(df_5m, threshold=2.5) and df_5m["ST_Direction"].iloc[-1] == 1:
            send_once(f"{symbol}_vol", f"{symbol}\n成交量爆量 + SuperTrend 多頭\n極強起漲！", 86400)

    # 多時框共振（5m + 15m + 1h 同時翻多）
    flips = 0
    for intv in ["5m", "15m", "1h"]:
        df = get_data(symbol, "5d", intv)
        if df is not None and len(df) > 60:
            df = supertrend(df)
            if df["ST_Direction"].iloc[-2] == -1 and df["ST_Direction"].iloc[-1] == 1:
                flips += 1
    if flips >= 2:
        send_once(f"{symbol}_tf", f"{symbol}\n多時框共振！{flips}個時間框同時翻多\n強勢啟動！", 86400*2)

# ============================ UI ============================
col1, col2 = st.columns([3, 1])
with col1:
    symbols_input = st.text_input("股票代號", value="AAPL,TSLA,NVDA,2330.TW,BTC-USD")
with col2:
    interval = st.selectbox("時間框", ["5m","15m","1h","1d"], index=1)

period = st.selectbox("期間", ["5d","10d","1mo","3mo"], index=0)

refresh = st.selectbox("自動刷新", ["關閉","30秒","1分鐘"], index=1)
if refresh != "關閉":
    sec = {"30秒":30, "1分鐘":60}[refresh]
    st.write(f"自動刷新中... {refresh}")
    time.sleep(sec)
    st.rerun()

st.info(f"更新時間：{datetime.now().strftime('%H:%M:%S')}")

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

for symbol in symbols:
    with st.container():
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.subheader(f"{symbol}")
            df = get_data(symbol, period, interval)
            if df is None or df.empty:
                st.error("無資料")
                continue
            df = add_macd(df)
            df = add_vwap(df)
            df = supertrend(df)
            df["MA20"] = df["Close"].rolling(20).mean()
            df["MA50"] = df["Close"].rolling(50).mean()

            trend = "上升" if df["MACD"].iloc[-1] > df["Signal"].iloc[-1] else "下降"
            strength = "強勢" if len(df) > 14 and df["Close"].rolling(14).std().iloc[-1] * 10 > df["Close"].iloc[-1] * 0.02 else "弱勢"
            vol_status = "爆量" if detect_volume_spike(df) else "正常"
            st.write(f"**趨勢**：{trend} | **強度**：{strength} | **成交量**：{vol_status}")

        with col_b:
            fig = plot_candlestick_chart(df, symbol)
            st.plotly_chart(fig, use_container_width=True)

        trigger_advanced_alerts(symbol)

st.success("專業監控完成！所有買賣訊號已標記在圖表上")
