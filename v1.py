# 直接複製這整份，存成 app.py 部署即可！
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="專業級股票監控", layout="wide")
st.title("專業級多股票監控（互動K線 + 買賣訊號 + 爆量 + 多時框）")

ALERT_LOG = {}

def send_telegram(text):
    if "telegram_token" not in st.secrets:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{st.secrets['telegram_token']}/sendMessage",
                      json={"chat_id": st.secrets["telegram_chat_id"], "text": text, "parse_mode": "HTML"}, timeout=10)
    except: pass

def send_once(key, text, cooldown=3600):
    now = time.time()
    if ALERT_LOG.get(key, 0) + cooldown < now:
        send_telegram(f"<b>強勢訊號</b>\n{text}")
        ALERT_LOG[key] = now

# 最穩 SuperTrend（向量版，永不爆炸）
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift(1))
    tr2 = abs(low - close.shift(1))
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    trend = pd.Series(0, index=df.index)
    supertrend_line = pd.Series(np.nan, index=df.index)
    for i in range(period, len(df)):
        if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
        if close.iloc[i] > final_upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
        supertrend_line.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
    df['SuperTrend'] = supertrend_line
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
    return df["Volume"].iloc[-1] > df["Volume"].iloc[-window:-1].mean() * threshold

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
        return df if not df.empty else None
    except: return None

def plot_candlestick_chart(df, symbol):
    df = df.tail(200).copy()
    df['Buy_Signal'] = (df['ST_Direction'].shift(1) == -1) & (df['ST_Direction'] == 1)
    df['Sell_Signal'] = (df['ST_Direction'].shift(1) == 1) & (df['ST_Direction'] == -1)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                                 name="K線", increasing_line_color='red', decreasing_line_color='green'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], name='SuperTrend', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='gold')))
    
    buy = df[df['Buy_Signal']]
    sell = df[df['Sell_Signal']]
    fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.985, mode='markers', name='買入',
                             marker=dict(symbol='triangle-up', size=16, color='lime')))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.015, mode='markers', name='賣出',
                             marker=dict(symbol='triangle-down', size=16, color='red')))
    
    colors = ['red' if o >= c else 'green' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors, yaxis='y2'))

    fig.update_layout(title=f"{symbol} 專業走勢圖", height=650, template="plotly_dark",
                      yaxis2=dict(title="成交量", overlaying='y', side='right'), legend=dict(x=0, y=1.1))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def trigger_alerts(symbol):
    df = get_data(symbol, "5d", "5m")
    if df is not None and len(df) > 60:
        df = supertrend(df)
        if detect_volume_spike(df, threshold=2.5) and df["ST_Direction"].iloc[-1] == 1:
            send_once(f"{symbol}_vol", f"{symbol}\n爆量 2.5 倍 + SuperTrend 多頭\n極強起漲！", 86400)

col1, col2 = st.columns([3,1])
with col1:
    symbols_input = st.text_input("股票代號", "AAPL,TSLA,NVDA,2330.TW,BTC-USD")
with col2:
    interval = st.selectbox("時間框", ["5m","15m","1h","1d"], index=1)

period = st.selectbox("期間", ["5d","10d","1mo"], index=0)

refresh = st.selectbox("刷新", ["關閉","30秒","1分鐘"], index=1)
if refresh != "關閉":
    sec = {"30秒":30,"1分鐘":60}[refresh]
    st.write(f"自動刷新中...")
    time.sleep(sec)
    st.rerun()

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

for symbol in symbols:
    with st.container():
        c1, c2 = st.columns([1,3])
        with c1:
            st.subheader(symbol)
            df = get_data(symbol, period, interval)
            if not df.empty:
                df = add_macd(df)
                df = add_vwap(df)
                df = supertrend(df)
                vol = "爆量" if detect_volume_spike(df) else "正常"
                st.write(f"趨勢：{'上升' if df['MACD'].iloc[-1]>df['Signal'].iloc[-1] else '下降'} | 成交量：{vol}")
        with c2:
            if not df.empty:
                fig = plot_candlestick_chart(df, symbol)
                st.plotly_chart(fig, use_container_width=True)
        trigger_alerts(symbol)

st.success("專業監控完成")
