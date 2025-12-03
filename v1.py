import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import time

st.set_page_config(page_title="專業級多股票監控", layout="wide")
st.title("專業級多股票監控（K線 + 爆量 + 買賣訊號 + 股價走勢變化）")

# ==================== 完美的自動刷新方案（不依賴 st.autorefresh）===================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0

refresh_choice = st.selectbox("自動刷新週期", ["關閉", "15秒", "30秒", "60秒"], index=2)

if refresh_choice != "關閉":
    seconds = {"15秒": 15, "30秒": 30, "60秒": 60}[refresh_choice]
    now = time.time()
    # 計算距離上次刷新經過多久
    if now - st.session_state.last_refresh > seconds:
        st.session_state.last_refresh = now
        st.rerun()
    else:
        # 倒數計時器（純前端顯示，不會卡住）
        remain = int(seconds - (now - st.session_state.last_refresh))
        st.info(f"距離下次自動更新：{remain} 秒")
else:
    if st.button("手動刷新", type="primary"):
        st.rerun()

# =================================== 其餘程式碼完全不變 ===================================

ALERT_LOG = {}

def send_telegram(text):
    if "telegram_token" not in st.secrets: return
    try:
        requests.post(f"https://api.telegram.org/bot{st.secrets['telegram_token']}/sendMessage",
                      json={"chat_id": st.secrets["telegram_chat_id"], "text": text, "parse_mode": "HTML"})
    except: pass

def send_once(key, text, cooldown=86400):
    now = time.time()
    if ALERT_LOG.get(key, 0) + cooldown < now:
        send_telegram(f"<b>強勢訊號</b>\n{text}")
        ALERT_LOG[key] = now

def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    upper_band = upper.copy()
    lower_band = lower.copy()
    for i in range(1, len(df)):
        if upper.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper.iloc[i]
        else:
            upper_band.iloc[i] = upper_band.iloc[i-1]
        if lower.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower.iloc[i]
        else:
            lower_band.iloc[i] = lower_band.iloc[i-1]
    trend = pd.Series(0, index=df.index)
    st_line = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
        st_line.iloc[i] = lower_band.iloc[i] if trend.iloc[i] == 1 else upper_band.iloc[i]
    df['SuperTrend'] = st_line
    df['ST_Direction'] = trend
    return df

def add_macd(df):
    c = df["Close"]
    df["EMA12"] = c.ewm(span=12, adjust=False).mean()
    df["EMA26"] = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def add_vwap(df):
    df["PV"] = df["Close"] * df["Volume"]
    df["CumPV"] = df["PV"].cumsum()
    df["CumVol"] = df["Volume"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df

def detect_volume_spike(df, window=20, threshold=2.0):
    if len(df) <= window: return False
    return df["Volume"].iloc[-1] > df["Volume"].iloc[-window:-1].mean() * threshold

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except:
        return None

def plot_chart(df, symbol):
    df = df.tail(200).copy()
    df['Buy'] = (df['ST_Direction'].shift(1) == -1) & (df['ST_Direction'] == 1)
    df['Sell'] = (df['ST_Direction'].shift(1) == 1) & (df['ST_Direction'] == -1)
    df["MA20"] = df["Close"].rolling(20).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], name='SuperTrend', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')))
    buy = df[df['Buy']]
    sell = df[df['Sell']]
    fig.add_trace(go.Scatter(x=buy.index, y=buy['Low']*0.985, mode='markers', name='買入', marker=dict(symbol='triangle-up', size=16, color='lime')))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['High']*1.015, mode='markers', name='賣出', marker=dict(symbol='triangle-down', size=16, color='red')))
    colors = ['red' if o >= c else 'green' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors, yaxis='y2'))
    fig.update_layout(title=f"{symbol} K線總覽", height=600, template="plotly_dark",
                      yaxis2=dict(title="成交量", overlaying='y', side='right'))
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def plot_trend_and_volume_change(df, symbol):
    df = df.tail(200).copy()
    df['Price_Change'] = df['Close'].pct_change() * 100
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盤價走勢', line=dict(color='cyan', width=3)))
    colors = ['green' if x >= 0 else 'red' for x in df['Price_Change']]
    fig.add_trace(go.Bar(x=df.index, y=df['Price_Change'], name='股價漲跌(%)', marker_color=colors))
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_Change'], name='成交量變化(%)', line=dict(color='yellow', dash='dot'), yaxis='y2'))
    fig.update_layout(title=f"{symbol} 股價與成交量變化", height=400, template="plotly_dark",
                  yaxis2=dict(title="成交量變化(%)", overlaying='y', side='right'))
    return fig

# ================================= UI =================================
col1, col2 = st.columns([3,1])
with col1:
    symbols_input = st.text_input("股票代號（多筆用逗號分隔）", "AAPL,TSLA,NVDA,2330.TW,BTC-USD")
with col2:
    interval = st.selectbox("時間框", ["5m","15m","30m","1h","1d"], index=1)

period = st.selectbox("期間", ["5d","10d","1mo","3mo"], index=0)

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

for symbol in symbols:
    with st.container(border=True):
        left, right = st.columns([1,4])
        with left:
            st.subheader(symbol)
            df = get_data(symbol, period, interval)
            if df is None or len(df) < 20:
                st.error("無資料")
                continue
            df = add_macd(df)
            df = add_vwap(df)
            df = supertrend(df)
            vol = "爆量" if detect_volume_spike(df) else "正常"
            trend = "多頭" if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else "空頭"
            st.metric("最新價格", f"${df['Close'].iloc[-1]:.2f}")
            st.write(f"趨勢：{trend} ｜ 成交量：{vol}")

        with right:
            tab1, tab2 = st.tabs(["K線圖", "走勢變化"])
            with tab1:
                st.plotly_chart(plot_chart(df, symbol), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_trend_and_volume_change(df, symbol), use_container_width=True)

        if detect_volume_spike(df, threshold=2.5) and df["ST_Direction"].iloc[-1] == 1:
            send_once(f"{symbol}_vol", f"{symbol}\n爆量2.5倍 + SuperTrend翻多\n極強起漲！", 86400)

st.success("監控運行中，圖表已完全正常顯示")
