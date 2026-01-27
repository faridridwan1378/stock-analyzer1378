#!/usr/bin/env python3
"""
📈 Stock Scalping Analyzer - Public Version
Streamlit Cloud Compatible
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="📈 Stock Scalping Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .buy-signal { 
        background-color: #00c853; 
        color: white; 
        padding: 0.5rem 1rem; 
        border-radius: 5px;
        font-weight: bold;
    }
    .sell-signal { 
        background-color: #ff1744; 
        color: white; 
        padding: 0.5rem 1rem; 
        border-radius: 5px;
        font-weight: bold;
    }
    .neutral-signal { 
        background-color: #ffc107; 
        color: black; 
        padding: 0.5rem 1rem; 
        border-radius: 5px;
        font-weight: bold;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CURRENCY HELPER ====================
class CurrencyHelper:
    CURRENCY_SYMBOLS = {
        'IDR': 'Rp', 'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
        'SGD': 'S$', 'HKD': 'HK$', 'CNY': '¥', 'KRW': '₩', 'THB': '฿',
        'MYR': 'RM', 'PHP': '₱', 'INR': '₹', 'AUD': 'A$', 'CAD': 'C$',
    }
    NO_DECIMAL = ['IDR', 'JPY', 'KRW', 'VND']
    
    @classmethod
    def get_symbol(cls, currency: str) -> str:
        return cls.CURRENCY_SYMBOLS.get(currency.upper(), currency)
    
    @classmethod
    def format_price(cls, price: float, currency: str) -> str:
        symbol = cls.get_symbol(currency)
        if currency.upper() in cls.NO_DECIMAL:
            return f"{symbol} {price:,.0f}"
        return f"{symbol} {price:,.2f}"
    
    @classmethod
    def format_large(cls, number: float, currency: str) -> str:
        symbol = cls.get_symbol(currency)
        if abs(number) >= 1e12:
            return f"{symbol} {number/1e12:.2f}T"
        elif abs(number) >= 1e9:
            return f"{symbol} {number/1e9:.2f}B"
        elif abs(number) >= 1e6:
            return f"{symbol} {number/1e6:.2f}M"
        return f"{symbol} {number:,.0f}"

# ==================== DATA FUNCTIONS ====================
@st.cache_data(ttl=300)  # Cache 5 menit
def get_stock_data(ticker: str, period: str, interval: str):
    """Fetch stock data with caching"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        info = stock.info
        return data, info
    except Exception as e:
        return None, None

def get_currency(ticker: str, info: dict) -> str:
    """Get currency from stock info"""
    currency = info.get('currency', None)
    if currency:
        return currency
    if ticker.endswith('.JK'):
        return 'IDR'
    elif ticker.endswith('.SI'):
        return 'SGD'
    elif ticker.endswith('.HK'):
        return 'HKD'
    elif ticker.endswith('.T'):
        return 'JPY'
    elif ticker.endswith('.L'):
        return 'GBP'
    return 'USD'

# ==================== TECHNICAL ANALYSIS ====================
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal']
    
    # Bollinger Bands
    data['SMA20'] = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    data['BB_Upper'] = data['SMA20'] + (std * 2)
    data['BB_Lower'] = data['SMA20'] - (std * 2)
    
    # Moving Averages
    data['SMA5'] = close.rolling(window=5).mean()
    data['SMA10'] = close.rolling(window=10).mean()
    data['EMA9'] = close.ewm(span=9, adjust=False).mean()
    data['EMA21'] = close.ewm(span=21, adjust=False).mean()
    
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=14).mean()
    
    # Volume
    data['Volume_SMA'] = volume.rolling(window=20).mean()
    data['Volume_Ratio'] = volume / data['Volume_SMA']
    
    # Stochastic
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    data['Stoch_K'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
    
    # ADX
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    atr_temp = data['ATR'].copy()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_temp)
    minus_di = abs(100 * (minus_dm.rolling(window=14).mean() / atr_temp))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    data['ADX'] = dx.rolling(window=14).mean()
    
    return data

def calculate_technical_score(data: pd.DataFrame) -> Tuple[int, List[Tuple[str, str, str]]]:
    """Calculate technical score and signals"""
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    score = 0
    signals = []
    
    # RSI
    rsi = latest['RSI']
    if rsi < 30:
        score += 10
        signals.append(("RSI", f"Oversold ({rsi:.1f})", "bullish"))
    elif rsi < 40:
        score += 5
        signals.append(("RSI", f"Mendekati Oversold ({rsi:.1f})", "bullish"))
    elif rsi > 70:
        score -= 10
        signals.append(("RSI", f"Overbought ({rsi:.1f})", "bearish"))
    elif rsi > 60:
        score -= 3
        signals.append(("RSI", f"Tinggi ({rsi:.1f})", "bearish"))
    else:
        signals.append(("RSI", f"Netral ({rsi:.1f})", "neutral"))
    
    # MACD
    if prev['MACD'] < prev['Signal'] and latest['MACD'] > latest['Signal']:
        score += 10
        signals.append(("MACD", "Bullish Crossover ✨", "bullish"))
    elif prev['MACD'] > prev['Signal'] and latest['MACD'] < latest['Signal']:
        score -= 10
        signals.append(("MACD", "Bearish Crossover ⚠️", "bearish"))
    elif latest['MACD'] > latest['Signal']:
        score += 5
        signals.append(("MACD", "Bullish", "bullish"))
    else:
        score -= 5
        signals.append(("MACD", "Bearish", "bearish"))
    
    # Bollinger Bands
    if latest['Close'] <= latest['BB_Lower']:
        score += 8
        signals.append(("Bollinger", "Di Lower Band (Oversold)", "bullish"))
    elif latest['Close'] >= latest['BB_Upper']:
        score -= 8
        signals.append(("Bollinger", "Di Upper Band (Overbought)", "bearish"))
    else:
        bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100
        signals.append(("Bollinger", f"Posisi: {bb_position:.0f}%", "neutral"))
    
    # Moving Average Trend
    if latest['Close'] > latest['SMA5'] > latest['SMA20']:
        score += 8
        signals.append(("Trend", "Uptrend (Price > SMA5 > SMA20)", "bullish"))
    elif latest['Close'] < latest['SMA5'] < latest['SMA20']:
        score -= 8
        signals.append(("Trend", "Downtrend (Price < SMA5 < SMA20)", "bearish"))
    elif latest['Close'] > latest['SMA20']:
        score += 3
        signals.append(("Trend", "Di atas SMA20", "bullish"))
    else:
        score -= 3
        signals.append(("Trend", "Di bawah SMA20", "bearish"))
    
    # ADX - Trend Strength
    adx = latest['ADX']
    if adx > 25:
        if score > 0:
            score += 5
        else:
            score -= 5
        signals.append(("ADX", f"Trend Kuat ({adx:.1f})", "bullish" if score > 0 else "bearish"))
    else:
        signals.append(("ADX", f"Trend Lemah ({adx:.1f})", "neutral"))
    
    # Volume
    vol_ratio = latest['Volume_Ratio']
    if vol_ratio > 2:
        score += 5
        signals.append(("Volume", f"Sangat Tinggi ({vol_ratio:.1f}x)", "bullish"))
    elif vol_ratio > 1.5:
        score += 3
        signals.append(("Volume", f"Tinggi ({vol_ratio:.1f}x)", "bullish"))
    elif vol_ratio > 1:
        score += 1
        signals.append(("Volume", f"Normal ({vol_ratio:.1f}x)", "neutral"))
    else:
        signals.append(("Volume", f"Rendah ({vol_ratio:.1f}x)", "bearish"))
    
    # Stochastic
    stoch_k = latest['Stoch_K']
    if stoch_k < 20:
        score += 5
        signals.append(("Stochastic", f"Oversold ({stoch_k:.1f})", "bullish"))
    elif stoch_k > 80:
        score -= 5
        signals.append(("Stochastic", f"Overbought ({stoch_k:.1f})", "bearish"))
    else:
        signals.append(("Stochastic", f"Netral ({stoch_k:.1f})", "neutral"))
    
    return max(-50, min(50, score)), signals

# ==================== FUNDAMENTAL ANALYSIS ====================
def calculate_fundamental_score(info: dict) -> Tuple[int, List[Tuple[str, str, str]], dict]:
    """Calculate fundamental score"""
    score = 0
    signals = []
    metrics = {}
    
    # Get metrics
    pe = info.get('trailingPE', 0) or 0
    forward_pe = info.get('forwardPE', 0) or 0
    peg = info.get('pegRatio', 0) or 0
    pb = info.get('priceToBook', 0) or 0
    ps = info.get('priceToSalesTrailing12Months', 0) or 0
    
    profit_margin = (info.get('profitMargins', 0) or 0) * 100
    roe = (info.get('returnOnEquity', 0) or 0) * 100
    roa = (info.get('returnOnAssets', 0) or 0) * 100
    
    current_ratio = info.get('currentRatio', 0) or 0
    debt_equity = (info.get('debtToEquity', 0) or 0) / 100
    
    revenue_growth = (info.get('revenueGrowth', 0) or 0) * 100
    earnings_growth = (info.get('earningsGrowth', 0) or 0) * 100
    
    fcf = info.get('freeCashflow', 0) or 0
    
    metrics = {
        'pe': pe, 'forward_pe': forward_pe, 'peg': peg, 'pb': pb, 'ps': ps,
        'profit_margin': profit_margin, 'roe': roe, 'roa': roa,
        'current_ratio': current_ratio, 'debt_equity': debt_equity,
        'revenue_growth': revenue_growth, 'earnings_growth': earnings_growth,
        'fcf': fcf
    }
    
    # ===== VALUATION (25 points) =====
    if pe > 0:
        if pe < 15:
            score += 8
            signals.append(("P/E Ratio", f"Undervalued ({pe:.1f})", "bullish"))
        elif pe < 25:
            score += 4
            signals.append(("P/E Ratio", f"Wajar ({pe:.1f})", "neutral"))
        elif pe < 35:
            score += 1
            signals.append(("P/E Ratio", f"Agak Tinggi ({pe:.1f})", "bearish"))
        else:
            score -= 3
            signals.append(("P/E Ratio", f"Overvalued ({pe:.1f})", "bearish"))
    
    if peg > 0:
        if peg < 1:
            score += 6
            signals.append(("PEG Ratio", f"Excellent ({peg:.2f})", "bullish"))
        elif peg < 1.5:
            score += 4
            signals.append(("PEG Ratio", f"Baik ({peg:.2f})", "bullish"))
        elif peg < 2:
            score += 2
        else:
            score -= 2
    
    if pb > 0:
        if pb < 1.5:
            score += 5
        elif pb < 3:
            score += 3
        elif pb > 5:
            score -= 2
    
    # ===== PROFITABILITY (25 points) =====
    if profit_margin > 20:
        score += 7
        signals.append(("Profit Margin", f"Excellent ({profit_margin:.1f}%)", "bullish"))
    elif profit_margin > 10:
        score += 4
        signals.append(("Profit Margin", f"Baik ({profit_margin:.1f}%)", "bullish"))
    elif profit_margin > 5:
        score += 2
    elif profit_margin > 0:
        score += 1
    else:
        score -= 3
        signals.append(("Profit Margin", f"Negatif ({profit_margin:.1f}%)", "bearish"))
    
    if roe > 20:
        score += 7
        signals.append(("ROE", f"Excellent ({roe:.1f}%)", "bullish"))
    elif roe > 15:
        score += 5
        signals.append(("ROE", f"Baik ({roe:.1f}%)", "bullish"))
    elif roe > 10:
        score += 2
    elif roe > 0:
        score += 1
    else:
        score -= 2
    
    if roa > 10:
        score += 6
    elif roa > 5:
        score += 3
    elif roa > 0:
        score += 1
    
    # ===== FINANCIAL HEALTH (25 points) =====
    if current_ratio >= 2:
        score += 7
        signals.append(("Current Ratio", f"Kuat ({current_ratio:.2f})", "bullish"))
    elif current_ratio >= 1.5:
        score += 5
        signals.append(("Current Ratio", f"Baik ({current_ratio:.2f})", "bullish"))
    elif current_ratio >= 1:
        score += 2
    elif current_ratio > 0:
        score -= 2
        signals.append(("Current Ratio", f"Rendah ({current_ratio:.2f})", "bearish"))
    
    if debt_equity <= 0.3:
        score += 7
        signals.append(("Debt/Equity", f"Rendah ({debt_equity:.2f})", "bullish"))
    elif debt_equity <= 0.5:
        score += 5
    elif debt_equity <= 1:
        score += 2
    elif debt_equity > 2:
        score -= 3
        signals.append(("Debt/Equity", f"Tinggi ({debt_equity:.2f})", "bearish"))
    
    if fcf > 0:
        score += 6
        signals.append(("Free Cash Flow", "Positif ✓", "bullish"))
    else:
        score -= 2
        signals.append(("Free Cash Flow", "Negatif ✗", "bearish"))
    
    # ===== GROWTH (25 points) =====
    if revenue_growth > 20:
        score += 7
        signals.append(("Revenue Growth", f"Excellent ({revenue_growth:.1f}%)", "bullish"))
    elif revenue_growth > 10:
        score += 5
        signals.append(("Revenue Growth", f"Baik ({revenue_growth:.1f}%)", "bullish"))
    elif revenue_growth > 5:
        score += 2
    elif revenue_growth > 0:
        score += 1
    else:
        score -= 2
    
    if earnings_growth > 20:
        score += 7
        signals.append(("Earnings Growth", f"Excellent ({earnings_growth:.1f}%)", "bullish"))
    elif earnings_growth > 10:
        score += 5
    elif earnings_growth > 0:
        score += 2
    else:
        score -= 2
    
    # Analyst Target
    target = info.get('targetMeanPrice', 0) or 0
    current = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or 0
    if target and current:
        upside = ((target - current) / current) * 100
        metrics['target_upside'] = upside
        if upside > 20:
            score += 5
            signals.append(("Analyst Target", f"+{upside:.1f}% upside", "bullish"))
        elif upside > 10:
            score += 3
        elif upside > 0:
            score += 1
        else:
            score -= 2
    
    return max(0, min(100, score)), signals, metrics

# ==================== CHART FUNCTIONS ====================
def create_main_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Create main technical chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f'{ticker} Price Chart', 'MACD', 'RSI', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Upper'],
        name='BB Upper', line=dict(color='rgba(128,128,128,0.5)', dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data['BB_Lower'],
        name='BB Lower', line=dict(color='rgba(128,128,128,0.5)', dash='dash'),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SMA5'],
        name='SMA5', line=dict(color='#2196F3', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data['SMA20'],
        name='SMA20', line=dict(color='#FF9800', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data['EMA21'],
        name='EMA21', line=dict(color='#9C27B0', width=1)
    ), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MACD'],
        name='MACD', line=dict(color='#2196F3')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Signal'],
        name='Signal', line=dict(color='#FF9800')
    ), row=2, col=1)
    
    colors = ['#26a69a' if val >= 0 else '#ef5350' for val in data['Histogram']]
    fig.add_trace(go.Bar(
        x=data.index, y=data['Histogram'],
        name='Histogram', marker_color=colors
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'],
        name='RSI', line=dict(color='#9C27B0')
    ), row=3, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(128,128,128,0.1)", line_width=0, row=3, col=1)
    
    # Volume
    colors = ['#ef5350' if data['Close'].iloc[i] < data['Open'].iloc[i] else '#26a69a' 
              for i in range(len(data))]
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'],
        name='Volume', marker_color=colors
    ), row=4, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Volume_SMA'],
        name='Vol SMA', line=dict(color='#FF9800', width=1)
    ), row=4, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    return fig

# ==================== STOCK LISTS ====================
IDX_STOCKS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK",
    "UNVR.JK", "HMSP.JK", "GGRM.JK", "ICBP.JK", "INDF.JK",
    "KLBF.JK", "PGAS.JK", "PTBA.JK", "SMGR.JK", "UNTR.JK",
    "ADRO.JK", "ANTM.JK", "INCO.JK", "MEDC.JK", "ITMG.JK",
    "BBNI.JK", "CPIN.JK", "ERAA.JK", "EXCL.JK", "BRIS.JK"
]

US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "AMD", "INTC", "NFLX", "DIS", "BA", "JPM", "V", "MA",
    "PYPL", "COIN", "UBER", "SNAP", "SHOP", "SE", "BABA"
]

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Scalping Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Technical + Fundamental Analysis for Smart Trading</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        market = st.selectbox(
            "🌍 Pilih Market",
            ["🇮🇩 Indonesia (IDX)", "🇺🇸 US Market"],
            index=0
        )
        
        is_idx = "Indonesia" in market
        stock_list = IDX_STOCKS if is_idx else US_STOCKS
        
        st.markdown("---")
        
        input_method = st.radio(
            "📝 Input Method",
            ["Pilih dari daftar", "Ketik manual"]
        )
        
        if input_method == "Pilih dari daftar":
            ticker = st.selectbox("📊 Pilih Saham", stock_list)
        else:
            ticker = st.text_input(
                "📊 Ketik Ticker",
                value=stock_list[0],
                help="Contoh: BBCA.JK untuk Indonesia, AAPL untuk US"
            ).upper()
            
            if is_idx and not ticker.endswith('.JK'):
                ticker = ticker + '.JK'
        
        st.markdown("---")
        
        interval = st.selectbox(
            "⏱️ Timeframe",
            ["5m", "15m", "30m", "1h", "1d"],
            index=1,
            help="Timeframe untuk analisis"
        )
        
        period = st.selectbox(
            "📅 Period",
            ["1d", "5d", "1mo", "3mo"],
            index=1,
            help="Rentang waktu data"
        )
        
        st.markdown("---")
        
        include_fundamental = st.checkbox(
            "📊 Include Fundamental Analysis",
            value=True
        )
        
        st.markdown("---")
        
        analyze_btn = st.button("🔍 ANALYZE", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📋 Quick Info")
        st.info(f"""
        **Market:** {"Indonesia" if is_idx else "US"}  
        **Ticker:** {ticker}  
        **Timeframe:** {interval}  
        **Period:** {period}
        """)
    
    # Main Content
    if analyze_btn:
        with st.spinner(f"🔄 Analyzing {ticker}..."):
            # Fetch data
            data, info = get_stock_data(ticker, period, interval)
            
            if data is None or data.empty:
                st.error(f"❌ Tidak dapat mengambil data untuk {ticker}. Pastikan ticker valid.")
                return
            
            if len(data) < 30:
                st.warning(f"⚠️ Data tidak cukup untuk analisis lengkap. Hanya {len(data)} data points.")
            
            # Get currency
            currency = get_currency(ticker, info)
            
            # Calculate indicators
            data = calculate_indicators(data)
            
            # Calculate scores
            tech_score, tech_signals = calculate_technical_score(data)
            
            fund_score = 0
            fund_signals = []
            fund_metrics = {}
            
            if include_fundamental and info:
                fund_score, fund_signals, fund_metrics = calculate_fundamental_score(info)
            
            # Combined score
            if include_fundamental:
                combined_score = int((tech_score * 0.6) + (fund_score * 0.4))
            else:
                combined_score = tech_score
            
            # Current values
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            atr = data['ATR'].iloc[-1]
            
            # Determine signal
            if combined_score >= 40:
                signal_text = "🟢 STRONG BUY"
                signal_class = "buy-signal"
            elif combined_score >= 20:
                signal_text = "🔵 BUY"
                signal_class = "buy-signal"
            elif combined_score >= -20:
                signal_text = "⚪ NEUTRAL"
                signal_class = "neutral-signal"
            elif combined_score >= -40:
                signal_text = "🟠 SELL"
                signal_class = "sell-signal"
            else:
                signal_text = "🔴 STRONG SELL"
                signal_class = "sell-signal"
            
            # ===== DISPLAY RESULTS =====
            
            # Company info
            company_name = info.get('shortName', ticker) if info else ticker
            sector = info.get('sector', 'N/A') if info else 'N/A'
            industry = info.get('industry', 'N/A') if info else 'N/A'
            
            st.markdown(f"## {company_name}")
            st.markdown(f"**Sector:** {sector} | **Industry:** {industry} | **Currency:** {currency}")
            
            st.markdown("---")
            
            # Key Metrics Row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "💰 Current Price",
                    CurrencyHelper.format_price(current_price, currency),
                    f"{price_change:+.2f}%"
                )
            
            with col2:
                st.metric("📊 Combined Score", f"{combined_score}")
            
            with col3:
                st.metric("📈 Technical", f"{tech_score}/50")
            
            with col4:
                if include_fundamental:
                    st.metric("📋 Fundamental", f"{fund_score}/100")
                else:
                    st.metric("📋 Fundamental", "N/A")
            
            with col5:
                st.markdown(f'<div class="{signal_class}">{signal_text}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Chart
            st.subheader("📊 Technical Chart")
            fig = create_main_chart(data, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Trading Plan & Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Trading Plan")
                
                # Calculate levels
                if combined_score > 0:
                    entry = current_price
                    target = current_price + (atr * 1.5)
                    stop_loss = current_price - atr
                    trade_type = "LONG 📈"
                else:
                    entry = current_price
                    target = current_price - (atr * 1.5)
                    stop_loss = current_price + atr
                    trade_type = "SHORT 📉"
                
                risk = abs(entry - stop_loss)
                reward = abs(target - entry)
                rr_ratio = reward / risk if risk > 0 else 0
                profit_pct = (reward / entry) * 100
                
                # Confidence
                if abs(combined_score) >= 40 and rr_ratio >= 1.5:
                    confidence = "🟢 HIGH"
                elif abs(combined_score) >= 20 and rr_ratio >= 1.2:
                    confidence = "🟡 MEDIUM"
                else:
                    confidence = "🔴 LOW"
                
                st.markdown(f"""
                | Parameter | Value |
                |-----------|-------|
                | **Trade Type** | {trade_type} |
                | **Entry** | {CurrencyHelper.format_price(entry, currency)} |
                | **Target** | {CurrencyHelper.format_price(target, currency)} |
                | **Stop Loss** | {CurrencyHelper.format_price(stop_loss, currency)} |
                | **Risk/Reward** | 1:{rr_ratio:.2f} |
                | **Potential Profit** | {profit_pct:.2f}% |
                | **Confidence** | {confidence} |
                """)
                
                # Support/Resistance
                support = data['Low'].tail(20).min()
                resistance = data['High'].tail(20).max()
                
                st.markdown("#### 📍 Key Levels")
                st.markdown(f"""
                - **Support:** {CurrencyHelper.format_price(support, currency)}
                - **Resistance:** {CurrencyHelper.format_price(resistance, currency)}
                - **ATR:** {CurrencyHelper.format_price(atr, currency)}
                - **Volatility:** {(atr/current_price)*100:.2f}%
                """)
            
            with col2:
                st.subheader("📈 Technical Signals")
                
                for indicator, value, signal_type in tech_signals:
                    if signal_type == "bullish":
                        st.success(f"✅ **{indicator}:** {value}")
                    elif signal_type == "bearish":
                        st.error(f"⚠️ **{indicator}:** {value}")
                    else:
                        st.info(f"📊 **{indicator}:** {value}")
            
            # Fundamental Analysis
            if include_fundamental and info:
                st.markdown("---")
                st.subheader("📊 Fundamental Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 💰 Valuation")
                    pe = fund_metrics.get('pe', 0)
                    peg = fund_metrics.get('peg', 0)
                    pb = fund_metrics.get('pb', 0)
                    
                    st.markdown(f"""
                    - **P/E Ratio:** {pe:.2f if pe else 'N/A'}
                    - **Forward P/E:** {fund_metrics.get('forward_pe', 0):.2f if fund_metrics.get('forward_pe') else 'N/A'}
                    - **PEG Ratio:** {peg:.2f if peg else 'N/A'}
                    - **P/B Ratio:** {pb:.2f if pb else 'N/A'}
                    """)
                
                with col2:
                    st.markdown("#### 📈 Profitability")
                    st.markdown(f"""
                    - **Profit Margin:** {fund_metrics.get('profit_margin', 0):.1f}%
                    - **ROE:** {fund_metrics.get('roe', 0):.1f}%
                    - **ROA:** {fund_metrics.get('roa', 0):.1f}%
                    """)
                
                with col3:
                    st.markdown("#### 🏦 Financial Health")
                    st.markdown(f"""
                    - **Current Ratio:** {fund_metrics.get('current_ratio', 0):.2f}
                    - **Debt/Equity:** {fund_metrics.get('debt_equity', 0):.2f}
                    - **FCF:** {"✅ Positif" if fund_metrics.get('fcf', 0) > 0 else "❌ Negatif"}
                    """)
                
                # Fundamental signals
                st.markdown("#### 📋 Fundamental Signals")
                cols = st.columns(3)
                for i, (indicator, value, signal_type) in enumerate(fund_signals):
                    with cols[i % 3]:
                        if signal_type == "bullish":
                            st.success(f"✅ {indicator}: {value}")
                        elif signal_type == "bearish":
                            st.error(f"⚠️ {indicator}: {value}")
                        else:
                            st.info(f"📊 {indicator}: {value}")
                
                # Analyst info
                if info.get('targetMeanPrice'):
                    st.markdown("---")
                    st.markdown("#### 🎯 Analyst Consensus")
                    
                    target_price = info.get('targetMeanPrice', 0)
                    num_analysts = info.get('numberOfAnalystOpinions', 0)
                    rating = info.get('recommendationKey', 'N/A')
                    upside = fund_metrics.get('target_upside', 0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Target Price", CurrencyHelper.format_price(target_price, currency))
                    with col2:
                        st.metric("Upside", f"{upside:+.1f}%")
                    with col3:
                        st.metric("Rating", rating.upper() if rating else "N/A")
                    with col4:
                        st.metric("# Analysts", num_analysts)
            
            # Disclaimer
            st.markdown("---")
            st.warning("""
            ⚠️ **DISCLAIMER:**
            - Analisis ini hanya untuk tujuan **edukasi** dan **informasi**
            - **BUKAN** merupakan rekomendasi atau ajakan untuk membeli/menjual saham
            - Trading saham memiliki **risiko kerugian finansial**
            - Selalu lakukan **riset mandiri** sebelum mengambil keputusan investasi
            - Konsultasikan dengan **financial advisor** profesional
            """)
    
    else:
        # Welcome screen
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Fitur Utama
            
            **📈 Technical Analysis:**
            - RSI (Relative Strength Index)
            - MACD (Moving Average Convergence Divergence)
            - Bollinger Bands
            - Stochastic Oscillator
            - Moving Averages (SMA, EMA)
            - Volume Analysis
            - ADX (Trend Strength)
            
            **📊 Fundamental Analysis:**
            - Valuation (P/E, PEG, P/B)
            - Profitability (ROE, ROA, Margins)
            - Financial Health (Current Ratio, D/E)
            - Growth Metrics
            - Analyst Consensus
            """)
        
        with col2:
            st.markdown("""
            ### 📖 Cara Penggunaan
            
            1. **Pilih Market** - Indonesia (IDX) atau US
            2. **Pilih Saham** - Dari daftar atau ketik manual
            3. **Atur Timeframe** - 5m, 15m, 30m, 1h, atau 1d
            4. **Klik ANALYZE** - Untuk memulai analisis
            
            ### ⚡ Quick Tips
            
            - **Combined Score > 40** = Strong Buy Signal
            - **Risk/Reward > 1.5** = Good Trade Setup
            - **Volume Ratio > 1.5** = High Interest
            - **RSI < 30** = Oversold (Buy Opportunity)
            - **RSI > 70** = Overbought (Sell Signal)
            """)
        
        st.markdown("---")
        st.info("👈 **Pilih saham di sidebar dan klik ANALYZE untuk memulai**")

if __name__ == "__main__":
    main()