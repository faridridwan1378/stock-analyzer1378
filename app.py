#!/usr/bin/env python3
"""
📈 Stock Scalping Analyzer - FIXED VERSION
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
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
</style>
""", unsafe_allow_html=True)

# ==================== CURRENCY HELPER ====================
class CurrencyHelper:
    CURRENCY_SYMBOLS = {
        'IDR': 'Rp', 'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
        'SGD': 'S$', 'HKD': 'HK$', 'CNY': '¥', 'KRW': '₩', 'MYR': 'RM',
    }
    NO_DECIMAL = ['IDR', 'JPY', 'KRW', 'VND']
    
    @classmethod
    def get_symbol(cls, currency):
        return cls.CURRENCY_SYMBOLS.get(currency.upper(), currency)
    
    @classmethod
    def format_price(cls, price, currency):
        if price is None or pd.isna(price):
            return "N/A"
        symbol = cls.get_symbol(currency)
        if currency.upper() in cls.NO_DECIMAL:
            return f"{symbol} {price:,.0f}"
        return f"{symbol} {price:,.2f}"

# ==================== DATA FETCHING ====================
def get_stock_data(ticker, period="1mo", interval="1d"):
    """
    Fetch stock data dengan error handling
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Coba ambil data
        data = stock.history(period=period, interval=interval)
        
        if data is None or data.empty:
            # Coba dengan interval berbeda
            fallback_intervals = ['1d', '1h', '5d']
            for fb_interval in fallback_intervals:
                if fb_interval != interval:
                    try:
                        data = stock.history(period=period, interval=fb_interval)
                        if data is not None and not data.empty:
                            break
                    except:
                        continue
        
        if data is None or data.empty:
            return None, None, "Data tidak tersedia"
        
        # Ambil info
        try:
            info = stock.info
        except:
            info = {'symbol': ticker}
        
        return data, info, None
        
    except Exception as e:
        return None, None, str(e)

# ==================== TECHNICAL INDICATORS ====================
def calculate_indicators(data):
    """Calculate technical indicators"""
    if data is None or data.empty:
        return data
    
    try:
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.001)
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'] = data['RSI'].fillna(50)
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
        data['MACD'] = ema12 - ema26
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        data['Histogram'] = data['MACD'] - data['Signal']
        
        # Bollinger Bands
        data['SMA20'] = close.rolling(window=20, min_periods=1).mean()
        std = close.rolling(window=20, min_periods=1).std()
        data['BB_Upper'] = data['SMA20'] + (std * 2)
        data['BB_Lower'] = data['SMA20'] - (std * 2)
        
        # Moving Averages
        data['SMA5'] = close.rolling(window=5, min_periods=1).mean()
        data['SMA10'] = close.rolling(window=10, min_periods=1).mean()
        data['EMA9'] = close.ewm(span=9, adjust=False, min_periods=1).mean()
        data['EMA21'] = close.ewm(span=21, adjust=False, min_periods=1).mean()
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(window=14, min_periods=1).mean()
        
        # Volume
        data['Volume_SMA'] = volume.rolling(window=20, min_periods=1).mean()
        vol_sma = data['Volume_SMA'].replace(0, 1)
        data['Volume_Ratio'] = volume / vol_sma
        
        # Stochastic
        lowest_low = low.rolling(window=14, min_periods=1).min()
        highest_high = high.rolling(window=14, min_periods=1).max()
        diff = highest_high - lowest_low
        diff = diff.replace(0, 0.001)
        data['Stoch_K'] = 100 * ((close - lowest_low) / diff)
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3, min_periods=1).mean()
        
        # Fill NaN
        data = data.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating indicators: {e}")
        return data

# ==================== TECHNICAL SCORE ====================
def calculate_technical_score(data):
    """Calculate technical score"""
    if data is None or len(data) < 2:
        return 0, [("Error", "Data tidak cukup", "neutral")]
    
    try:
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        score = 0
        signals = []
        
        # RSI
        rsi = float(latest.get('RSI', 50))
        if rsi < 30:
            score += 10
            signals.append(("RSI", f"Oversold ({rsi:.1f})", "bullish"))
        elif rsi < 40:
            score += 5
            signals.append(("RSI", f"Low ({rsi:.1f})", "bullish"))
        elif rsi > 70:
            score -= 10
            signals.append(("RSI", f"Overbought ({rsi:.1f})", "bearish"))
        elif rsi > 60:
            score -= 3
            signals.append(("RSI", f"High ({rsi:.1f})", "bearish"))
        else:
            signals.append(("RSI", f"Netral ({rsi:.1f})", "neutral"))
        
        # MACD
        macd_curr = float(latest.get('MACD', 0))
        signal_curr = float(latest.get('Signal', 0))
        macd_prev = float(prev.get('MACD', 0))
        signal_prev = float(prev.get('Signal', 0))
        
        if macd_prev < signal_prev and macd_curr > signal_curr:
            score += 10
            signals.append(("MACD", "Bullish Cross ✨", "bullish"))
        elif macd_prev > signal_prev and macd_curr < signal_curr:
            score -= 10
            signals.append(("MACD", "Bearish Cross ⚠️", "bearish"))
        elif macd_curr > signal_curr:
            score += 5
            signals.append(("MACD", "Bullish", "bullish"))
        else:
            score -= 5
            signals.append(("MACD", "Bearish", "bearish"))
        
        # Bollinger Bands
        close = float(latest.get('Close', 0))
        bb_upper = float(latest.get('BB_Upper', close))
        bb_lower = float(latest.get('BB_Lower', close))
        
        if close <= bb_lower:
            score += 8
            signals.append(("Bollinger", "Lower Band (Oversold)", "bullish"))
        elif close >= bb_upper:
            score -= 8
            signals.append(("Bollinger", "Upper Band (Overbought)", "bearish"))
        else:
            signals.append(("Bollinger", "Normal Range", "neutral"))
        
        # Trend
        sma5 = float(latest.get('SMA5', close))
        sma20 = float(latest.get('SMA20', close))
        
        if close > sma5 > sma20:
            score += 8
            signals.append(("Trend", "Uptrend 📈", "bullish"))
        elif close < sma5 < sma20:
            score -= 8
            signals.append(("Trend", "Downtrend 📉", "bearish"))
        else:
            signals.append(("Trend", "Sideways ➡️", "neutral"))
        
        # Volume
        vol_ratio = float(latest.get('Volume_Ratio', 1))
        if vol_ratio > 2:
            score += 5
            signals.append(("Volume", f"Very High ({vol_ratio:.1f}x)", "bullish"))
        elif vol_ratio > 1.5:
            score += 3
            signals.append(("Volume", f"High ({vol_ratio:.1f}x)", "bullish"))
        else:
            signals.append(("Volume", f"Normal ({vol_ratio:.1f}x)", "neutral"))
        
        # Stochastic
        stoch_k = float(latest.get('Stoch_K', 50))
        if stoch_k < 20:
            score += 5
            signals.append(("Stochastic", f"Oversold ({stoch_k:.1f})", "bullish"))
        elif stoch_k > 80:
            score -= 5
            signals.append(("Stochastic", f"Overbought ({stoch_k:.1f})", "bearish"))
        else:
            signals.append(("Stochastic", f"Netral ({stoch_k:.1f})", "neutral"))
        
        score = max(-50, min(50, score))
        return score, signals
        
    except Exception as e:
        return 0, [("Error", str(e), "neutral")]

# ==================== FUNDAMENTAL SCORE ====================
def calculate_fundamental_score(info):
    """Calculate fundamental score"""
    if not info:
        return 0, [], {}
    
    score = 0
    signals = []
    metrics = {}
    
    try:
        # Get metrics
        pe = info.get('trailingPE') or 0
        peg = info.get('pegRatio') or 0
        pb = info.get('priceToBook') or 0
        
        profit_margin = (info.get('profitMargins') or 0) * 100
        roe = (info.get('returnOnEquity') or 0) * 100
        roa = (info.get('returnOnAssets') or 0) * 100
        
        current_ratio = info.get('currentRatio') or 0
        debt_equity = (info.get('debtToEquity') or 0) / 100
        
        revenue_growth = (info.get('revenueGrowth') or 0) * 100
        earnings_growth = (info.get('earningsGrowth') or 0) * 100
        
        fcf = info.get('freeCashflow') or 0
        
        metrics = {
            'pe': pe,
            'peg': peg,
            'pb': pb,
            'profit_margin': profit_margin,
            'roe': roe,
            'roa': roa,
            'current_ratio': current_ratio,
            'debt_equity': debt_equity,
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'fcf': fcf
        }
        
        # P/E
        if pe > 0:
            if pe < 15:
                score += 8
                signals.append(("P/E", f"Murah ({pe:.1f})", "bullish"))
            elif pe < 25:
                score += 4
                signals.append(("P/E", f"Wajar ({pe:.1f})", "neutral"))
            else:
                score -= 2
                signals.append(("P/E", f"Mahal ({pe:.1f})", "bearish"))
        
        # PEG
        if peg > 0 and peg < 1:
            score += 6
            signals.append(("PEG", f"Excellent ({peg:.2f})", "bullish"))
        elif peg > 0 and peg < 1.5:
            score += 3
        
        # Profitability
        if profit_margin > 20:
            score += 7
            signals.append(("Margin", f"Tinggi ({profit_margin:.1f}%)", "bullish"))
        elif profit_margin > 10:
            score += 4
        elif profit_margin < 0:
            score -= 3
            signals.append(("Margin", f"Negatif ({profit_margin:.1f}%)", "bearish"))
        
        if roe > 20:
            score += 7
            signals.append(("ROE", f"Excellent ({roe:.1f}%)", "bullish"))
        elif roe > 15:
            score += 4
        elif roe < 0:
            score -= 2
        
        # Financial Health
        if current_ratio >= 2:
            score += 6
            signals.append(("Current Ratio", f"Kuat ({current_ratio:.2f})", "bullish"))
        elif current_ratio >= 1.5:
            score += 4
        elif current_ratio > 0 and current_ratio < 1:
            score -= 2
            signals.append(("Current Ratio", f"Lemah ({current_ratio:.2f})", "bearish"))
        
        if debt_equity > 0 and debt_equity <= 0.5:
            score += 6
            signals.append(("D/E", f"Rendah ({debt_equity:.2f})", "bullish"))
        elif debt_equity > 2:
            score -= 3
            signals.append(("D/E", f"Tinggi ({debt_equity:.2f})", "bearish"))
        
        if fcf > 0:
            score += 5
            signals.append(("FCF", "Positif ✓", "bullish"))
        elif fcf < 0:
            score -= 2
            signals.append(("FCF", "Negatif ✗", "bearish"))
        
        # Growth
        if revenue_growth > 20:
            score += 6
            signals.append(("Revenue", f"+{revenue_growth:.1f}%", "bullish"))
        elif revenue_growth > 10:
            score += 3
        elif revenue_growth < 0:
            score -= 2
        
        if earnings_growth > 20:
            score += 6
            signals.append(("Earnings", f"+{earnings_growth:.1f}%", "bullish"))
        elif earnings_growth > 10:
            score += 3
        
        # Analyst
        target = info.get('targetMeanPrice') or 0
        current = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        
        if target > 0 and current > 0:
            upside = ((target - current) / current) * 100
            metrics['target_price'] = target
            metrics['target_upside'] = upside
            
            if upside > 20:
                score += 5
                signals.append(("Target", f"+{upside:.1f}%", "bullish"))
            elif upside > 10:
                score += 3
        
        score = max(0, min(100, score))
        return score, signals, metrics
        
    except Exception as e:
        return 0, [("Error", str(e), "neutral")], {}

# ==================== CHART ====================
def create_chart(data, ticker):
    """Create chart"""
    if data is None or data.empty:
        return None
    
    try:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'{ticker} Price', 'MACD', 'RSI', 'Volume')
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        # Bollinger
        if 'BB_Upper' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(128,128,128,0.5)', dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(128,128,128,0.5)', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ), row=1, col=1)
        
        # MA
        if 'SMA5' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA5'],
                name='SMA5',
                line=dict(color='#2196F3', width=1)
            ), row=1, col=1)
        
        if 'SMA20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA20'],
                name='SMA20',
                line=dict(color='#FF9800', width=1)
            ), row=1, col=1)
        
        # MACD
        if 'MACD' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MACD'],
                name='MACD',
                line=dict(color='#2196F3')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Signal'],
                name='Signal',
                line=dict(color='#FF9800')
            ), row=2, col=1)
            
            colors = ['#26a69a' if v >= 0 else '#ef5350' for v in data['Histogram']]
            fig.add_trace(go.Bar(
                x=data.index, y=data['Histogram'],
                name='Histogram',
                marker_color=colors
            ), row=2, col=1)
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['RSI'],
                name='RSI',
                line=dict(color='#9C27B0')
            ), row=3, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Volume
        colors = []
        for i in range(len(data)):
            if data['Close'].iloc[i] < data['Open'].iloc[i]:
                colors.append('#ef5350')
            else:
                colors.append('#26a69a')
        
        fig.add_trace(go.Bar(
            x=data.index, y=data['Volume'],
            name='Volume',
            marker_color=colors
        ), row=4, col=1)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Chart error: {e}")
        return None

# ==================== STOCK LISTS ====================
IDX_STOCKS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "UNVR.JK", "HMSP.JK", "ICBP.JK", "INDF.JK",
    "KLBF.JK", "PGAS.JK", "PTBA.JK", "SMGR.JK", "UNTR.JK",
    "ADRO.JK", "ANTM.JK", "INCO.JK", "MEDC.JK", "ITMG.JK"
]

US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "AMD", "INTC", "NFLX", "DIS", "BA", "JPM", "V", "MA",
    "PYPL", "COIN", "UBER", "SNAP", "SHOP"
]

def get_currency(ticker, info):
    """Get currency"""
    if info:
        curr = info.get('currency')
        if curr:
            return curr
    
    if ticker.endswith('.JK'):
        return 'IDR'
    elif ticker.endswith('.SI'):
        return 'SGD'
    elif ticker.endswith('.HK'):
        return 'HKD'
    return 'USD'

# ==================== MAIN APP ====================
def main():
    st.markdown('<h1 class="main-header">📈 Stock Scalping Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Technical + Fundamental Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        market = st.selectbox(
            "🌍 Market",
            ["🇮🇩 Indonesia (IDX)", "🇺🇸 US Market"]
        )
        
        is_idx = "Indonesia" in market
        stock_list = IDX_STOCKS if is_idx else US_STOCKS
        
        st.markdown("---")
        
        input_method = st.radio("📝 Input", ["Pilih dari daftar", "Ketik manual"])
        
        if input_method == "Pilih dari daftar":
            ticker = st.selectbox("📊 Saham", stock_list)
        else:
            default_val = "BBCA" if is_idx else "AAPL"
            ticker = st.text_input("📊 Ticker", value=default_val).upper().strip()
            if is_idx and ticker and not ticker.endswith('.JK'):
                ticker = ticker + '.JK'
        
        st.markdown("---")
        
        interval = st.selectbox(
            "⏱️ Interval",
            ["1d", "1h", "30m", "15m", "5m"],
            index=0
        )
        
        period = st.selectbox(
            "📅 Period",
            ["5d", "1mo", "3mo", "6mo", "1y"],
            index=1
        )
        
        st.markdown("---")
        
        include_fundamental = st.checkbox("📊 Fundamental", value=True)
        
        st.markdown("---")
        
        analyze_btn = st.button("🔍 ANALYZE", type="primary", use_container_width=True)
    
    # Main Content
    if analyze_btn:
        if not ticker:
            st.error("❌ Masukkan ticker!")
            return
        
        with st.spinner(f"Analyzing {ticker}..."):
            # Fetch data
            data, info, error = get_stock_data(ticker, period, interval)
            
            if error or data is None or data.empty:
                st.error(f"❌ Gagal mengambil data untuk {ticker}")
                if error:
                    st.write(f"Error: {error}")
                
                st.info("""
                💡 **Tips:**
                - Gunakan interval 1d (daily) untuk hasil lebih stabil
                - Pastikan ticker valid (contoh: BBCA.JK, AAPL)
                - Coba lagi dalam beberapa saat
                """)
                return
            
            # Calculate
            data = calculate_indicators(data)
            
            currency = get_currency(ticker, info)
            
            tech_score, tech_signals = calculate_technical_score(data)
            
            fund_score = 0
            fund_signals = []
            fund_metrics = {}
            
            if include_fundamental and info:
                fund_score, fund_signals, fund_metrics = calculate_fundamental_score(info)
            
            # Combined score
            if include_fundamental and fund_score > 0:
                combined_score = int((tech_score * 0.6) + (fund_score * 0.4))
            else:
                combined_score = tech_score
            
            # Values
            current_price = float(data['Close'].iloc[-1])
            prev_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
            
            atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else current_price * 0.02
            
            # Signal
            if combined_score >= 40:
                signal_text = "🟢 STRONG BUY"
            elif combined_score >= 20:
                signal_text = "🔵 BUY"
            elif combined_score >= -20:
                signal_text = "⚪ NEUTRAL"
            elif combined_score >= -40:
                signal_text = "🟠 SELL"
            else:
                signal_text = "🔴 STRONG SELL"
            
            # Display
            st.success(f"✅ Data berhasil diambil!")
            
            company_name = info.get('shortName', ticker) if info else ticker
            sector = info.get('sector', 'N/A') if info else 'N/A'
            
            st.markdown(f"## {company_name}")
            st.markdown(f"**{sector}** | Currency: **{currency}**")
            
            st.markdown("---")
            
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "💰 Price",
                    CurrencyHelper.format_price(current_price, currency),
                    f"{price_change:+.2f}%"
                )
            
            with col2:
                st.metric("📊 Combined", f"{combined_score}")
            
            with col3:
                st.metric("📈 Technical", f"{tech_score}/50")
            
            with col4:
                if include_fundamental:
                    st.metric("📋 Fundamental", f"{fund_score}/100")
                else:
                    st.metric("📋 Fundamental", "N/A")
            
            with col5:
                st.markdown(f"### {signal_text}")
            
            st.markdown("---")
            
            # Chart
            st.subheader("📊 Chart")
            fig = create_chart(data, ticker)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Trading Plan")
                
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
                profit_pct = (reward / entry) * 100 if entry > 0 else 0
                
                st.write(f"**Type:** {trade_type}")
                st.write(f"**Entry:** {CurrencyHelper.format_price(entry, currency)}")
                st.write(f"**Target:** {CurrencyHelper.format_price(target, currency)}")
                st.write(f"**Stop Loss:** {CurrencyHelper.format_price(stop_loss, currency)}")
                st.write(f"**R/R:** 1:{rr_ratio:.2f}")
                st.write(f"**Profit:** {profit_pct:.2f}%")
                
                support = float(data['Low'].tail(20).min())
                resistance = float(data['High'].tail(20).max())
                
                st.markdown("---")
                st.write(f"**Support:** {CurrencyHelper.format_price(support, currency)}")
                st.write(f"**Resistance:** {CurrencyHelper.format_price(resistance, currency)}")
            
            with col2:
                st.subheader("📈 Technical Signals")
                
                for indicator, value, sig_type in tech_signals:
                    if sig_type == "bullish":
                        st.success(f"✅ **{indicator}:** {value}")
                    elif sig_type == "bearish":
                        st.error(f"⚠️ **{indicator}:** {value}")
                    else:
                        st.info(f"📊 **{indicator}:** {value}")
            
            # Fundamental
            if include_fundamental and fund_signals:
                st.markdown("---")
                st.subheader("📊 Fundamental")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**P/E:** {fund_metrics.get('pe', 0):.1f}")
                    st.write(f"**PEG:** {fund_metrics.get('peg', 0):.2f}")
                    st.write(f"**P/B:** {fund_metrics.get('pb', 0):.2f}")
                
                with col2:
                    st.write(f"**Margin:** {fund_metrics.get('profit_margin', 0):.1f}%")
                    st.write(f"**ROE:** {fund_metrics.get('roe', 0):.1f}%")
                    st.write(f"**ROA:** {fund_metrics.get('roa', 0):.1f}%")
                
                with col3:
                    st.write(f"**Current:** {fund_metrics.get('current_ratio', 0):.2f}")
                    st.write(f"**D/E:** {fund_metrics.get('debt_equity', 0):.2f}")
                    fcf_status = "✅" if fund_metrics.get('fcf', 0) > 0 else "❌"
                    st.write(f"**FCF:** {fcf_status}")
                
                st.markdown("**Signals:**")
                cols = st.columns(3)
                for i, (ind, val, sig) in enumerate(fund_signals):
                    with cols[i % 3]:
                        if sig == "bullish":
                            st.success(f"{ind}: {val}")
                        elif sig == "bearish":
                            st.error(f"{ind}: {val}")
                        else:
                            st.info(f"{ind}: {val}")
            
            # Disclaimer
            st.markdown("---")
            st.warning("⚠️ **DISCLAIMER:** Analisis ini hanya untuk edukasi. Bukan rekomendasi investasi.")
    
    else:
        # Welcome
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Fitur
            - RSI, MACD, Bollinger Bands
            - Moving Averages
            - Volume Analysis
            - Stochastic
            - Fundamental Metrics
            - Trading Plan
            """)
        
        with col2:
            st.markdown("""
            ### 📖 Cara Pakai
            1. Pilih market (IDX/US)
            2. Pilih atau ketik ticker
            3. Atur interval & period
            4. Klik **ANALYZE**
            """)
        
        st.info("👈 Pilih saham di sidebar dan klik **ANALYZE**")

if __name__ == "__main__":
    main()
