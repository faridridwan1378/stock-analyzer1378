#!/usr/bin/env python3
"""
📈 Stock Scalping Analyzer - FIXED VERSION
Dengan error handling yang lebih baik
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
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
    def get_symbol(cls, currency: str) -> str:
        return cls.CURRENCY_SYMBOLS.get(currency.upper(), currency)
    
    @classmethod
    def format_price(cls, price: float, currency: str) -> str:
        if price is None or pd.isna(price):
            return "N/A"
        symbol = cls.get_symbol(currency)
        if currency.upper() in cls.NO_DECIMAL:
            return f"{symbol} {price:,.0f}"
        return f"{symbol} {price:,.2f}"

# ==================== DATA FETCHING WITH ERROR HANDLING ====================

def validate_ticker(ticker: str) -> dict:
    """Validasi apakah ticker valid"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if info contains valid data
        if info is None:
            return {'valid': False, 'error': 'Ticker tidak ditemukan'}
        
        # Check for common invalid ticker indicators
        if info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
            # Try to get basic info
            if info.get('symbol') is None:
                return {'valid': False, 'error': 'Ticker tidak valid'}
        
        return {'valid': True, 'info': info}
    
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def fetch_stock_data_robust(ticker: str, period: str = "5d", interval: str = "15m", max_retries: int = 3):
    """
    Fetch stock data dengan error handling yang robust
    """
    
    # Interval fallback order
    interval_fallbacks = {
        '1m': ['5m', '15m', '30m', '1h'],
        '5m': ['15m', '30m', '1h', '1d'],
        '15m': ['30m', '1h', '1d', '5m'],
        '30m': ['1h', '15m', '1d', '5m'],
        '1h': ['30m', '1d', '15m', '5m'],
        '1d': ['1h', '30m', '15m', '5m'],
    }
    
    # Period adjustments for different intervals
    period_for_interval = {
        '1m': '7d',      # 1m data only available for 7 days
        '5m': '60d',     # 5m data available for 60 days
        '15m': '60d',
        '30m': '60d',
        '1h': '730d',    # 1h data available for 2 years
        '1d': 'max',
    }
    
    errors = []
    
    # Try primary interval first
    intervals_to_try = [interval] + interval_fallbacks.get(interval, [])
    
    for current_interval in intervals_to_try:
        # Adjust period based on interval limitations
        adjusted_period = period
        if current_interval in ['1m']:
            adjusted_period = '5d' if period in ['1mo', '3mo', 'max'] else period
        elif current_interval in ['5m', '15m', '30m']:
            adjusted_period = '5d' if period in ['3mo', 'max'] else period
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                # Add delay between retries to avoid rate limiting
                if attempt > 0:
                    time.sleep(1)
                
                # Fetch historical data
                data = stock.history(
                    period=adjusted_period,
                    interval=current_interval,
                    auto_adjust=True,
                    prepost=False
                )
                
                # Check if data is valid
                if data is not None and not data.empty and len(data) >= 20:
                    # Get info
                    info = stock.info
                    
                    return {
                        'success': True,
                        'data': data,
                        'info': info,
                        'interval_used': current_interval,
                        'period_used': adjusted_period,
                        'message': f"Data berhasil diambil dengan interval {current_interval}"
                    }
                else:
                    errors.append(f"Interval {current_interval}: Data kosong atau tidak cukup")
                    
            except Exception as e:
                error_msg = str(e)
                errors.append(f"Interval {current_interval}, Attempt {attempt+1}: {error_msg}")
                
                # If rate limited, wait longer
                if 'rate' in error_msg.lower() or 'limit' in error_msg.lower():
                    time.sleep(2)
                
                continue
    
    # All attempts failed
    return {
        'success': False,
        'data': None,
        'info': None,
        'errors': errors,
        'message': 
