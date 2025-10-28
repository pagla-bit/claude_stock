"""
Enhanced Streamlit Stock Dashboard v2.0
Key improvements:
- Fixed signal weighting logic with proper modifiers
- Advanced Monte Carlo with fat-tailed distributions
- Comprehensive risk metrics (Sharpe, Sortino, Max DD, Calmar)
- Strategy backtesting with performance comparison
- Improved data validation
- Optimized batch fetching
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Optional sentiment dependencies
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    SENTIMENT_AVAILABLE = True
except Exception:
    SENTIMENT_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Enhanced Stock Dashboard v2.0 (by Sadiq)")

# -------------------- Data Fetching & Validation --------------------

@st.cache_data(ttl=3600)
def get_data_optimized(ticker: str, period: str = "1y", interval: str = "1d", fetch_info: bool = True):
    """
    Optimized data fetch with selective info retrieval
    Returns (hist_df, info_dict) or (empty_df, error_dict)
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if hist.empty:
            raise ValueError("Empty history returned")
        
        missing = set(required_cols) - set(hist.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if len(hist) < 50:
            raise ValueError("Need at least 50 data points")
        
        # Only fetch essential info to reduce latency
        info = {}
        if fetch_info:
            try:
                raw_info = tk.info
                info = {
                    'forwardPE': raw_info.get('forwardPE'),
                    'trailingPE': raw_info.get('trailingPE'),
                    'marketCap': raw_info.get('marketCap'),
                    'shortName': raw_info.get('shortName', ticker)
                }
            except:
                info = {'shortName': ticker}
        
        return hist, info
    except Exception as e:
        return pd.DataFrame(), {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_spy_data(period="1y", interval="1d"):
    """Cache SPY data for correlation and beta calculations"""
    hist, _ = get_data_optimized("SPY", period=period, interval=interval, fetch_info=False)
    return hist

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Fetch CNN Fear & Greed Index with fallback"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}
    base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    
    for days_back in range(0, 3):
        d = (date.today() - timedelta(days=days_back)).isoformat()
        try:
            resp = requests.get(base_url + d, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            fg = data.get("fear_and_greed", {})
            score = fg.get("score")
            rating = fg.get("rating", "N/A")
            
            if score is None:
                continue
            
            if score < 25:
                color = "🟥 Extreme Fear"
            elif score < 45:
                color = "🔴 Fear"
            elif score < 55:
                color = "🟡 Neutral"
            elif score < 75:
                color = "🟢 Greed"
            else:
                color = "🟩 Extreme Greed"
            
            return score, rating, color
        except Exception:
            continue
    
    return None, "N/A", "N/A"

# -------------------- Indicator Calculations --------------------

def calc_indicators(df: pd.DataFrame,
                    rsi_period=14,
                    macd_fast=12, macd_slow=26, macd_signal=9,
                    sma_short=20, sma_long=50,
                    bb_period=20, atr_period=14, adx_period=14):
    """Calculate technical indicators with validation"""
    df = df.copy()
    
    # Moving Averages
    df["SMA_short"] = df["Close"].rolling(sma_short).mean()
    df["SMA_long"] = df["Close"].rolling(sma_long).mean()
    df["EMA_short"] = df["Close"].ewm(span=sma_short, adjust=False).mean()
    
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period - 1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    # Bollinger Bands
    bb_mid = df["Close"].rolling(bb_period).mean()
    bb_std = df["Close"].rolling(bb_period).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_mid"] = bb_mid
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    
    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()
    df["ATR_pct"] = (df["ATR"] / df["Close"]) * 100
    
    # ADX
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr_smooth = tr.ewm(span=adx_period).mean()
    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX"] = dx.ewm(span=adx_period).mean()
    df["DI_plus"] = plus_di
    df["DI_minus"] = minus_di
    
    # Stochastic Oscillator
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    
    return df

def validate_indicators(df: pd.DataFrame, min_valid_pct=0.7):
    """Validate that indicators have sufficient valid data"""
    required = ['RSI', 'MACD', 'ATR', 'ADX']
    validation_results = {}
    
    for col in required:
        if col in df:
            valid_pct = df[col].notna().mean()
            validation_results[col] = valid_pct
            if valid_pct < min_valid_pct:
                return False, f"{col} has only {valid_pct*100:.1f}% valid data (need {min_valid_pct*100}%)"
    
    return True, validation_results

# -------------------- Enhanced Signal Generation --------------------

def rule_based_signal_v2(df: pd.DataFrame,
                         rsi_oversold=30,
                         rsi_overbought=70,
                         weights=None):
    """
    Enhanced signal generation with proper modifiers
    Returns (recommendation, signals_list, confidence, raw_scores)
    """
    if weights is None:
        weights = {'RSI': 2.0, 'MACD': 1.5, 'SMA': 1.0, 'BB': 1.0, 'Stoch': 0.8, 'Volume': 0.5, 'ADX': 1.0}
    
    if len(df) < 3:
        return "HOLD", [], 0.0, {'buy': 0, 'sell': 0}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    buy_score = 0.0
    sell_score = 0.0
    trend_multiplier = 1.0
    volume_multiplier = 1.0
    signals = []
    
    # ===== DIRECTIONAL SIGNALS =====
    
    # RSI
    if not np.isnan(latest['RSI']):
        if latest['RSI'] < rsi_oversold:
            buy_score += weights['RSI']
            signals.append(("RSI oversold", "BUY", weights['RSI'], f"RSI={latest['RSI']:.1f}"))
        elif latest['RSI'] > rsi_overbought:
            sell_score += weights['RSI']
            signals.append(("RSI overbought", "SELL", weights['RSI'], f"RSI={latest['RSI']:.1f}"))
        elif 45 <= latest['RSI'] <= 55:
            signals.append(("RSI neutral", "NEUTRAL", 0, f"RSI={latest['RSI']:.1f}"))
    
    # MACD Crossover
    if not np.isnan(latest['MACD']) and not np.isnan(prev['MACD']):
        if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
            buy_score += weights['MACD']
            signals.append(("MACD bullish crossover", "BUY", weights['MACD'], ""))
        elif (prev['MACD'] > prev['MACD_signal']) and (latest['MACD'] < latest['MACD_signal']):
            sell_score += weights['MACD']
            signals.append(("MACD bearish crossover", "SELL", weights['MACD'], ""))
        
        # MACD momentum
        if not np.isnan(latest['MACD_hist']):
            if latest['MACD_hist'] > 0 and latest['MACD_hist'] > prev['MACD_hist']:
                buy_score += weights['MACD'] * 0.5
                signals.append(("MACD momentum increasing", "BUY", weights['MACD']*0.5, ""))
            elif latest['MACD_hist'] < 0 and latest['MACD_hist'] < prev['MACD_hist']:
                sell_score += weights['MACD'] * 0.5
                signals.append(("MACD momentum decreasing", "SELL", weights['MACD']*0.5, ""))
    
    # SMA Trend
    if not np.isnan(latest['SMA_long']) and not np.isnan(latest['SMA_short']):
        if latest['Close'] > latest['SMA_long']:
            buy_score += weights['SMA']
            signals.append(("Price above long SMA", "BUY", weights['SMA'], ""))
        else:
            sell_score += weights['SMA']
            signals.append(("Price below long SMA", "SELL", weights['SMA'], ""))
        
        # Golden/Death Cross
        if latest['SMA_short'] > latest['SMA_long'] and prev['SMA_short'] <= prev['SMA_long']:
            buy_score += weights['SMA'] * 1.5
            signals.append(("Golden Cross (SMA)", "BUY", weights['SMA']*1.5, "🌟"))
        elif latest['SMA_short'] < latest['SMA_long'] and prev['SMA_short'] >= prev['SMA_long']:
            sell_score += weights['SMA'] * 1.5
            signals.append(("Death Cross (SMA)", "SELL", weights['SMA']*1.5, "⚠️"))
    
    # Bollinger Bands
    if not np.isnan(latest['BB_lower']) and not np.isnan(latest['BB_upper']):
        bb_pos = latest.get('BB_position', 0.5)
        if latest['Close'] < latest['BB_lower']:
            buy_score += weights['BB']
            signals.append(("Price below BB lower", "BUY", weights['BB'], f"BB pos={bb_pos:.2f}"))
        elif latest['Close'] > latest['BB_upper']:
            sell_score += weights['BB']
            signals.append(("Price above BB upper", "SELL", weights['BB'], f"BB pos={bb_pos:.2f}"))
        
        # BB squeeze (low volatility, potential breakout)
        if not np.isnan(latest['BB_width']) and latest['BB_width'] < 0.05:
            signals.append(("BB squeeze detected", "WATCH", 0, "Potential breakout"))
    
    # Stochastic Oscillator
    if 'Stoch_K' in latest and not np.isnan(latest['Stoch_K']):
        if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
            buy_score += weights.get('Stoch', 0.8)
            signals.append(("Stochastic oversold", "BUY", weights.get('Stoch', 0.8), f"K={latest['Stoch_K']:.1f}"))
        elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
            sell_score += weights.get('Stoch', 0.8)
            signals.append(("Stochastic overbought", "SELL", weights.get('Stoch', 0.8), f"K={latest['Stoch_K']:.1f}"))
    
    # ===== MODIFIERS (MULTIPLIERS) =====
    
    # Volume Confirmation
    vol_avg20 = df['Volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Volume'].mean()
    if vol_avg20 > 0 and latest['Volume'] > vol_avg20 * 1.5:
        volume_multiplier = 1.3
        signals.append(("High volume confirmation", "CONFIRM", 0, f"{latest['Volume']/vol_avg20:.1f}x avg"))
    elif vol_avg20 > 0 and latest['Volume'] < vol_avg20 * 0.5:
        volume_multiplier = 0.7
        signals.append(("Low volume warning", "CAUTION", 0, f"{latest['Volume']/vol_avg20:.1f}x avg"))
    
    # ADX Trend Strength
    if not np.isnan(latest['ADX']):
        if latest['ADX'] > 25:
            # Strong trend - amplify signals
            trend_multiplier = 1.0 + min((latest['ADX'] - 25) / 100, 0.5)
            signals.append((f"Strong trend", "AMPLIFY", 0, f"ADX={latest['ADX']:.1f}"))
        elif latest['ADX'] < 20:
            # Weak trend - dampen signals
            trend_multiplier = 0.6
            signals.append((f"Weak trend", "DAMPEN", 0, f"ADX={latest['ADX']:.1f}"))
    
    # Apply multipliers
    buy_score = buy_score * trend_multiplier * volume_multiplier
    sell_score = sell_score * trend_multiplier * volume_multiplier
    
    # Calculate confidence
    total_score = buy_score + sell_score
    if total_score > 0:
        confidence = ((buy_score - sell_score) / total_score) * 100
    else:
        confidence = 0.0
    
    # Determine recommendation
    net_score = buy_score - sell_score
    strong_threshold = max(weights.values()) * 2.5
    
    if net_score > strong_threshold:
        recommendation = "STRONG BUY"
    elif net_score > 0.5:
        recommendation = "BUY"
    elif net_score < -strong_threshold:
        recommendation = "STRONG SELL"
    elif net_score < -0.5:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    raw_scores = {'buy': buy_score, 'sell': sell_score, 'net': net_score}
    
    return recommendation, signals, confidence, raw_scores

# -------------------- Risk Metrics --------------------

def calculate_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.04):
    """Calculate comprehensive risk metrics"""
    returns = df['Close'].pct_change().dropna()
    
    if len(returns) < 2:
        return None
    
    # Annualized metrics
    annual_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252/len(df)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    excess_returns = returns - risk_free_rate/252
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Sortino Ratio (downside deviation only)
    downside = returns[returns < 0]
    sortino = np.sqrt(252) * excess_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Calmar Ratio
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    # Value at Risk (95% confidence)
    var_95 = returns.quantile(0.05)
    
    # Conditional VaR (Expected Shortfall)
    cvar_95 = returns[returns <= var_95].mean()
    
    return {
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'var_95': var_95,
        'cvar_95': cvar_95
    }

# -------------------- Advanced Monte Carlo --------------------

def estimate_days_to_target_advanced(df: pd.DataFrame, current_price: float,
                                     target_return: float, sims: int = 5000,
                                     max_days: int = 365):
    """
    Advanced Monte Carlo with:
    - Student's t-distribution for fat tails
    - Exponentially weighted recent returns
    - Volatility clustering
    """
    returns = df['Close'].pct_change().dropna().values
    if len(returns) < 30:
        return {'probability': 0.0, 'median_days': None, '90pct_days': None, '10pct_days': None}
    
    # Weight recent returns more (exponential decay)
    weights = np.exp(np.linspace(-2, 0, len(returns)))
    weights /= weights.sum()
    
    mu = np.average(returns, weights=weights)
    sigma = np.sqrt(np.average((returns - mu)**2, weights=weights))
    
    if sigma == 0:
        return {'probability': 0.0, 'median_days': None, '90pct_days': None, '10pct_days': None}
    
    # Use Student's t-distribution (df=5 for moderate fat tails)
    t_samples = stats.t.rvs(df=5, loc=mu, scale=sigma, size=(sims, max_days))
    
    # Simple volatility clustering
    vol_factor = np.ones((sims, max_days))
    for d in range(1, max_days):
        vol_factor[:, d] = 0.85 * vol_factor[:, d-1] + 0.15 * (1 + np.abs(t_samples[:, d-1]))
    
    t_samples *= vol_factor
    
    # Calculate price paths
    price_paths = current_price * np.cumprod(1 + t_samples, axis=1)
    threshold = current_price * (1 + target_return)
    hits = price_paths >= threshold
    
    # Find first hit day
    first_hit = np.argmax(hits, axis=1) + 1
    no_hit_mask = ~hits.any(axis=1)
    first_hit = first_hit.astype(float)
    first_hit[no_hit_mask] = np.nan
    
    valid = ~np.isnan(first_hit)
    prob_reach = valid.mean()
    median_days = float(np.nanmedian(first_hit)) if prob_reach > 0 else None
    pct90 = float(np.nanpercentile(first_hit[valid], 90)) if prob_reach > 0 else None
    pct10 = float(np.nanpercentile(first_hit[valid], 10)) if prob_reach > 0 else None
    mean_days = float(np.nanmean(first_hit)) if prob_reach > 0 else None
    
    return {
        'probability': prob_reach,
        'median_days': median_days,
        '90pct_days': pct90,
        '10pct_days': pct10,
        'mean_days': mean_days
    }

# -------------------- Backtesting --------------------

def backtest_strategy(df: pd.DataFrame, weights: dict, 
                     initial_capital: float = 10000,
                     confidence_threshold: float = 20,
                     stop_loss_pct: float = 0.05,
                     take_profit_pct: float = 0.15):
    """
    Backtest the signal strategy with transaction costs and risk management
    """
    df = df.copy()
    positions = []
    capital = initial_capital
    shares = 0
    entry_price = 0
    transaction_cost = 0.001  # 0.1% per trade
    
    buy_signals = []
    sell_signals = []
    equity_curve = []
    
    for i in range(50, len(df)):
        window = df.iloc[:i+1]
        
        # Skip if insufficient data for indicators
        if len(window) < 50:
            continue
            
        rec, _, conf, scores = rule_based_signal_v2(window, weights=weights)
        current_price = df['Close'].iloc[i]
        current_date = df.index[i]
        
        # Calculate current portfolio value
        if shares > 0:
            portfolio_value = shares * current_price
        else:
            portfolio_value = capital
        
        equity_curve.append({'date': current_date, 'value': portfolio_value})
        
        # Check stop-loss and take-profit if holding position
        if shares > 0:
            return_pct = (current_price - entry_price) / entry_price
            
            # Stop loss
            if return_pct <= -stop_loss_pct:
                capital = shares * current_price * (1 - transaction_cost)
                positions.append(('STOP_LOSS', current_date, current_price, capital, return_pct))
                sell_signals.append(i)
                shares = 0
                entry_price = 0
                continue
            
            # Take profit
            if return_pct >= take_profit_pct:
                capital = shares * current_price * (1 - transaction_cost)
                positions.append(('TAKE_PROFIT', current_date, current_price, capital, return_pct))
                sell_signals.append(i)
                shares = 0
                entry_price = 0
                continue
        
        # Buy signal
        if rec in ['BUY', 'STRONG BUY'] and shares == 0 and abs(conf) > confidence_threshold:
            shares = (capital * (1 - transaction_cost)) / current_price
            entry_price = current_price
            capital = 0
            positions.append(('BUY', current_date, current_price, shares, conf))
            buy_signals.append(i)
        
        # Sell signal
        elif rec in ['SELL', 'STRONG SELL'] and shares > 0:
            return_pct = (current_price - entry_price) / entry_price
            capital = shares * current_price * (1 - transaction_cost)
            positions.append(('SELL', current_date, current_price, capital, return_pct))
            sell_signals.append(i)
            shares = 0
            entry_price = 0
    
    # Close final position at current price
    if shares > 0:
        final_price = df['Close'].iloc[-1]
        return_pct = (final_price - entry_price) / entry_price
        capital = shares * final_price * (1 - transaction_cost)
        positions.append(('FINAL_CLOSE', df.index[-1], final_price, capital, return_pct))
    
    # Calculate performance metrics
    final_capital = capital if shares == 0 else shares * df['Close'].iloc[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Buy and hold comparison
    buy_hold_shares = (initial_capital * (1 - transaction_cost)) / df['Close'].iloc[50]
    buy_hold_final = buy_hold_shares * df['Close'].iloc[-1] * (1 - transaction_cost)
    buy_hold_return = (buy_hold_final - initial_capital) / initial_capital
    
    # Calculate win rate
    trades_with_returns = [p for p in positions if len(p) > 4 and isinstance(p[4], (int, float))]
    winning_trades = [p for p in trades_with_returns if p[4] > 0]
    win_rate = len(winning_trades) / len(trades_with_returns) if trades_with_returns else 0
    
    # Average win/loss
    wins = [p[4] for p in winning_trades]
    losses = [p[4] for p in trades_with_returns if p[4] <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    return {
        'final_capital': final_capital,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'alpha': total_return - buy_hold_return,
        'num_trades': len(positions),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'positions': positions,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'equity_curve': equity_curve
    }

# -------------------- UI Layout --------------------

st.title("📈 Enhanced Stock Dashboard v2.0 (by Sadiq)")
st.caption("Advanced technical analysis with risk metrics and strategy backtesting")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Market cap groups
group_choice = st.sidebar.radio("Market Cap Group",
                                ["Big Cap (>$10B)", "Medium Cap ($1B–$10B)", "Small Cap (<$1B)"])

if group_choice.startswith("Big"):
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA"
elif group_choice.startswith("Medium"):
    default_tickers = "AMD, ADBE, PYPL, SQ, DOCU"
else:
    default_tickers = "SOFI, HOOD, RKT, BB"

tickers_input = st.sidebar.text_area("Tickers (comma separated)", value=default_tickers, height=100)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

lookback = st.sidebar.selectbox("Lookback Period", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Data Interval", ["1d", "1wk"], index=0)

# Indicator Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Indicator Parameters")
with st.sidebar.expander("RSI Settings"):
    rsi_period = st.slider("RSI Period", 10, 30, 14)
    rsi_oversold = st.slider("RSI Oversold", 10, 40, 30)
    rsi_overbought = st.slider("RSI Overbought", 60, 85, 70)

with st.sidebar.expander("SMA Settings"):
    sma_short = st.slider("SMA Short Window", 10, 40, 20)
    sma_long = st.slider("SMA Long Window", 30, 200, 50)

# Signal Weights
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Signal Weights")
with st.sidebar.expander("Adjust Signal Weights"):
    w_rsi = st.slider("RSI weight", 0.0, 5.0, 2.0, 0.1)
    w_macd = st.slider("MACD weight", 0.0, 5.0, 1.5, 0.1)
    w_sma = st.slider("SMA weight", 0.0, 5.0, 1.0, 0.1)
    w_bb = st.slider("BB weight", 0.0, 5.0, 1.0, 0.1)
    w_stoch = st.slider("Stochastic weight", 0.0, 3.0, 0.8, 0.1)
    w_vol = st.slider("Volume weight", 0.0, 2.0, 0.5, 0.1)
    w_adx = st.slider("ADX weight", 0.0, 3.0, 1.0, 0.1)

weights = {
    'RSI': w_rsi, 
    'MACD': w_macd, 
    'SMA': w_sma, 
    'BB': w_bb, 
    'Stoch': w_stoch,
    'Volume': w_vol, 
    'ADX': w_adx
}

# Monte Carlo & Backtest Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Simulation & Backtest")
with st.sidebar.expander("Monte Carlo Settings"):
    sim_count = st.select_slider("Simulation count", options=[500, 1000, 2500, 5000, 10000], value=2500)
    max_days = st.slider("Max days for sim", 90, 730, 365, 30)

with st.sidebar.expander("Backtest Settings"):
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
    confidence_threshold = st.slider("Min confidence for trade (%)", 0, 50, 20, 5)
    stop_loss_pct = st.slider("Stop Loss (%)", 1, 20, 5, 1) / 100
    take_profit_pct = st.slider("Take Profit (%)", 5, 50, 15, 5) / 100

# Performance Settings
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Performance")
max_workers = st.sidebar.slider("Parallel workers", 1, 12, 5)
limit_batch = st.sidebar.number_input("Batch limit (max tickers)", min_value=1, max_value=50, value=10)

# Refresh Button
st.sidebar.markdown("---")
refresh_button = st.sidebar.button("🔄 Refresh Data", type="primary")

# -------------------- Fear & Greed & SPY --------------------

fg_score, fg_rating, fg_color = get_fear_greed_index()
spy_hist = get_spy_data(period=lookback, interval=interval)

# -------------------- Session State --------------------
if "data_cache" not in st.session_state:
    st.session_state["data_cache"] = {}
if "batch_errors" not in st.session_state:
    st.session_state["batch_errors"] = {}

# -------------------- Batch Fetching --------------------

def fetch_and_process(ticker):
    """Fetch and process single ticker for batch summary"""
    hist, info = get_data_optimized(ticker, period=lookback, interval=interval)
    if hist.empty:
        err_msg = info.get("_error", "Unknown error")
        return None, f"{ticker}: {err_msg}"
    
    df = calc_indicators(hist, rsi_period=rsi_period, sma_short=sma_short, sma_long=sma_long)
    rec, signals, conf, scores = rule_based_signal_v2(df, rsi_oversold=rsi_oversold,
                                                       rsi_overbought=rsi_overbought,
                                                       weights=weights)
    
    # Risk metrics
    risk = calculate_risk_metrics(df)
    
    # P/E
    pe = info.get("forwardPE") or info.get("trailingPE") or "N/A"
    
    # SPY correlation
    corr = 0.0
    try:
        if not spy_hist.empty:
            min_len = min(len(spy_hist), len(df))
            corr = df['Close'].iloc[-min_len:].corr(spy_hist['Close'].iloc[-min_len:])
            corr = 0.0 if np.isnan(corr) else corr
    except:
        corr = 0.0
    
    price_str = f"${df['Close'].iloc[-1]:.2f}"
    sharpe = risk['sharpe'] if risk else 0
    
    return {
        "Ticker": ticker,
        "Price": price_str,
        "Rec": rec,
        "Conf (%)": f"{conf:.1f}",
        "Sharpe": f"{sharpe:.2f}",
        "SPY Corr": f"{corr:.2f}",
        "P/E": f"{pe:.1f}" if isinstance(pe, (int, float)) else pe
    }, None

st.subheader("📊 Watchlist Summary")

if refresh_button or len(st.session_state["data_cache"]) == 0:
    with st.spinner("Fetching batch data..."):
        batch_results = []
        errors = []
        to_fetch = tickers[:int(limit_batch)]
        
        with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
            future_to_ticker = {executor.submit(fetch_and_process, t): t for t in to_fetch}
            for fut in as_completed(future_to_ticker):
                res, err = fut.result()
                if err:
                    errors.append(err)
                elif res:
                    batch_results.append(res)
        
        st.session_state["batch_errors"] = {e.split(":")[0]: e for e in errors}
        
        if batch_results:
            df_batch = pd.DataFrame(batch_results).sort_values(by="Conf (%)", ascending=False)
            st.dataframe(df_batch.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No batch results available.")
        
        if errors:
            with st.expander("⚠️ Errors"):
                for e in errors:
                    st.write(f"- {e}")

# -------------------- Market Overview --------------------

st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.subheader("🌍 Market Sentiment")
    c1, c2 = st.columns(2)
    with c1:
        if fg_score is not None:
            st.metric("Fear & Greed Score", fg_score)
            st.progress(fg_score / 100)
        else:
            st.metric("Fear & Greed Score", "N/A")
    with c2:
        st.write(f"**{fg_rating}**")
        st.write(fg_color)

with col2:
    st.subheader("📈 SPY")
    if not spy_hist.empty:
        spy_change = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-2]) / spy_hist['Close'].iloc[-2] * 100
        st.metric("SPY", f"${spy_hist['Close'].iloc[-1]:.2f}", f"{spy_change:+.2f}%")

with col3:
    st.subheader("📍 Select Stock")
    selected = st.selectbox("Choose ticker to analyze", options=tickers if tickers else ["AAPL"], label_visibility="collapsed")

# -------------------- Single Stock Analysis --------------------

st.markdown("---")
st.header(f"🔍 Deep Dive: {selected}")

# Fetch selected ticker
cache_key = f"{selected}_{lookback}_{interval}"
if cache_key in st.session_state["data_cache"]:
    hist, info = st.session_state["data_cache"][cache_key]
else:
    with st.spinner(f"Loading {selected}..."):
        hist, info = get_data_optimized(selected, period=lookback, interval=interval)
        st.session_state["data_cache"][cache_key] = (hist, info)

if hist.empty:
    st.error(f"❌ No data for {selected}. Reason: {info.get('_error', 'Unknown')}")
    st.stop()

# Calculate indicators
df = calc_indicators(hist, rsi_period=rsi_period, sma_short=sma_short, sma_long=sma_long)

# Validate indicators
is_valid, validation = validate_indicators(df)
if not is_valid:
    st.warning(f"⚠️ Data quality issue: {validation}")

latest = df.iloc[-1]

# Top metrics
st.subheader("💰 Key Metrics")
m1, m2, m3, m4, m5 = st.columns(5)

price_str = f"${latest['Close']:.2f}"
vol_str = f"{latest['Volume'] / 1_000_000:.2f}M"
market_cap = info.get("marketCap")
mc_str = f"${market_cap/1_000_000_000:.2f}B" if market_cap else "N/A"
pe_val = info.get("forwardPE") or info.get("trailingPE") or "N/A"
pe_str = f"{pe_val:.1f}x" if isinstance(pe_val, (int, float)) else pe_val

m1.metric("Price", price_str)
m2.metric("Volume", vol_str)
m3.metric("Market Cap", mc_str)
m4.metric("Fwd P/E", pe_str)

# SPY Correlation
corr = 0.0
if not spy_hist.empty:
    try:
        min_len = min(len(spy_hist), len(df))
        corr = df['Close'].iloc[-min_len:].corr(spy_hist['Close'].iloc[-min_len:])
        corr = 0.0 if np.isnan(corr) else corr
    except:
        corr = 0.0
m5.metric("SPY Correlation", f"{corr:.2f}")

# Risk Metrics
st.subheader("⚠️ Risk Analysis")
risk_metrics = calculate_risk_metrics(df)

if risk_metrics:
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Annual Return", f"{risk_metrics['annual_return']*100:.2f}%")
    r2.metric("Volatility", f"{risk_metrics['volatility']*100:.2f}%")
    r3.metric("Sharpe Ratio", f"{risk_metrics['sharpe']:.2f}")
    r4.metric("Sortino Ratio", f"{risk_metrics['sortino']:.2f}")
    r5.metric("Max Drawdown", f"{risk_metrics['max_drawdown']*100:.2f}%")

# -------------------- Charts --------------------

st.markdown("---")
st.subheader("📉 Price Chart with Technical Indicators")

# Create subplots
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=('Price & Indicators', 'RSI', 'MACD')
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ),
    row=1, col=1
)

# SMAs
if 'SMA_short' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_short'], mode='lines', 
                             name=f'SMA {sma_short}', line=dict(width=1, color='orange')), row=1, col=1)
if 'SMA_long' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_long'], mode='lines',
                             name=f'SMA {sma_long}', line=dict(width=1, color='blue')), row=1, col=1)

# Bollinger Bands
if 'BB_upper' in df and 'BB_lower' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines',
                             name='BB Upper', line=dict(dash='dot', width=1, color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines',
                             name='BB Lower', line=dict(dash='dot', width=1, color='gray')), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

# MACD
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='Signal', line=dict(color='red')), row=3, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram'), row=3, col=1)

fig.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=True)
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="RSI", row=2, col=1)
fig.update_yaxes(title_text="MACD", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# -------------------- Signals & Recommendation --------------------

st.markdown("---")
st.subheader("🎯 Trading Signals & Recommendation")

recommendation, signals, confidence, raw_scores = rule_based_signal_v2(
    df, 
    rsi_oversold=rsi_oversold,
    rsi_overbought=rsi_overbought,
    weights=weights
)

col1, col2 = st.columns([1, 2])

with col1:
    # Recommendation box
    color = "green" if "BUY" in recommendation else "red" if "SELL" in recommendation else "orange"
    st.markdown(f"<h2 style='color:{color}; text-align:center;'>{recommendation}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center;'>Confidence: {confidence:.1f}%</h4>", unsafe_allow_html=True)
    
    # Score breakdown
    st.metric("Buy Score", f"{raw_scores['buy']:.2f}")
    st.metric("Sell Score", f"{raw_scores['sell']:.2f}")
    st.metric("Net Score", f"{raw_scores['net']:.2f}")

with col2:
    # Signal details
    st.write("**Signal Breakdown:**")
    
    for signal_text, signal_type, weight, extra in signals:
        if signal_type == "BUY":
            emoji = "🟢"
        elif signal_type == "SELL":
            emoji = "🔴"
        elif signal_type == "CONFIRM":
            emoji = "✅"
        elif signal_type == "AMPLIFY":
            emoji = "📈"
        elif signal_type == "DAMPEN":
            emoji = "📉"
        elif signal_type == "CAUTION":
            emoji = "⚠️"
        elif signal_type == "WATCH":
            emoji = "👀"
        else:
            emoji = "⚪"
        
        display_text = f"{emoji} {signal_text}"
        if extra:
            display_text += f" ({extra})"
        if weight > 0:
            display_text += f" [w={weight:.2f}]"
        
        st.write(display_text)

# -------------------- Backtest Results --------------------

st.markdown("---")
st.subheader("📊 Strategy Backtest Performance")

with st.spinner("Running backtest..."):
    backtest_results = backtest_strategy(
        df, 
        weights=weights,
        initial_capital=initial_capital,
        confidence_threshold=confidence_threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )

b1, b2, b3, b4, b5 = st.columns(5)

b1.metric("Final Capital", f"${backtest_results['final_capital']:.2f}")
b2.metric("Strategy Return", f"{backtest_results['total_return']*100:.2f}%")
b3.metric("Buy & Hold Return", f"{backtest_results['buy_hold_return']*100:.2f}%")
b4.metric("Alpha", f"{backtest_results['alpha']*100:.2f}%", 
          delta=f"{backtest_results['alpha']*100:.2f}%")
b5.metric("Number of Trades", backtest_results['num_trades'])

b6, b7, b8 = st.columns(3)
b6.metric("Win Rate", f"{backtest_results['win_rate']*100:.1f}%")
b7.metric("Avg Win", f"{backtest_results['avg_win']*100:.2f}%")
b8.metric("Avg Loss", f"{backtest_results['avg_loss']*100:.2f}%")

# Equity Curve
if backtest_results['equity_curve']:
    st.subheader("📈 Equity Curve")
    equity_df = pd.DataFrame(backtest_results['equity_curve'])
    
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df['date'],
        y=equity_df['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='green', width=2)
    ))
    
    # Add buy/sell markers
    if backtest_results['buy_signals']:
        buy_dates = [df.index[i] for i in backtest_results['buy_signals']]
        buy_prices = [df['Close'].iloc[i] for i in backtest_results['buy_signals']]
        fig_equity.add_trace(go.Scatter(
            x=buy_dates,
            y=[equity_df[equity_df['date'] == d]['value'].iloc[0] if len(equity_df[equity_df['date'] == d]) > 0 else 0 for d in buy_dates],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    if backtest_results['sell_signals']:
        sell_dates = [df.index[i] for i in backtest_results['sell_signals']]
        fig_equity.add_trace(go.Scatter(
            x=sell_dates,
            y=[equity_df[equity_df['date'] == d]['value'].iloc[0] if len(equity_df[equity_df['date'] == d]) > 0 else 0 for d in sell_dates],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    fig_equity.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_equity, use_container_width=True)

# Trade History
with st.expander("📋 View Trade History"):
    if backtest_results['positions']:
        trades_df = pd.DataFrame(backtest_results['positions'], 
                                columns=['Action', 'Date', 'Price', 'Value/Shares', 'Return/Conf'])
        st.dataframe(trades_df, use_container_width=True)

# -------------------- Monte Carlo Projections --------------------

st.markdown("---")
st.subheader("🎲 Monte Carlo Price Target Projections")

targets = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
sim_results = []
current_price = float(latest['Close'])

with st.spinner("Running Monte Carlo simulations..."):
    for t in targets:
        res = estimate_days_to_target_advanced(
            df, 
            current_price, 
            target_return=t,
            sims=sim_count,
            max_days=max_days
        )
        sim_results.append({
            "Target (%)": int(t*100),
            "Target Price": f"${current_price * (1+t):.2f}",
            "Probability (%)": f"{res['probability']*100:.1f}",
            "Median Days": res['median_days'],
            "90th Pctl Days": res['90pct_days'],
            "10th Pctl Days": res['10pct_days']
        })

mc_df = pd.DataFrame(sim_results)
st.dataframe(mc_df, use_container_width=True)

st.info(f"💡 Based on {sim_count:,} simulations with {max_days} day horizon using Student's t-distribution")

# -------------------- Footer --------------------

st.markdown("---")
st.subheader("📝 Notes & Disclaimers")

st.write("""
### Improvements in v2.0:
- ✅ **Fixed signal weighting logic** - Proper separation of directional signals and modifiers
- ✅ **Enhanced Monte Carlo** - Student's t-distribution for fat tails, volatility clustering
- ✅ **Comprehensive risk metrics** - Sharpe, Sortino, Max DD, Calmar, VaR
- ✅ **Full backtesting** - Transaction costs, stop-loss, take-profit, win rate analysis
- ✅ **Better data validation** - Indicator quality checks
- ✅ **Optimized performance** - Selective info fetching, parallel batch processing

### Limitations & Disclaimers:
- ⚠️ **NOT FINANCIAL ADVICE** - This tool is for educational purposes only
- ⚠️ Past performance does not guarantee future results
- ⚠️ Backtest results may not reflect actual trading due to slippage, market impact
- ⚠️ Always do your own research and consult a financial advisor
- ⚠️ Consider paper trading before using real capital

### Recommended Next Steps:
1. Test different weight configurations to optimize for your risk tolerance
2. Paper trade the strategy for at least 3 months before deploying real capital
3. Consider adding fundamental analysis (earnings, revenue growth, debt ratios)
4. Implement position sizing and portfolio-level risk management
5. Add real-time alerts for signal generation
6. Consider machine learning models trained on your optimized weights
""")

st.markdown("---")
st.caption("Enhanced Stock Dashboard v2.0 (by Sadiq) | Built with Streamlit | Data: Yahoo Finance")
