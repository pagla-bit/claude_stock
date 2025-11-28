"""
Enhanced Streamlit Stock Dashboard v2.4 - COMPLETE IMPLEMENTATION
"""
# ==================== IMPORTS ====================
import logging
import sys
import gc
import time
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import streamlit as st
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam

# Sentiment analysis dependencies
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

# Google News RSS parsing
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# BeautifulSoup for web scraping
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

# ==================== CONFIGURATION ====================
class DashboardConfig:
    """Centralized configuration for all dashboard parameters"""
    
    # Caching settings
    CACHE_TTL = 3600  # 1 hour
    NEWS_CACHE_TTL = 1800  # 30 minutes
    
    # API settings
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # Data settings
    MIN_DATA_POINTS = 50
    DEFAULT_LOOKBACK = "1y"
    DEFAULT_INTERVAL = "1d"
    
    # ML settings
    ML_LOOKBACK = 60
    SEQUENCE_LENGTH = 60
    MONTE_CARLO_SIMS = 1000
    MAX_MC_DAYS = 365
    
    # News settings
    MAX_NEWS_ITEMS = 10
    NEWS_SOURCES = ["Finviz", "Google News"]
    
    # Risk management
    RISK_FREE_RATE = 0.04
    DEFAULT_STOP_LOSS = 0.05
    DEFAULT_TAKE_PROFIT = 0.15
    
    # Technical indicators
    DEFAULT_RSI_PERIOD = 14
    DEFAULT_RSI_OVERSOLD = 30
    DEFAULT_RSI_OVERBOUGHT = 70
    DEFAULT_SMA_SHORT = 20
    DEFAULT_SMA_LONG = 50
    
    # Signal weights
    DEFAULT_WEIGHTS = {
        'RSI': 2.0, 'MACD': 1.5, 'SMA': 1.0, 
        'BB': 1.0, 'Stoch': 0.8, 'Volume': 0.5, 'ADX': 1.0
    }

# ==================== LOGGING SETUP ====================
def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dashboard_errors.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==================== ERROR HANDLING & RETRY MECHANISMS ====================
class DashboardError(Exception):
    """Custom exception for dashboard errors"""
    pass

class DataFetchError(DashboardError):
    """Raised when data fetching fails"""
    pass

class MLTrainingError(DashboardError):
    """Raised when ML model training fails"""
    pass

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"All retries failed for {func.__name__}: {str(e)}")
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}. Retrying in {delay}s: {str(e)}")
            time.sleep(delay)

# ==================== MEMORY MANAGEMENT ====================
class MemoryManager:
    """Manage memory usage and cleanup"""
    
    @staticmethod
    def clear_cache():
        """Clear various caches to free memory"""
        gc.collect()
        
    @staticmethod
    def cleanup_ml_models():
        """Clean up ML models and free GPU memory"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

# ==================== DATA FETCHING MODULE ====================
class DataFetcher:
    """Handle all data fetching operations with error handling and retries"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; StreamlitApp/1.0)'
        })
    
    @st.cache_data(ttl=3600)
    def get_stock_data(_self, ticker: str, period: str = "1y", interval: str = "1d") -> Tuple[pd.DataFrame, Dict]:
        """Fetch stock data with comprehensive error handling"""
        def fetch_data():
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
                
                if hist.empty:
                    raise DataFetchError(f"No data returned for {ticker}")
                
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing = set(required_cols) - set(hist.columns)
                if missing:
                    raise DataFetchError(f"Missing columns {missing} for {ticker}")
                
                if len(hist) < _self.config.MIN_DATA_POINTS:
                    raise DataFetchError(f"Insufficient data points for {ticker}")
                
                # Fetch essential info with timeout
                info = {}
                try:
                    raw_info = tk.info
                    info = {
                        'forwardPE': raw_info.get('forwardPE'),
                        'trailingPE': raw_info.get('trailingPE'),
                        'marketCap': raw_info.get('marketCap'),
                        'shortName': raw_info.get('shortName', ticker)
                    }
                except Exception as e:
                    logger.warning(f"Could not fetch info for {ticker}: {str(e)}")
                    info = {'shortName': ticker}
                
                return hist, info
                
            except Exception as e:
                raise DataFetchError(f"Failed to fetch data for {ticker}: {str(e)}")
        
        return retry_with_backoff(fetch_data, max_retries=_self.config.MAX_RETRIES)
    
    @st.cache_data(ttl=3600)
    def get_spy_data(_self) -> pd.DataFrame:
        """Fetch SPY data for correlation analysis"""
        try:
            hist, _ = _self.get_stock_data("SPY", period=_self.config.DEFAULT_LOOKBACK)
            return hist
        except Exception as e:
            logger.error(f"Failed to fetch SPY data: {str(e)}")
            return pd.DataFrame()
    
    def get_fear_greed_index(self) -> Tuple[Optional[float], str, str]:
        """Fetch Fear & Greed Index with fallback"""
        def fetch_fgi():
            base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
            
            for days_back in range(0, 3):
                target_date = (date.today() - timedelta(days=days_back)).isoformat()
                try:
                    resp = self.session.get(base_url + target_date, timeout=self.config.REQUEST_TIMEOUT)
                    resp.raise_for_status()
                    data = resp.json()
                    fg = data.get("fear_and_greed", {})
                    score = fg.get("score")
                    rating = fg.get("rating", "N/A")
                    
                    if score is not None:
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
                except Exception as e:
                    logger.debug(f"FGI fetch failed for {target_date}: {str(e)}")
                    continue
            
            return None, "N/A", "N/A"
        
        return retry_with_backoff(fetch_fgi, max_retries=self.config.MAX_RETRIES)

# ==================== TECHNICAL INDICATORS MODULE ====================
class TechnicalIndicators:
    """Calculate and validate technical indicators"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
    
    def calculate_indicators(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Calculate all technical indicators with validation"""
        df = df.copy()
        
        # Use provided parameters or defaults
        rsi_period = params.get('rsi_period', self.config.DEFAULT_RSI_PERIOD)
        sma_short = params.get('sma_short', self.config.DEFAULT_SMA_SHORT)
        sma_long = params.get('sma_long', self.config.DEFAULT_SMA_LONG)
        
        try:
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
            ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
            ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema_fast - ema_slow
            df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
            
            # Bollinger Bands
            bb_period = 20
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
            df["ATR"] = tr.ewm(span=14, adjust=False).mean()
            df["ATR_pct"] = (df["ATR"] / df["Close"]) * 100
            
            # ADX
            plus_dm = df["High"].diff()
            minus_dm = -df["Low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            tr_smooth = tr.ewm(span=14).mean()
            plus_di = 100 * (plus_dm.ewm(span=14).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.ewm(span=14).mean() / tr_smooth)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            df["ADX"] = dx.ewm(span=14).mean()
            df["DI_plus"] = plus_di
            df["DI_minus"] = minus_di
            
            # Stochastic Oscillator
            low_14 = df["Low"].rolling(14).min()
            high_14 = df["High"].rolling(14).max()
            df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
            df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {str(e)}")
            raise MLTrainingError(f"Technical indicator calculation failed: {str(e)}")
    
    def validate_indicators(self, df: pd.DataFrame, min_valid_pct: float = 0.7) -> Tuple[bool, Dict]:
        """Validate that indicators have sufficient valid data"""
        required = ['RSI', 'MACD', 'ATR', 'ADX']
        validation_results = {}
        
        for col in required:
            if col in df:
                valid_pct = df[col].notna().mean()
                validation_results[col] = valid_pct
                if valid_pct < min_valid_pct:
                    return False, f"{col} has only {valid_pct*100:.1f}% valid data"
        
        return True, validation_results

    def rule_based_signal_v2(self, df: pd.DataFrame, rsi_oversold=30, rsi_overbought=70, weights=None):
        """Enhanced signal generation with proper modifiers"""
        if weights is None:
            weights = self.config.DEFAULT_WEIGHTS
        
        if len(df) < 3:
            return "HOLD", [], 0.0, {'buy': 0, 'sell': 0}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        buy_score = 0.0
        sell_score = 0.0
        trend_multiplier = 1.0
        volume_multiplier = 1.0
        signals = []
        
        # RSI
        if not np.isnan(latest['RSI']):
            if latest['RSI'] < rsi_oversold:
                buy_score += weights['RSI']
                signals.append(("RSI oversold", "BUY", weights['RSI'], f"RSI={latest['RSI']:.1f}"))
            elif latest['RSI'] > rsi_overbought:
                sell_score += weights['RSI']
                signals.append(("RSI overbought", "SELL", weights['RSI'], f"RSI={latest['RSI']:.1f}"))
        
        # MACD
        if not np.isnan(latest['MACD']) and not np.isnan(prev['MACD']):
            if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
                buy_score += weights['MACD']
                signals.append(("MACD bullish crossover", "BUY", weights['MACD'], ""))
            elif (prev['MACD'] > prev['MACD_signal']) and (latest['MACD'] < latest['MACD_signal']):
                sell_score += weights['MACD']
                signals.append(("MACD bearish crossover", "SELL", weights['MACD'], ""))
        
        # SMA
        if not np.isnan(latest['SMA_long']):
            if latest['Close'] > latest['SMA_long']:
                buy_score += weights['SMA']
                signals.append(("Price above long SMA", "BUY", weights['SMA'], ""))
            else:
                sell_score += weights['SMA']
                signals.append(("Price below long SMA", "SELL", weights['SMA'], ""))
        
        # Bollinger Bands
        if not np.isnan(latest['BB_lower']) and not np.isnan(latest['BB_upper']):
            if latest['Close'] < latest['BB_lower']:
                buy_score += weights['BB']
                signals.append(("Price below BB lower", "BUY", weights['BB'], ""))
            elif latest['Close'] > latest['BB_upper']:
                sell_score += weights['BB']
                signals.append(("Price above BB upper", "SELL", weights['BB'], ""))
        
        # Stochastic
        if 'Stoch_K' in latest and not np.isnan(latest['Stoch_K']):
            if latest['Stoch_K'] < 20:
                buy_score += weights.get('Stoch', 0.8)
                signals.append(("Stochastic oversold", "BUY", weights.get('Stoch', 0.8), ""))
            elif latest['Stoch_K'] > 80:
                sell_score += weights.get('Stoch', 0.8)
                signals.append(("Stochastic overbought", "SELL", weights.get('Stoch', 0.8), ""))
        
        # Volume
        vol_avg20 = df['Volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Volume'].mean()
        if vol_avg20 > 0 and latest['Volume'] > vol_avg20 * 1.5:
            volume_multiplier = 1.3
            signals.append(("High volume", "CONFIRM", 0, ""))
        
        # ADX
        if not np.isnan(latest['ADX']):
            if latest['ADX'] > 25:
                trend_multiplier = 1.0 + min((latest['ADX'] - 25) / 100, 0.5)
                signals.append(("Strong trend", "AMPLIFY", 0, f"ADX={latest['ADX']:.1f}"))
            elif latest['ADX'] < 20:
                trend_multiplier = 0.6
                signals.append(("Weak trend", "DAMPEN", 0, f"ADX={latest['ADX']:.1f}"))
        
        buy_score = buy_score * trend_multiplier * volume_multiplier
        sell_score = sell_score * trend_multiplier * volume_multiplier
        
        total_score = buy_score + sell_score
        confidence = ((buy_score - sell_score) / total_score * 100) if total_score > 0 else 0.0
        
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

    def calculate_risk_metrics(self, df: pd.DataFrame, risk_free_rate: float = 0.04):
        """Calculate comprehensive risk metrics"""
        returns = df['Close'].pct_change().dropna()
        if len(returns) < 2:
            return None
        
        annual_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252/len(df)) - 1
        volatility = returns.std() * np.sqrt(252)
        excess_returns = returns - risk_free_rate/252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        downside = returns[returns < 0]
        sortino = np.sqrt(252) * excess_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        var_95 = returns.quantile(0.05)
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

# ==================== NEWS FETCHING MODULE ====================
class NewsFetcher:
    """Handle news fetching from multiple sources"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=1800)
    def fetch_finviz_news(_self, ticker: str) -> List[Dict]:
        """Fetch news from Finviz with error handling"""
        if not BEAUTIFULSOUP_AVAILABLE:
            logger.warning("BeautifulSoup not available for Finviz news")
            return []
        
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
            response = _self.session.get(url, timeout=_self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_table = soup.find('table', {'id': 'news-table'})
            
            if not news_table:
                return []
            
            news_list = []
            current_date = None
            
            for row in news_table.find_all('tr')[:_self.config.MAX_NEWS_ITEMS]:
                td_timestamp = row.find('td', {'align': 'right', 'width': '130'})
                td_content = row.find('td', {'align': 'left'})
                
                if not td_timestamp or not td_content:
                    continue
                
                timestamp_text = td_timestamp.get_text().strip()
                timestamp_parts = timestamp_text.split()
                
                if len(timestamp_parts) == 2:
                    current_date = timestamp_parts[0]
                    time_str = timestamp_parts[1]
                else:
                    time_str = timestamp_parts[0]
                
                link_tag = td_content.find('a')
                if link_tag:
                    title = link_tag.get_text().strip()
                    link = link_tag.get('href', '')
                    
                    source_span = td_content.find('span', {'class': 'news-link-right'})
                    source = source_span.get_text().strip() if source_span else 'Finviz'
                    
                    if link and not link.startswith('http'):
                        link = 'https://finviz.com/' + link
                    
                    news_list.append({
                        'Date': current_date,
                        'Time': time_str,
                        'Source': source,
                        'Title': title,
                        'Link': link
                    })
            
            return news_list
            
        except Exception as e:
            logger.error(f"Finviz news fetch failed for {ticker}: {str(e)}")
            return []
    
    @st.cache_data(ttl=1800)
    def fetch_google_news(_self, ticker: str) -> List[Dict]:
        """Fetch news from Google News RSS"""
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available for Google News")
            return []
        
        try:
            query = f"{ticker} stock"
            rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
            
            response = _self.session.get(rss_url, timeout=_self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            if not feed.entries:
                return []
            
            news_list = []
            
            for entry in feed.entries[:_self.config.MAX_NEWS_ITEMS]:
                published = entry.get('published_parsed', None)
                if published:
                    dt = datetime(*published[:6])
                    date_str = dt.strftime('%b-%d-%y')
                    time_str = dt.strftime('%I:%M%p')
                else:
                    date_str = 'Today'
                    time_str = 'N/A'
                
                title = entry.get('title', 'No title')
                link = entry.get('link', '#')
                
                source = 'Google News'
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    if len(parts) == 2:
                        title = parts[0]
                        source = parts[1]
                
                news_list.append({
                    'Date': date_str,
                    'Time': time_str,
                    'Source': source,
                    'Title': title,
                    'Link': link
                })
            
            return news_list
            
        except Exception as e:
            logger.error(f"Google News fetch failed for {ticker}: {str(e)}")
            return []
    
    def get_news(self, ticker: str, source: str) -> List[Dict]:
        """Get news from specified source"""
        if source == "Finviz":
            return self.fetch_finviz_news(ticker)
        elif source == "Google News":
            return self.fetch_google_news(ticker)
        else:
            return []

# ==================== SENTIMENT ANALYSIS MODULE ====================
class SentimentAnalyzer:
    """Handle sentiment analysis with FinBERT"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
    
    def load_model(self):
        """Load FinBERT model with error handling"""
        if self._model_loaded:
            return self.tokenizer, self.model
        
        if not FINBERT_AVAILABLE:
            logger.warning("FinBERT dependencies not available")
            return None, None
        
        try:
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            self._model_loaded = True
            logger.info("FinBERT model loaded successfully")
            return self.tokenizer, self.model
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {str(e)}")
            return None, None
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment of text using FinBERT"""
        tokenizer, model = self.load_model()
        if tokenizer is None or model is None:
            return "N/A", 0.0
        
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            negative = predictions[0][0].item()
            neutral = predictions[0][1].item()
            positive = predictions[0][2].item()
            
            compound = positive - negative
            
            if positive > negative and positive > neutral:
                label = "Positive"
            elif negative > positive and negative > neutral:
                label = "Negative"
            else:
                label = "Neutral"
            
            return label, compound
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return "N/A", 0.0
    
    def analyze_news_batch(self, news_data: List[Dict], use_bert: bool = True) -> List[Dict]:
        """Analyze sentiment for a batch of news items"""
        if not news_data:
            return []
        
        if use_bert:
            tokenizer, model = self.load_model()
            bert_available = tokenizer is not None and model is not None
        else:
            bert_available = False
        
        # Analyze each news item
        for item in news_data:
            title = item['Title']
            
            if bert_available and use_bert:
                label, score = self.analyze_sentiment(title)
                
                if score >= 0.05:
                    final_label = 'Positive'
                    final_emoji = '🟢'
                    final_color = 'green'
                elif score <= -0.05:
                    final_label = 'Negative'
                    final_emoji = '🔴'
                    final_color = 'red'
                else:
                    final_label = 'Neutral'
                    final_emoji = '🟡'
                    final_color = 'orange'
                
                item.update({
                    'sentiment_label': final_label,
                    'sentiment_score': score,
                    'sentiment_emoji': final_emoji,
                    'sentiment_color': final_color,
                    'bert_score': score,
                    'bert_label': label,
                    'bert_available': True
                })
            else:
                # Default to neutral if FinBERT not available or not used
                item.update({
                    'sentiment_label': 'N/A',
                    'sentiment_score': 0.0,
                    'sentiment_emoji': '⚪',
                    'sentiment_color': 'gray',
                    'bert_available': False
                })
        
        return news_data

# ==================== MAIN DASHBOARD CLASS ====================
class EnhancedStockDashboard:
    """Main dashboard class integrating all modules"""
    
    def __init__(self):
        self.config = DashboardConfig()
        self.data_fetcher = DataFetcher(self.config)
        self.news_fetcher = NewsFetcher(self.config)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_indicators = TechnicalIndicators(self.config)
        self.memory_manager = MemoryManager()
        
        # Initialize session state
        if "data_cache" not in st.session_state:
            st.session_state["data_cache"] = {}
        if "news_cache" not in st.session_state:
            st.session_state["news_cache"] = {}
    
    def setup_ui(self):
        """Setup the Streamlit UI"""
        st.set_page_config(layout="wide", page_title="Enhanced Stock Dashboard v2.4")
        st.title("📈 Enhanced Stock Dashboard v2.4 (Improved)")
        st.caption("Advanced technical analysis with robust error handling and improved performance")
    
    def render_sidebar(self):
        """Render the sidebar configuration"""
        st.sidebar.header("⚙️ Configuration")
        
        # Simplified ticker input (removed market cap groups)
        st.sidebar.subheader("📊 Stock Selection")
        default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA"
        tickers_input = st.sidebar.text_area(
            "Tickers (comma separated)", 
            value=default_tickers, 
            height=100,
            help="Enter stock tickers separated by commas"
        )
        
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        # Data settings
        st.sidebar.subheader("📈 Data Settings")
        lookback = st.sidebar.selectbox("Lookback Period", ["6mo", "1y", "2y", "5y"], index=1)
        interval = st.sidebar.selectbox("Data Interval", ["1d", "1wk"], index=0)
        
        # Sentiment analysis
        st.sidebar.subheader("🤖 Sentiment Analysis")
        use_finbert = st.sidebar.checkbox(
            "Use FinBERT (slower, more accurate)", 
            value=False,
            help="FinBERT is more accurate but takes ~10 seconds"
        )
        
        # Technical indicators
        st.sidebar.subheader("📊 Indicator Parameters")
        with st.sidebar.expander("RSI Settings"):
            rsi_period = st.slider("RSI Period", 10, 30, self.config.DEFAULT_RSI_PERIOD)
            rsi_oversold = st.slider("RSI Oversold", 10, 40, self.config.DEFAULT_RSI_OVERSOLD)
            rsi_overbought = st.slider("RSI Overbought", 60, 85, self.config.DEFAULT_RSI_OVERBOUGHT)
        
        # Signal weights
        st.sidebar.subheader("⚖️ Signal Weights")
        with st.sidebar.expander("Adjust Signal Weights"):
            weights = {}
            for key, default in self.config.DEFAULT_WEIGHTS.items():
                weights[key] = st.slider(f"{key} weight", 0.0, 5.0, default, 0.1)
        
        st.sidebar.markdown("---")
        refresh_btn = st.sidebar.button("🔄 Refresh All Data", type="primary")
        
        return {
            'tickers': tickers,
            'lookback': lookback,
            'interval': interval,
            'use_finbert': use_finbert,
            'rsi_params': {
                'period': rsi_period,
                'oversold': rsi_oversold,
                'overbought': rsi_overbought
            },
            'weights': weights,
            'refresh': refresh_btn
        }
    
    def render_market_overview(self, tickers: List[str]):
        """Render market overview section"""
        try:
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.subheader("🌍 Market Sentiment")
                fg_score, fg_rating, fg_color = self.data_fetcher.get_fear_greed_index()
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
                spy_data = self.data_fetcher.get_spy_data()
                if not spy_data.empty:
                    spy_change = (spy_data['Close'].iloc[-1] - spy_data['Close'].iloc[-2]) / spy_data['Close'].iloc[-2] * 100
                    st.metric("SPY", f"${spy_data['Close'].iloc[-1]:.2f}", f"{spy_change:+.2f}%")
                else:
                    st.metric("SPY", "N/A")
            
            with col3:
                st.subheader("📍 Select Stock")
                if tickers:
                    selected = st.selectbox(
                        "Choose ticker to analyze", 
                        options=tickers, 
                        label_visibility="collapsed"
                    )
                    return selected
                else:
                    st.warning("Please enter tickers in the sidebar")
                    return None
            
        except Exception as e:
            logger.error(f"Market overview rendering failed: {str(e)}")
            st.error("❌ Failed to load market overview data")
            return None
    
    def render_key_metrics(self, df: pd.DataFrame, info: Dict, spy_data: pd.DataFrame):
        """Render key stock metrics"""
        st.subheader("💰 Key Metrics")
        
        latest = df.iloc[-1]
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

        # Calculate correlation with SPY
        corr = 0.0
        if not spy_data.empty:
            try:
                min_len = min(len(spy_data), len(df))
                corr = df['Close'].iloc[-min_len:].corr(spy_data['Close'].iloc[-min_len:])
                corr = 0.0 if np.isnan(corr) else corr
            except:
                corr = 0.0
        m5.metric("SPY Correlation", f"{corr:.2f}")
    
    def render_risk_metrics(self, df: pd.DataFrame):
        """Render risk metrics"""
        st.subheader("⚠️ Risk Analysis")
        risk_metrics = self.technical_indicators.calculate_risk_metrics(df)

        if risk_metrics:
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Annual Return", f"{risk_metrics['annual_return']*100:.2f}%")
            r2.metric("Volatility", f"{risk_metrics['volatility']*100:.2f}%")
            r3.metric("Sharpe Ratio", f"{risk_metrics['sharpe']:.2f}")
            r4.metric("Sortino Ratio", f"{risk_metrics['sortino']:.2f}")
            r5.metric("Max Drawdown", f"{risk_metrics['max_drawdown']*100:.2f}%")
    
    def render_price_charts(self, df: pd.DataFrame, settings: Dict):
        """Render price charts with technical indicators"""
        st.markdown("---")
        st.subheader("📉 Price Chart with Technical Indicators")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                          row_heights=[0.5, 0.25, 0.25], subplot_titles=('Price & Indicators', 'RSI', 'MACD'))

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

        if 'SMA_short' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_short'], mode='lines', name=f'SMA {settings["rsi_params"]["period"]}', line=dict(width=1, color='orange')), row=1, col=1)
        if 'SMA_long' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_long'], mode='lines', name=f'SMA 50', line=dict(width=1, color='blue')), row=1, col=1)
        if 'BB_upper' in df and 'BB_lower' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB Upper', line=dict(dash='dot', width=1, color='gray')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB Lower', line=dict(dash='dot', width=1, color='gray')), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=settings['rsi_params']['overbought'], line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=settings['rsi_params']['oversold'], line_dash="dash", line_color="green", row=2, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='Signal', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram'), row=3, col=1)

        fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_rule_based_signals(self, df: pd.DataFrame, settings: Dict):
        """Render rule-based trading signals"""
        st.markdown("---")
        st.subheader("🎯 Rule-Based Trading Signals")

        recommendation, signals, confidence, raw_scores = self.technical_indicators.rule_based_signal_v2(
            df, 
            rsi_oversold=settings['rsi_params']['oversold'], 
            rsi_overbought=settings['rsi_params']['overbought'], 
            weights=settings['weights']
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            color = "green" if "BUY" in recommendation else "red" if "SELL" in recommendation else "orange"
            st.markdown(f"<h2 style='color:{color}; text-align:center;'>{recommendation}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center;'>Confidence: {confidence:.1f}%</h4>", unsafe_allow_html=True)
            st.metric("Buy Score", f"{raw_scores['buy']:.2f}")
            st.metric("Sell Score", f"{raw_scores['sell']:.2f}")
            st.metric("Net Score", f"{raw_scores['net']:.2f}")

        with col2:
            st.write("**Signal Breakdown:**")
            for signal_text, signal_type, weight, extra in signals:
                emoji = {"BUY": "🟢", "SELL": "🔴", "CONFIRM": "✅", "AMPLIFY": "📈", "DAMPEN": "📉"}.get(signal_type, "⚪")
                display_text = f"{emoji} {signal_text}"
                if extra:
                    display_text += f" ({extra})"
                if weight > 0:
                    display_text += f" [w={weight:.2f}]"
                st.write(display_text)
    
    def render_news_section(self, ticker: str, settings: Dict):
        """Render news section with sentiment analysis"""
        st.markdown("---")
        st.header(f"📰 Latest News & Sentiment for {ticker}")

        # News source selection
        col_source, col_refresh = st.columns([3, 1])
        with col_source:
            news_source = st.selectbox(
                "📡 News Source:",
                options=self.config.NEWS_SOURCES,
                index=0,
                key="news_source_selector"
            )
        with col_refresh:
            st.write("")
            if st.button("🔄 Refresh News", key="refresh_news", type="secondary"):
                # Clear news cache for this ticker
                cache_key = f"news_{ticker}_{news_source}"
                if cache_key in st.session_state["news_cache"]:
                    del st.session_state["news_cache"][cache_key]
                st.rerun()

        # Fetch and display news
        cache_key = f"news_{ticker}_{news_source}"
        if cache_key in st.session_state["news_cache"]:
            news_data = st.session_state["news_cache"][cache_key]
        else:
            with st.spinner(f"Fetching news from {news_source}..."):
                news_data = self.news_fetcher.get_news(ticker, news_source)
                # Analyze sentiment
                if news_data:
                    news_data = self.sentiment_analyzer.analyze_news_batch(news_data, settings['use_finbert'])
                st.session_state["news_cache"][cache_key] = news_data

        if news_data:
            st.subheader(f"📰 Latest News from {news_source}")
            
            # Create DataFrame for display
            news_display = []
            for item in news_data:
                news_display.append({
                    'Time': f"{item.get('Date', 'N/A')} {item.get('Time', 'N/A')}",
                    'Headline': item['Title'],
                    'Source': item['Source'],
                    'Sentiment': f"{item.get('sentiment_emoji', '⚪')} {item.get('sentiment_label', 'N/A')}",
                    'Score': f"{item.get('sentiment_score', 0):.2f}"
                })
            
            news_df = pd.DataFrame(news_display)
            st.dataframe(news_df, use_container_width=True, hide_index=True)
            
            # Sentiment summary
            if len(news_data) > 0:
                sentiment_scores = [item.get('sentiment_score', 0) for item in news_data]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                positive_count = sum(1 for s in sentiment_scores if s >= 0.05)
                neutral_count = sum(1 for s in sentiment_scores if -0.05 < s < 0.05)
                negative_count = sum(1 for s in sentiment_scores if s <= -0.05)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🟢 Positive", positive_count)
                col2.metric("🟡 Neutral", neutral_count)
                col3.metric("🔴 Negative", negative_count)
                col4.metric("Avg Score", f"{avg_sentiment:.2f}")
        else:
            st.info(f"📭 No recent news available from {news_source}")
    
    def run(self):
        """Main method to run the dashboard"""
        try:
            self.setup_ui()
            settings = self.render_sidebar()
            
            if not settings['tickers']:
                st.warning("Please enter at least one stock ticker in the sidebar")
                return
            
            selected_ticker = self.render_market_overview(settings['tickers'])
            
            if not selected_ticker:
                return

            # Analyze selected stock
            with st.spinner(f"Analyzing {selected_ticker}..."):
                self.analyze_stock(selected_ticker, settings)
            
            # Cleanup on refresh
            if settings['refresh']:
                self.memory_manager.clear_cache()
                st.rerun()
                
        except Exception as e:
            logger.error(f"Dashboard runtime error: {str(e)}")
            st.error("❌ A critical error occurred. Please refresh the page.")
    
    def analyze_stock(self, ticker: str, settings: Dict):
        """Analyze a single stock"""
        try:
            # Fetch data
            cache_key = f"{ticker}_{settings['lookback']}_{settings['interval']}"
            
            if cache_key in st.session_state["data_cache"] and not settings['refresh']:
                hist, info = st.session_state["data_cache"][cache_key]
            else:
                hist, info = self.data_fetcher.get_stock_data(
                    ticker, 
                    period=settings['lookback'], 
                    interval=settings['interval']
                )
                st.session_state["data_cache"][cache_key] = (hist, info)
            
            if hist.empty:
                st.error(f"❌ No data available for {ticker}")
                return
            
            # Calculate indicators
            df = self.technical_indicators.calculate_indicators(hist, **settings['rsi_params'])
            is_valid, validation = self.technical_indicators.validate_indicators(df)
            if not is_valid:
                st.warning(f"⚠️ Data quality issue: {validation}")
            
            # Render analysis sections
            st.header(f"🔍 Deep Dive: {ticker} ({info.get('shortName', ticker)})")
            
            # Get SPY data for correlation
            spy_data = self.data_fetcher.get_spy_data()
            
            # Render all sections
            self.render_key_metrics(df, info, spy_data)
            self.render_risk_metrics(df)
            self.render_price_charts(df, settings)
            self.render_rule_based_signals(df, settings)
            self.render_news_section(ticker, settings)
            
        except DataFetchError as e:
            st.error(f"❌ Data fetch error for {ticker}: {str(e)}")
        except Exception as e:
            logger.error(f"Stock analysis failed for {ticker}: {str(e)}")
            st.error(f"❌ Analysis failed for {ticker}. Please try again.")

# ==================== APPLICATION ENTRY POINT ====================
def main():
    """Main application entry point"""
    try:
        dashboard = EnhancedStockDashboard()
        dashboard.run()
    except Exception as e:
        st.error("""
        ❌ Critical application error. 
        
        Please try the following:
        1. Refresh the page
        2. Check your internet connection
        3. Reduce the number of tickers being analyzed
        """)

if __name__ == "__main__":
    main()
