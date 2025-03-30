import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas_ta as ta
from pypfopt import EfficientFrontier, risk_models, expected_returns
from typing import Dict
import tensorflow as tf
import warnings
import os
from sklearn.preprocessing import MinMaxScaler

# Check for API key
try:
    from config import ALPHA_VANTAGE_API_KEY
except ImportError:
    ALPHA_VANTAGE_API_KEY = None
    st.warning("⚠️ Please set your Alpha Vantage API key in config.py")

# Comprehensive warning filters
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure TensorFlow logging and disable deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Use compat.v1 for logging

# Enable eager execution for model training
tf.compat.v1.enable_eager_execution()

# Reset the default graph
tf.compat.v1.reset_default_graph()

# Define Keras components using tf.keras
Sequential = tf.keras.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam

# Enhanced Technical Analysis
class EnhancedTechnicalAnalysis:
    """Robust technical analysis with validation and advanced features"""
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.model = self._load_deep_learning_model()
        
        # Initialize status tracking
        self.status = {
            "model_loaded": False,
            "features_calculated": False,
            "predictions_generated": False
        }
        
        # Log initialization status
        self._log_initialization_status()

    def _load_deep_learning_model(self):
        try:
            model = tf.keras.models.load_model('lstm_model.h5')
            print("Basic model loaded successfully")
            print(f"Model architecture: {model.summary()}")
            print(f"Input shape: {model.input_shape}")
            return model
        except Exception as e:
            print(f"Error loading basic model: {str(e)}")
            return None
            
    def _log_initialization_status(self):
        """Log the initialization status of various components"""
        print("\n📊 Model Status")
        
        # Check and display model status
        self.status["model_loaded"] = self.model is not None
        print(f"Basic Model: {'Loaded ✅' if self.status['model_loaded'] else 'Not Found ❌'}")
        
        if not self.status["model_loaded"]:
            print("⚠️ Basic model not found. Please train the model first.")

    def calculate_all_indicators(self) -> pd.DataFrame:
        try:
            print("\n🔄 Processing Data for Basic Model")
            
            # Create a copy of the data to avoid modifying the original
            df = self.data.copy()
            
            # Initialize DL_Prediction column with Close prices
            df['DL_Prediction'] = df['Close']
            
            # Calculate technical indicators first with proper error handling
            try:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral RSI
                print("✅ RSI calculated successfully")
                self.status["features_calculated"] = True
            except Exception as e:
                df['RSI'] = 50
                print(f"❌ RSI calculation failed: {str(e)}")
                
            try:
                df['SMA20'] = ta.sma(df['Close'], length=20)
                df['SMA20'] = df['SMA20'].fillna(method='ffill').fillna(df['Close'])
                print("✅ SMA20 calculated successfully")
            except Exception as e:
                df['SMA20'] = df['Close']
                print(f"❌ SMA20 calculation failed: {str(e)}")
                
            try:
                df['SMA50'] = ta.sma(df['Close'], length=50)
                df['SMA50'] = df['SMA50'].fillna(method='ffill').fillna(df['Close'])
                print("✅ SMA50 calculated successfully")
            except Exception as e:
                df['SMA50'] = df['Close']
                print(f"❌ SMA50 calculation failed: {str(e)}")
            
            # Calculate volatility with proper error handling
            try:
                returns = df['Close'].pct_change()
                df['Volatility'] = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
                df['Volatility'] = df['Volatility'].fillna(method='ffill').fillna(0)
                print("✅ Volatility calculated successfully")
            except Exception as e:
                df['Volatility'] = 0
                print(f"❌ Volatility calculation failed: {str(e)}")
            
            # Add predictions if model exists
            if self.model:
                try:
                    print("\n🤖 Generating model predictions...")
                    sequence_length = 60
                    
                    # Prepare data for prediction
                    close_data = df['Close'].values.reshape(-1, 1)
                    close_scaler = MinMaxScaler()
                    scaled_close = close_scaler.fit_transform(close_data)
                    
                    # Create sequences for historical predictions
                    X = []
                    for i in range(sequence_length, len(scaled_close)):
                        X.append(scaled_close[i-sequence_length:i])
                    X = np.array(X)
                    
                    if len(X) > 0:
                        # Generate historical predictions
                        scaled_predictions = self.model.predict(X)
                        predictions = close_scaler.inverse_transform(scaled_predictions)
                        
                        # Add historical predictions to DataFrame
                        df['DL_Prediction'] = np.nan
                        df.iloc[sequence_length:sequence_length + len(predictions), df.columns.get_loc('DL_Prediction')] = predictions.flatten()
                        
                        # Calculate price statistics for constraints
                        mean_price = df['Close'].mean()
                        std_price = df['Close'].std()
                        max_price = df['Close'].max()
                        min_price = df['Close'].min()
                        
                        # Define reasonable bounds (3 standard deviations from mean)
                        upper_bound = max(max_price * 1.1, mean_price + 3 * std_price)
                        lower_bound = min(min_price * 0.9, mean_price - 3 * std_price)
                        
                        # Generate future predictions
                        future_steps = 30
                        last_sequence = scaled_close[-sequence_length:]
                        future_predictions = []
                        current_sequence = last_sequence.copy()
                        last_pred = df['Close'].iloc[-1]
                        
                        # Smoothing factor
                        alpha = 0.3
                        
                        for _ in range(future_steps):
                            # Reshape sequence for prediction (batch_size, sequence_length, features)
                            current_sequence_reshaped = current_sequence.reshape((1, sequence_length, 1))
                            next_pred_scaled = self.model.predict(current_sequence_reshaped)
                            next_pred = close_scaler.inverse_transform(next_pred_scaled.reshape(-1, 1))[0, 0]
                            
                            # Apply constraints
                            max_change = last_pred * 0.1  # Max 10% change per step
                            next_pred = np.clip(next_pred, last_pred - max_change, last_pred + max_change)
                            next_pred = np.clip(next_pred, lower_bound, upper_bound)
                            
                            # Apply exponential smoothing
                            next_pred = alpha * next_pred + (1 - alpha) * last_pred
                            
                            future_predictions.append(next_pred)
                            last_pred = next_pred
                            
                            # Update sequence
                            current_sequence = np.roll(current_sequence, -1, axis=0)
                            current_sequence[-1] = close_scaler.transform([[next_pred]])[0]
                        
                        # Create future dates
                        last_date = df.index[-1]
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
                        
                        # Create future predictions DataFrame
                        future_df = pd.DataFrame(index=future_dates, columns=df.columns)
                        future_df['DL_Prediction'] = future_predictions
                        future_df['Close'] = future_predictions
                        
                        # Copy last known values for technical indicators
                        future_df['RSI'] = df['RSI'].iloc[-1]
                        future_df['SMA20'] = df['SMA20'].iloc[-1]
                        future_df['SMA50'] = df['SMA50'].iloc[-1]
                        future_df['Volatility'] = df['Volatility'].iloc[-1]
                        
                        # Combine historical and future predictions
                        df = pd.concat([df, future_df])
                        print("✅ Predictions generated successfully")
                        self.status["predictions_generated"] = True
                    
                except Exception as e:
                    print(f"❌ Model prediction failed: {str(e)}")
                    df['DL_Prediction'] = df['Close']
            else:
                print("⚠️ No model available, using Close price as prediction")
                df['DL_Prediction'] = df['Close']
            
            # Fill any remaining NaN values with appropriate defaults
            df['RSI'] = df['RSI'].fillna(50)
            df['SMA20'] = df['SMA20'].fillna(df['Close'])
            df['SMA50'] = df['SMA50'].fillna(df['Close'])
            df['Volatility'] = df['Volatility'].fillna(method='ffill').fillna(0)
            df['DL_Prediction'] = df['DL_Prediction'].fillna(method='ffill').fillna(df['Close'])
            
            print("\n📊 Basic Model Analysis Complete")
            return df
            
        except Exception as e:
            print(f"❌ Basic technical analysis failed: {str(e)}")
            # Ensure the returned DataFrame has all required columns
            result = self.data.copy()
            result['DL_Prediction'] = result['Close']
            result['RSI'] = 50
            result['SMA20'] = result['Close']
            result['SMA50'] = result['Close']
            result['Volatility'] = 0
            return result

    def _detect_volatility_clusters(self) -> pd.Series:
        returns = self.data['Close'].pct_change()
        return (returns.rolling(20).std() > returns.std()).astype(int)

# Market Data Integrator
class MarketDataIntegrator:
    """Enhanced market data integration with multiple sentiment sources"""
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Initialize API key from global variable
        self.api_key = ALPHA_VANTAGE_API_KEY

    def get_market_movers(self) -> Dict:
        try:
            print("📊 Fetching market movers data...")
            
            # Use Yahoo Finance Screener API for more reliable data
            gainers = []
            losers = []
            active = []
            
            try:
                # Get gainers from Yahoo Finance Screener
                gainers_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=day_gainers&count=5"
                gainers_response = requests.get(gainers_url, headers=self.headers)
                if gainers_response.status_code == 200:
                    gainers_data = gainers_response.json()
                    gainers = [quote['symbol'] for quote in gainers_data.get('finance', {}).get('result', [{}])[0].get('quotes', [])]
                    print("✅ Gainers data fetched successfully")
                
                # Get losers from Yahoo Finance Screener
                losers_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=day_losers&count=5"
                losers_response = requests.get(losers_url, headers=self.headers)
                if losers_response.status_code == 200:
                    losers_data = losers_response.json()
                    losers = [quote['symbol'] for quote in losers_data.get('finance', {}).get('result', [{}])[0].get('quotes', [])]
                    print("✅ Losers data fetched successfully")
                
                # Get most active from Yahoo Finance Screener
                active_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=most_actives&count=5"
                active_response = requests.get(active_url, headers=self.headers)
                if active_response.status_code == 200:
                    active_data = active_response.json()
                    active = [quote['symbol'] for quote in active_data.get('finance', {}).get('result', [{}])[0].get('quotes', [])]
                    print("✅ Active stocks data fetched successfully")
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Network error while fetching market data: {str(e)}")
            
            # If any list is empty, try alternative Yahoo Finance API endpoint
            if not (gainers and losers and active):
                print("⚠️ Trying alternative Yahoo Finance API endpoint...")
                try:
                    # Get market movers from alternative endpoint
                    alt_url = "https://query2.finance.yahoo.com/v6/finance/quote/watchlist/prebuilt/trending"
                    alt_response = requests.get(alt_url, headers=self.headers)
                    if alt_response.status_code == 200:
                        alt_data = alt_response.json()
                        quotes = alt_data.get('quoteResponse', {}).get('result', [])
                        if quotes:
                            # Sort by volume for active stocks
                            volume_sorted = sorted(quotes, key=lambda x: x.get('regularMarketVolume', 0), reverse=True)
                            # Sort by percent change for gainers/losers
                            change_sorted = sorted(quotes, key=lambda x: x.get('regularMarketChangePercent', 0), reverse=True)
                            
                            if not gainers:
                                gainers = [quote['symbol'] for quote in change_sorted[:5]]
                            if not losers:
                                losers = [quote['symbol'] for quote in change_sorted[-5:]]
                            if not active:
                                active = [quote['symbol'] for quote in volume_sorted[:5]]
                            print("✅ Market data fetched from alternative endpoint")
                except Exception as alt_e:
                    print(f"⚠️ Alternative endpoint failed: {str(alt_e)}")
            
            # If still no data, use default values
            if not (gainers and losers and active):
                print("⚠️ Using default market data:")
                gainers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
                losers = ["META", "NFLX", "TSLA", "AMD", "INTC"]
                active = ["SPY", "QQQ", "IWM", "DIA", "VIX"]
                print("  - Default Gainers:", ", ".join(gainers))
                print("  - Default Losers:", ", ".join(losers))
                print("  - Default Active:", ", ".join(active))
            else:
                print("✅ Live market data fetched successfully")
                print("  - Top Gainers:", ", ".join(gainers))
                print("  - Top Losers:", ", ".join(losers))
                print("  - Most Active:", ", ".join(active))
            
            return {
                "gainers": gainers[:5],  # Ensure we return at most 5 symbols
                "losers": losers[:5],
                "active": active[:5]
            }
            
        except Exception as e:
            print(f"❌ Market data fetch failed: {str(e)}")
            print("⚠️ Using default market data:")
            default_data = {
                "gainers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                "losers": ["META", "NFLX", "TSLA", "AMD", "INTC"],
                "active": ["SPY", "QQQ", "IWM", "DIA", "VIX"]
            }
            print("  - Default Gainers:", ", ".join(default_data["gainers"]))
            print("  - Default Losers:", ", ".join(default_data["losers"]))
            print("  - Default Active:", ", ".join(default_data["active"]))
            return default_data

    def get_market_sentiment(self, symbol: str) -> float:
        try:
            print(f"📊 Fetching multi-source sentiment data for {symbol}...")
            
            # Handle international symbols
            base_symbol = symbol.split('.')[0]  # Remove exchange suffix (e.g., .DE)
            
            # Initialize sentiment scores from different sources
            sentiment_scores = []
            
            # 1. Try Yahoo Finance
            try:
                print("🔍 Checking Yahoo Finance sentiment...")
                yf_ticker = yf.Ticker(symbol)
                recommendations = yf_ticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    latest_rec = recommendations.iloc[-1]
                    # Convert recommendation to sentiment score
                    rec_map = {'Strong Buy': 1.0, 'Buy': 0.75, 'Hold': 0.5, 'Sell': 0.25, 'Strong Sell': 0.0}
                    if latest_rec['To Grade'] in rec_map:
                        sentiment_scores.append(rec_map[latest_rec['To Grade']])
                        print(f"✅ Yahoo Finance sentiment: {rec_map[latest_rec['To Grade']]:.2f}")
            except Exception as e:
                print(f"⚠️ Yahoo Finance sentiment unavailable: {str(e)}")

            # 2. Try Finviz (for US stocks)
            try:
                print(f"🔍 Checking Finviz sentiment for {base_symbol}...")
                url = f"https://finviz.com/quote.ashx?t={base_symbol}"
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    sentiment = self._analyze_sentiment(response.text)
                    sentiment_scores.append(sentiment)
                    print(f"✅ Finviz sentiment: {sentiment:.2f}")
            except Exception as e:
                print(f"⚠️ Finviz sentiment unavailable: {str(e)}")

            # 3. Try Alpha Vantage News Sentiment (if API key available)
            if self.api_key:
                try:
                    print("🔍 Checking Alpha Vantage news sentiment...")
                    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        if "feed" in data:
                            # Calculate average sentiment from news
                            sentiments = [float(article.get('overall_sentiment_score', 0.5)) for article in data['feed']]
                            if sentiments:
                                avg_sentiment = sum(sentiments) / len(sentiments)
                                # Convert from [-1,1] to [0,1] range
                                normalized_sentiment = (avg_sentiment + 1) / 2
                                sentiment_scores.append(normalized_sentiment)
                                print(f"✅ Alpha Vantage sentiment: {normalized_sentiment:.2f}")
                except Exception as e:
                    print(f"⚠️ Alpha Vantage sentiment unavailable: {str(e)}")

            # 4. Technical Sentiment (as fallback)
            try:
                print("🔍 Calculating technical sentiment...")
                tech_sentiment = self._calculate_technical_sentiment(symbol)
                sentiment_scores.append(tech_sentiment)
                print(f"✅ Technical sentiment: {tech_sentiment:.2f}")
            except Exception as e:
                print(f"⚠️ Technical sentiment calculation failed: {str(e)}")

            # Combine all available sentiment scores
            if sentiment_scores:
                final_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                print(f"✅ Combined sentiment from {len(sentiment_scores)} sources: {final_sentiment:.2f}")
                return final_sentiment
            else:
                print("⚠️ No sentiment data available, using neutral value: 0.5")
                return 0.5

        except Exception as e:
            print(f"❌ Sentiment analysis failed: {str(e)}")
            print("⚠️ Using default neutral sentiment value: 0.5")
            return 0.5

    def _calculate_technical_sentiment(self, symbol: str) -> float:
        """Calculate sentiment based on technical indicators"""
        try:
            # Get recent data
            data = yf.download(symbol, period="1mo", progress=False)
            if data is None or data.empty:
                print("⚠️ No data available for technical sentiment calculation")
                return 0.5

            # Calculate basic technical indicators
            close = data['Close']
            if close is None or close.empty:
                print("⚠️ No close price data available")
                return 0.5
                
            sma20 = ta.sma(close, length=20)
            rsi = ta.rsi(close, length=14)
            
            # Get latest values with safety checks
            try:
                latest_close = float(close.iloc[-1])
                latest_sma20 = float(sma20.iloc[-1])
                latest_rsi = float(rsi.iloc[-1])
            except (IndexError, AttributeError, TypeError):
                print("⚠️ Could not access latest indicator values")
                return 0.5
            
            # Calculate sentiment components
            trend_signal = 1 if latest_close > latest_sma20 else 0  # Above SMA20 is bullish
            rsi_signal = (latest_rsi / 100)  # RSI normalized to [0,1]
            
            # Combine signals
            tech_sentiment = (trend_signal + rsi_signal) / 2
            return tech_sentiment

        except Exception as e:
            print(f"Technical sentiment calculation error: {str(e)}")
            return 0.5

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment from text content"""
        try:
            # Extended word lists for better sentiment analysis
            positive_words = [
                'buy', 'strong', 'growth', 'bullish', 'positive', 'upgrade', 'outperform',
                'overweight', 'opportunity', 'upside', 'momentum', 'beat', 'exceed'
            ]
            negative_words = [
                'sell', 'weak', 'decline', 'bearish', 'negative', 'downgrade', 'underperform',
                'underweight', 'risk', 'downside', 'miss', 'below', 'warning'
            ]
            
            # Count word occurrences with context
            text_lower = text.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)
            
            # Add weight to recent sentiment signals
            if 'upgrade' in text_lower or 'downgrade' in text_lower:
                if text_lower.count('upgrade') > text_lower.count('downgrade'):
                    positive_count += 2
                else:
                    negative_count += 2
            
            total = positive_count + negative_count
            if total == 0:
                print("⚠️ No sentiment indicators found, using default neutral value: 0.5")
                return 0.5
                
            sentiment = positive_count / total
            return sentiment
            
        except Exception as e:
            print(f"❌ Sentiment calculation failed: {str(e)}")
            print("⚠️ Using default neutral sentiment value: 0.5")
            return 0.5

# Portfolio Optimization
class PortfolioEngine:
    """Fixed portfolio optimization with robust error handling"""
    def __init__(self):
        self.risk_free_rate = 0.02

    def optimize_allocation(self, returns: pd.DataFrame) -> Dict:
        try:
            # Calculate expected returns (annualized)
            mu = returns.mean() * 252
            
            # Calculate covariance matrix (annualized) with simple shrinkage
            sample_cov = returns.cov() * 252
            shrinkage_factor = 0.1  # Shrinkage intensity
            target = np.diag(np.diag(sample_cov))  # Diagonal matrix of variances
            S = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target
            
            # Ensure matrix is positive definite
            min_eigenval = np.linalg.eigvals(S).min()
            if min_eigenval < 1e-10:
                S += (1e-10 - min_eigenval) * np.eye(S.shape[0])
            
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)  # Long-only constraint
            ef.add_constraint(lambda w: sum(w) == 1)  # Fully invested constraint
            
            try:
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            except Exception as e:
                print(f"Max Sharpe optimization failed: {str(e)}, trying minimum volatility")
                weights = ef.min_volatility()  # Fallback to minimum volatility
                
            cleaned_weights = ef.clean_weights()
            performance = self.calculate_portfolio_metrics(returns, cleaned_weights)
            return {
                "weights": cleaned_weights,
                "metrics": performance
            }
        except Exception as e:
            print(f"Portfolio optimization error: {str(e)}")
            n_assets = len(returns.columns)
            equal_weights = {col: 1.0/n_assets for col in returns.columns}
            return {
                "weights": equal_weights,
                "metrics": {
                    "expected_return": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0
                }
            }

    def calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: Dict) -> Dict:
        try:
            w = np.array([weights[col] for col in returns.columns])
            
            # Calculate annualized return
            portfolio_return = np.sum(returns.mean() * w) * 252
            
            # Calculate annualized volatility with numerical stability check
            cov_matrix = returns.cov() * 252
            portfolio_var = np.dot(w.T, np.dot(cov_matrix, w))
            portfolio_vol = np.sqrt(max(portfolio_var, 1e-10))  # Ensure non-negative
            
            # Calculate Sharpe ratio with safety checks
            excess_return = portfolio_return - self.risk_free_rate
            sharpe = excess_return / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe
            }
        except Exception as e:
            print(f"Metrics calculation error: {str(e)}")
            return {
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0
            }

# Trading Assistant
class TradingAssistant:
    """Main trading analysis engine with improved signal generation"""
    def __init__(self):
        self.asset_options = {
            "Stocks": ["BAYN.DE", "TSLA", "CVAC", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
            "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD"],
            "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
            "ETF": ["SPYI.DE", "IEAG.AS"]
        }

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict:
        try:
            # Handle multi-index DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                # Create a single-level DataFrame with the correct symbol
                df = pd.DataFrame({
                    'Close': data['Close', symbol],
                    'Open': data['Open', symbol],
                    'High': data['High', symbol],
                    'Low': data['Low', symbol],
                    'Volume': data['Volume', symbol]
                }, index=data.index)
            else:
                df = data.copy()

            # Calculate indicators
            analysis = EnhancedTechnicalAnalysis(df)
            results = analysis.calculate_all_indicators()
            
            # Get the last values
            last_close = float(results['Close'].iloc[-1])
            rsi = float(results['RSI'].iloc[-1]) if 'RSI' in results.columns else 50
            sma20 = float(results['SMA20'].iloc[-1]) if 'SMA20' in results.columns else last_close
            sma50 = float(results['SMA50'].iloc[-1]) if 'SMA50' in results.columns else last_close
            volatility = int(results['Volatility'].iloc[-1]) if 'Volatility' in results.columns else 0
            
            # Generate signal
            signal = "HOLD"
            if rsi < 30 and last_close < sma20:
                signal = "BUY"
            elif rsi > 70 and last_close > sma20:
                signal = "SELL"
                
            return {
                "symbol": symbol,
                "signal": signal,
                "details": {
                    "RSI": rsi,
                    "SMA20": sma20,
                    "SMA50": sma50,
                    "Last_Close": last_close,
                    "Volatility": volatility
                }
            }
        except Exception as e:
            st.error(f"Signal generation failed: {str(e)}")
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "details": {
                    "RSI": 50,
                    "SMA20": None,
                    "SMA50": None,
                    "Last_Close": None,
                    "Volatility": 0
                }
            }

def get_stock_data(symbol: str, period: str, max_retries: int = 3) -> pd.DataFrame:
    """Helper function to download stock data with retries"""
    for attempt in range(max_retries):
        try:
            data = yf.download(symbol, period=period, progress=False)
            if not data.empty:
                # Convert multi-index to single-level if needed
                if isinstance(data.columns, pd.MultiIndex):
                    data = pd.DataFrame({
                        'Open': data['Open', symbol],
                        'High': data['High', symbol],
                        'Low': data['Low', symbol],
                        'Close': data['Close', symbol],
                        'Volume': data['Volume', symbol]
                    }, index=data.index)
                return data
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(f"Failed to download data for {symbol} after {max_retries} attempts: {str(e)}")
            continue
    return pd.DataFrame()  # Return empty DataFrame if all attempts fail

def create_lstm_model(sequence_length: int, n_features: int = 1):
    """Create and return an LSTM model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_lstm_model(symbol: str, sequence_length: int = 60) -> None:
    """Train and save the LSTM model for a given symbol"""
    try:
        print(f"Starting model training for {symbol}")
        
        # Get historical data
        data = get_stock_data(symbol, "5y")  # Get 5 years of data for training
        if data.empty:
            raise Exception("No data available for training")
            
        # Prepare data - data is already in single-level format from get_stock_data
        df = data.copy()
        
        # Calculate technical indicators
        df['SMA20'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])
        X, y = np.array(X), np.array(y)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create and train the model
        model = create_lstm_model(sequence_length)
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the model and scaler
        model.save('lstm_model.h5')
        print("Model saved successfully")
        
        # Calculate and print model performance
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {test_loss}")
        
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise e

class EnhancedMultiModalAnalysis:
    """Advanced multi-modal technical analysis with enhanced prediction capabilities"""
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.model = self._load_enhanced_model()
        self.sentiment_model = self._load_sentiment_model()
        self.news_data = None
        self.social_sentiment = None
        self.market_regime = None
        self.api_key = ALPHA_VANTAGE_API_KEY
        
        # Initialize logging status
        self.status = {
            "model_loaded": False,
            "sentiment_model_loaded": False,
            "alpha_vantage_enabled": False,
            "news_data_available": False,
            "features_calculated": False,
            "market_regime_analyzed": False
        }
        
        # Log initialization status
        self._log_initialization_status()
    
    def _load_enhanced_model(self):
        """Load the enhanced LSTM model"""
        try:
            model = tf.keras.models.load_model('enhanced_lstm_model.h5')
            print("Enhanced model loaded successfully")
            print(f"Model architecture: {model.summary()}")
            print(f"Input shape: {model.input_shape}")
            return model
        except Exception as e:
            print(f"Error loading enhanced model: {str(e)}")
            return None
            
    def _load_sentiment_model(self):
        """Load the sentiment analysis model"""
        try:
            # For now, return a simple sentiment analyzer
            # This can be replaced with a more sophisticated model later
            return lambda x: 0.5  # Default neutral sentiment
        except Exception as e:
            print(f"Error loading sentiment model: {str(e)}")
            return None
            
    def _log_initialization_status(self):
        """Log the initialization status of various components"""
        print("\n📊 Model Status")
        
        # Check and display model status
        self.status["model_loaded"] = self.model is not None
        print(f"Enhanced Model: {'Loaded ✅' if self.status['model_loaded'] else 'Not Found ❌'}")
        
        # Check and display sentiment model status
        self.status["sentiment_model_loaded"] = self.sentiment_model is not None
        print(f"Sentiment Model: {'Loaded ✅' if self.status['sentiment_model_loaded'] else 'Not Found ❌'}")
        
        # Check and display Alpha Vantage status
        self.status["alpha_vantage_enabled"] = self.api_key is not None
        print(f"Alpha Vantage API: {'Connected ✅' if self.status['alpha_vantage_enabled'] else 'Not Configured ❌'}")
        
        if not self.status["alpha_vantage_enabled"]:
            print("⚠️ Alpha Vantage API key not configured. News sentiment and additional market data will not be available.")
        
        if not self.status["model_loaded"]:
            print("⚠️ Enhanced model not found. Please train the model first.")

    def calculate_all_indicators(self) -> pd.DataFrame:
        try:
            print("\n🔄 Processing Data for Enhanced Model")
            
            # Create a copy of the data to avoid modifying the original
            df = self.data.copy()
            
            # Initialize DL_Prediction column with Close prices
            df['DL_Prediction'] = df['Close']
            
            # Calculate technical indicators first with proper error handling
            try:
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral RSI
                print("✅ RSI calculated successfully")
                self.status["features_calculated"] = True
            except Exception as e:
                df['RSI'] = 50
                print(f"❌ RSI calculation failed: {str(e)}")
                
            try:
                df['SMA20'] = ta.sma(df['Close'], length=20)
                df['SMA20'] = df['SMA20'].fillna(method='ffill').fillna(df['Close'])
                print("✅ SMA20 calculated successfully")
            except Exception as e:
                df['SMA20'] = df['Close']
                print(f"❌ SMA20 calculation failed: {str(e)}")
                
            try:
                df['SMA50'] = ta.sma(df['Close'], length=50)
                df['SMA50'] = df['SMA50'].fillna(method='ffill').fillna(df['Close'])
                print("✅ SMA50 calculated successfully")
            except Exception as e:
                df['SMA50'] = df['Close']
                print(f"❌ SMA50 calculation failed: {str(e)}")
            
            # Calculate volatility with proper error handling
            try:
                returns = df['Close'].pct_change()
                df['Volatility'] = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
                df['Volatility'] = df['Volatility'].fillna(method='ffill').fillna(0)
                print("✅ Volatility calculated successfully")
            except Exception as e:
                df['Volatility'] = 0
                print(f"❌ Volatility calculation failed: {str(e)}")
            
            # Analyze market regime
            try:
                self.market_regime = self._analyze_market_regime()
                print(f"✅ Market regime analyzed: {self.market_regime}")
            except Exception as e:
                self.market_regime = "Unknown"
                print(f"❌ Market regime analysis failed: {str(e)}")
            
            # Add predictions if model exists
            if self.model:
                try:
                    print("\n🤖 Generating model predictions...")
                    sequence_length = 60
                    
                    # Prepare features for enhanced model
                    feature_columns = ['Close', 'RSI', 'SMA20', 'SMA50', 'Volatility']
                    features = df[feature_columns].values
                    
                    # Scale all features
                    feature_scaler = MinMaxScaler()
                    scaled_features = feature_scaler.fit_transform(features)
                    
                    # Create sequences for historical predictions
                    X = []
                    for i in range(sequence_length, len(scaled_features)):
                        X.append(scaled_features[i-sequence_length:i])
                    X = np.array(X)
                    
                    if len(X) > 0:
                        # Generate historical predictions
                        scaled_predictions = self.model.predict(X)
                        
                        # We only need to inverse transform the Close price predictions
                        close_scaler = MinMaxScaler()
                        close_scaler.fit_transform(df[['Close']])  # Fit to Close prices
                        predictions = close_scaler.inverse_transform(scaled_predictions)
                        
                        # Add historical predictions to DataFrame
                        df['DL_Prediction'] = np.nan
                        df.iloc[sequence_length:sequence_length + len(predictions), df.columns.get_loc('DL_Prediction')] = predictions.flatten()
                        
                        # Calculate price statistics for constraints
                        mean_price = df['Close'].mean()
                        std_price = df['Close'].std()
                        max_price = df['Close'].max()
                        min_price = df['Close'].min()
                        
                        # Define reasonable bounds (3 standard deviations from mean)
                        upper_bound = max(max_price * 1.1, mean_price + 3 * std_price)
                        lower_bound = min(min_price * 0.9, mean_price - 3 * std_price)
                        
                        # Generate future predictions
                        future_steps = 30
                        last_sequence = scaled_features[-sequence_length:]
                        future_predictions = []
                        current_sequence = last_sequence.copy()
                        last_pred = df['Close'].iloc[-1]
                        
                        # Smoothing factor
                        alpha = 0.3
                        
                        for _ in range(future_steps):
                            # Reshape sequence for prediction
                            current_sequence_reshaped = current_sequence.reshape((1, sequence_length, len(feature_columns)))
                            next_pred_scaled = self.model.predict(current_sequence_reshaped)
                            next_pred = close_scaler.inverse_transform(next_pred_scaled.reshape(-1, 1))[0, 0]
                            
                            # Apply constraints
                            max_change = last_pred * 0.1  # Max 10% change per step
                            next_pred = np.clip(next_pred, last_pred - max_change, last_pred + max_change)
                            next_pred = np.clip(next_pred, lower_bound, upper_bound)
                            
                            # Apply exponential smoothing
                            next_pred = alpha * next_pred + (1 - alpha) * last_pred
                            
                            future_predictions.append(next_pred)
                            last_pred = next_pred
                            
                            # Update sequence - maintain all features
                            current_sequence = np.roll(current_sequence, -1, axis=0)
                            # Update all features for next prediction
                            next_features = np.array([
                                next_pred,  # Close
                                df['RSI'].iloc[-1],  # RSI
                                df['SMA20'].iloc[-1],  # SMA20
                                df['SMA50'].iloc[-1],  # SMA50
                                df['Volatility'].iloc[-1]  # Volatility
                            ])
                            current_sequence[-1] = feature_scaler.transform(next_features.reshape(1, -1))
                        
                        # Create future dates
                        last_date = df.index[-1]
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
                        
                        # Create future predictions DataFrame
                        future_df = pd.DataFrame(index=future_dates, columns=df.columns)
                        future_df['DL_Prediction'] = future_predictions
                        future_df['Close'] = future_predictions
                        
                        # Copy last known values for technical indicators
                        future_df['RSI'] = df['RSI'].iloc[-1]
                        future_df['SMA20'] = df['SMA20'].iloc[-1]
                        future_df['SMA50'] = df['SMA50'].iloc[-1]
                        future_df['Volatility'] = df['Volatility'].iloc[-1]
                        
                        # Combine historical and future predictions
                        df = pd.concat([df, future_df])
                        print("✅ Predictions generated successfully")
                        self.status["predictions_generated"] = True
                    
                except Exception as e:
                    print(f"❌ Model prediction failed: {str(e)}")
                    df['DL_Prediction'] = df['Close']
            else:
                print("⚠️ No model available, using Close price as prediction")
                df['DL_Prediction'] = df['Close']
            
            # Fill any remaining NaN values with appropriate defaults
            df['RSI'] = df['RSI'].fillna(50)
            df['SMA20'] = df['SMA20'].fillna(df['Close'])
            df['SMA50'] = df['SMA50'].fillna(df['Close'])
            df['Volatility'] = df['Volatility'].fillna(method='ffill').fillna(0)
            df['DL_Prediction'] = df['DL_Prediction'].fillna(method='ffill').fillna(df['Close'])
            
            print("\n📊 Enhanced Model Analysis Complete")
            return df
            
        except Exception as e:
            print(f"❌ Enhanced technical analysis failed: {str(e)}")
            # Ensure the returned DataFrame has all required columns
            result = self.data.copy()
            result['DL_Prediction'] = result['Close']
            result['RSI'] = 50
            result['SMA20'] = result['Close']
            result['SMA50'] = result['Close']
            result['Volatility'] = 0
            return result

    def _fetch_news_data(self, symbol: str) -> pd.DataFrame:
        """Fetch and process news data for the given symbol"""
        if not self.api_key:
            return pd.DataFrame()
            
        try:
            st.info("Connecting to Alpha Vantage API...")
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                st.error(f"API request failed with status code: {response.status_code}")
                return pd.DataFrame()
            
            news_data = response.json()
            
            if "Note" in news_data:
                st.warning(f"Alpha Vantage API limit reached: {news_data['Note']}")
                return pd.DataFrame()
            
            # Process news data
            processed_news = pd.DataFrame(news_data.get('feed', []))
            if not processed_news.empty:
                processed_news['time_published'] = pd.to_datetime(processed_news['time_published'])
                processed_news['sentiment_score'] = processed_news['overall_sentiment_score'].astype(float)
                st.success(f"Successfully processed {len(processed_news)} news articles")
                return processed_news
                
            st.warning("No news data found in API response")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching news data: {str(e)}")
            return pd.DataFrame()

    def _analyze_market_regime(self) -> str:
        """Analyze current market regime using volatility and trend indicators"""
        try:
            # Calculate volatility
            returns = self.data['Close'].pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
            current_volatility = volatility.iloc[-1]
            avg_volatility = volatility.mean()
            
            # Calculate trend
            sma20 = ta.sma(self.data['Close'], length=20).fillna(method='ffill')
            sma50 = ta.sma(self.data['Close'], length=50).fillna(method='ffill')
            
            # Get latest values
            latest_sma20 = sma20.iloc[-1]
            latest_sma50 = sma50.iloc[-1]
            
            # Determine regime
            if current_volatility > avg_volatility:
                if latest_sma20 > latest_sma50:
                    return "High Volatility Bullish"
                else:
                    return "High Volatility Bearish"
            else:
                if latest_sma20 > latest_sma50:
                    return "Low Volatility Bullish"
                else:
                    return "Low Volatility Bearish"
        except Exception as e:
            print(f"Error analyzing market regime: {str(e)}")
            return "Unknown"

    def _calculate_technical_features(self) -> pd.DataFrame:
        """Calculate advanced technical indicators"""
        try:
            features = pd.DataFrame(index=self.data.index)
            
            # Price-based indicators
            features['returns'] = self.data['Close'].pct_change()
            features['log_returns'] = np.log1p(features['returns'])
            features['volatility'] = features['returns'].rolling(window=20).std()
            
            # Trend indicators
            features['sma20'] = ta.sma(self.data['Close'], length=20)
            features['sma50'] = ta.sma(self.data['Close'], length=50)
            features['ema20'] = ta.ema(self.data['Close'], length=20)
            features['ema50'] = ta.ema(self.data['Close'], length=50)
            
            # Momentum indicators
            features['rsi'] = ta.rsi(self.data['Close'], length=14)
            macd = ta.macd(self.data['Close'])
            features['macd'] = macd['MACD_12_26_9']
            features['macd_signal'] = macd['MACDs_12_26_9']
            
            # Volume indicators
            features['obv'] = ta.obv(self.data['Close'], self.data['Volume'])
            features['volume_sma'] = ta.sma(self.data['Volume'], length=20)
            
            # Volatility indicators
            bb = ta.bbands(self.data['Close'], length=20, std=2)
            features['bb_upper'] = bb['BBU_20_2.0']
            features['bb_lower'] = bb['BBL_20_2.0']
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_lower']
            
            # Fill NaN values with appropriate defaults
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            return features
        except Exception as e:
            print(f"Error calculating technical features: {str(e)}")
            return pd.DataFrame()

def train_enhanced_model(symbol: str, sequence_length: int = 60) -> None:
    """Train and save the enhanced multi-modal model"""
    try:
        print(f"Starting enhanced model training for {symbol}")
        
        # Get historical data
        data = get_stock_data(symbol, "5y")
        if data.empty:
            raise Exception("No data available for training")
        
        # Create enhanced analysis instance
        analysis = EnhancedMultiModalAnalysis(data)
        
        # Calculate features and add Close price
        features = analysis._calculate_technical_features()
        features['Close'] = data['Close']  # Add Close price to features
        
        # Ensure all required columns exist
        required_columns = ['Close', 'volatility', 'rsi', 'macd', 'bb_width']
        missing_columns = [col for col in required_columns if col not in features.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {missing_columns}")
        
        # Drop any rows with NaN values
        features = features.dropna(subset=required_columns)
        
        if len(features) <= sequence_length:
            raise Exception(f"Insufficient data after preprocessing: {len(features)} rows")
        
        # Prepare data for training
        feature_data = features[required_columns].values
        target_data = features['Close'].values.reshape(-1, 1)
        
        # Normalize data
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        scaled_features = feature_scaler.fit_transform(feature_data)
        scaled_target = target_scaler.fit_transform(target_data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(scaled_target[i])
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create enhanced model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(sequence_length, len(required_columns))),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(25),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model.save('enhanced_lstm_model.h5')
        print("Enhanced model saved successfully")
        
        # Evaluate model
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {test_loss}, Test MAE: {test_mae}")
        
    except Exception as e:
        print(f"Error training enhanced model: {str(e)}")
        raise e

# Main Application
def main():
    st.set_page_config(page_title="AI Trading Pro 2", layout="wide", page_icon="📈")
    st.title("🚀 AI Trading Pro 2 - Enhanced Viewing & Prediction")
    
    market_data = MarketDataIntegrator()
    trading_engine = TradingAssistant()
    portfolio_optimizer = PortfolioEngine()

    # Initialize session state for model selection if it doesn't exist
    if 'model_type' not in st.session_state:
        st.session_state.model_type = "Basic LSTM"

    with st.sidebar:
        st.header("⚙️ Configuration")
        # Move model selection to sidebar for global access
        st.session_state.model_type = st.radio(
            "Select Model",
            ["Basic LSTM", "Enhanced Multi-Modal"],
            horizontal=True,
            key="model_selection"
        )
        asset_type = st.selectbox("Asset Class", list(trading_engine.asset_options.keys()))
        symbol = st.selectbox("Symbol", trading_engine.asset_options[asset_type])
        timeframe = st.select_slider("Analysis Period", options=["1D", "5D", "1MO", "3MO", "1Y", "YTD"], value="1MO")
        with st.expander("🔧 Advanced Tools"):
            enable_portfolio = st.checkbox("Portfolio Optimization")
            show_education = st.checkbox("Tutorial Mode")
            risk_level = st.select_slider("Risk Profile", ["Low", "Medium", "High"])

    tab_analysis, tab_predict, tab_portfolio, tab_learn = st.tabs(["Analysis", "Predictions", "Portfolio", "Academy"])

    with tab_analysis:
        if st.button("Run Analysis"):
            with st.spinner("Processing market data..."):
                try:
                    print(f"Starting analysis for symbol: {symbol}")
                    data = get_stock_data(symbol, timeframe.lower())
                    print(f"Data shape: {data.shape if not data.empty else 'Empty'}")
                    
                    if data.empty or len(data.index) == 0:
                        print("Error: Empty data received")
                        st.error(f"No data available for {symbol}. Please try another symbol or try again later.")
                        return
                    
                    # Ensure we have valid data before proceeding
                    if not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
                        print(f"Error: Invalid data format. Type: {type(data)}, Columns: {data.columns if isinstance(data, pd.DataFrame) else 'Not DataFrame'}")
                        st.error("Invalid data format received. Please try again.")
                        return
                    
                    print(f"Using model type: {st.session_state.model_type}")
                    # Use the selected model for analysis
                    if st.session_state.model_type == "Enhanced Multi-Modal":
                        analysis_model = EnhancedMultiModalAnalysis(data)
                    else:
                        analysis_model = EnhancedTechnicalAnalysis(data)
                    
                    results = analysis_model.calculate_all_indicators()
                    
                    # Generate trading signal using the results
                    signal_data = {
                        "symbol": symbol,
                        "signal": "HOLD",  # Default signal
                        "details": {
                            "RSI": float(results['RSI'].iloc[-1]) if 'RSI' in results.columns else 50,
                            "SMA20": float(results['SMA20'].iloc[-1]) if 'SMA20' in results.columns else None,
                            "SMA50": float(results['SMA50'].iloc[-1]) if 'SMA50' in results.columns else None,
                            "Last_Close": float(results['Close'].iloc[-1]) if 'Close' in results.columns else None,
                            "Volatility": float(results['Volatility'].iloc[-1]) if 'Volatility' in results.columns else 0
                        }
                    }
                    
                    # Update signal based on indicators
                    rsi = signal_data['details']['RSI']
                    last_close = signal_data['details']['Last_Close']
                    sma20 = signal_data['details']['SMA20']
                    
                    if rsi < 30 and last_close < sma20:
                        signal_data['signal'] = "BUY"
                    elif rsi > 70 and last_close > sma20:
                        signal_data['signal'] = "SELL"
                    
                    print(f"Analysis result: {signal_data}")
                    
                    print("Getting market movers...")
                    movers = market_data.get_market_movers()
                    print(f"Market movers: {movers}")
                    
                    print("Getting market sentiment...")
                    sentiment = market_data.get_market_sentiment(symbol)
                    print(f"Sentiment score: {sentiment}")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${signal_data['details']['Last_Close']:.2f}" if signal_data['details']['Last_Close'] is not None else "N/A")
                    col2.metric("Signal", signal_data['signal'])
                    col3.metric("RSI", f"{signal_data['details']['RSI']:.1f}" if signal_data['details']['RSI'] is not None else "N/A")
                    col4.metric("Sentiment", f"{sentiment:.2f}" if sentiment is not None else "N/A")
                    
                    # Add model type indicator
                    st.info(f"Analysis performed using {st.session_state.model_type}")
                    
                    # Create candlestick chart
                    print("Creating candlestick chart...")
                    try:
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name="Price"
                        ))
                        
                        # Add historical predictions if available (only up to current date)
                        if 'DL_Prediction' in results.columns:
                            current_date = datetime.now()
                            historical_mask = results.index <= current_date
                            fig.add_trace(go.Scatter(
                                x=results.index[historical_mask],
                                y=results['DL_Prediction'][historical_mask],
                                name="Model Prediction",
                                line=dict(color='lightgrey', dash='dash')
                            ))
                        
                        fig.update_layout(
                            title=f"{symbol} Price Chart ({st.session_state.model_type})",
                            yaxis_title="Price",
                            xaxis_title="Date",
                            height=600,
                            xaxis_rangeslider_visible=False,
                            template="plotly_dark"
                        )
                        
                        print("Chart created successfully")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        print(f"Error creating candlestick chart: {str(e)}")
                        st.error("Failed to create price chart")
                    
                    with st.expander("📈 Market Movers"):
                        cols = st.columns(3)
                        cols[0].subheader("Top Gainers")
                        cols[1].subheader("Top Losers")
                        cols[2].subheader("Most Active")
                        for i in range(5):
                            if i < len(movers['gainers']):
                                cols[0].write(f"{i+1}. {movers['gainers'][i]}")
                            if i < len(movers['losers']):
                                cols[1].write(f"{i+1}. {movers['losers'][i]}")
                            if i < len(movers['active']):
                                cols[2].write(f"{i+1}. {movers['active'][i]}")
                                
                    # Display additional information for Enhanced Multi-Modal
                    if st.session_state.model_type == "Enhanced Multi-Modal" and isinstance(analysis_model, EnhancedMultiModalAnalysis):
                        st.subheader("Additional Analysis")
                        st.write(f"Market Regime: {analysis_model.market_regime}")
                        
                        if analysis_model.news_data is not None and not analysis_model.news_data.empty:
                            st.subheader("Latest News Sentiment")
                            for _, news in analysis_model.news_data.head(3).iterrows():
                                st.write(f"- {news['title']} (Sentiment: {news['sentiment_score']:.2f})")
                    
                except Exception as e:
                    print(f"Analysis failed with error: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    st.error(f"Analysis failed: {str(e)}")

    with tab_predict:
        st.subheader("🔮 Price Predictions")
        
        # Add model training section
        with st.expander("🔄 Train Models"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Train Basic LSTM Model")
                if st.button("Train Basic Model"):
                    with st.spinner("Training basic LSTM model..."):
                        try:
                            train_lstm_model(symbol)
                            st.success("Basic model trained successfully!")
                        except Exception as e:
                            st.error(f"Basic model training failed: {str(e)}")
            
            with col2:
                st.write("Train Enhanced Multi-Modal Model")
                if st.button("Train Enhanced Model"):
                    with st.spinner("Training enhanced model..."):
                        try:
                            train_enhanced_model(symbol)
                            st.success("Enhanced model trained successfully!")
                        except Exception as e:
                            st.error(f"Enhanced model training failed: {str(e)}")
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating price predictions..."):
                try:
                    # Get historical data
                    historical_data = get_stock_data(symbol, "1y")
                    if historical_data.empty:
                        st.error("No historical data available for predictions.")
                        return

                    if st.session_state.model_type == "Basic LSTM":
                        # Use basic LSTM model
                        analysis = EnhancedTechnicalAnalysis(historical_data)
                        results = analysis.calculate_all_indicators()
                        
                        # Split data into historical and future predictions
                        current_date = datetime.now()
                        historical_mask = results.index <= current_date
                        future_mask = results.index > current_date
                        
                        # Plot predictions
                        fig = go.Figure()
                        
                        # Add historical prices
                        fig.add_trace(go.Scatter(
                            x=results.index[historical_mask],
                            y=results['Close'][historical_mask],
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add historical predictions (dashed)
                        fig.add_trace(go.Scatter(
                            x=results.index[historical_mask],
                            y=results['DL_Prediction'][historical_mask],
                            name='Historical Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Add future predictions (solid)
                        fig.add_trace(go.Scatter(
                            x=results.index[future_mask],
                            y=results['DL_Prediction'][future_mask],
                            name='Future Prediction',
                            line=dict(color='red')
                        ))
                        
                        fig.update_layout(
                            title=f"{symbol} Price Prediction ({st.session_state.model_type})",
                            yaxis_title="Price",
                            xaxis_title="Date",
                            height=600,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display prediction metrics
                        last_price = float(results['Close'][historical_mask].iloc[-1])
                        final_pred_price = float(results['DL_Prediction'][future_mask].iloc[-1])
                        change = ((final_pred_price - last_price) / last_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"${last_price:.2f}")
                        col2.metric("Predicted Price", f"${final_pred_price:.2f}")
                        col3.metric("Predicted Change", f"{change:.2f}%")
                        
                        # Calculate and display confidence metrics
                        st.subheader("Prediction Confidence Metrics")
                        
                        # Calculate prediction accuracy based on historical predictions
                        historical_accuracy = 1 - np.mean(np.abs(
                            results['Close'][historical_mask] - results['DL_Prediction'][historical_mask]
                        ) / results['Close'][historical_mask])
                        
                        # Calculate trend strength
                        trend_strength = abs(change) / 10  # Normalize the change percentage
                        
                        # Combined confidence score
                        confidence = min(100, max(0, (historical_accuracy * 70 + trend_strength * 30)))
                        st.progress(confidence/100)
                        st.write(f"Overall Confidence: {confidence:.1f}%")
                        
                        # Display technical indicators
                        st.subheader("Technical Indicators")
                        latest_indicators = results[historical_mask].iloc[-1]
                        indicator_cols = st.columns(4)
                        
                        # Calculate volatility correctly
                        returns = results['Close'].pct_change()
                        volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility
                        
                        try:
                            with indicator_cols[0]:
                                rsi_value = float(latest_indicators.get('RSI', np.nan))
                                st.metric("RSI", f"{rsi_value:.1f}" if not np.isnan(rsi_value) else "N/A")
                        except Exception as e:
                            with indicator_cols[0]:
                                st.metric("RSI", "N/A")
                                
                        try:
                            with indicator_cols[1]:
                                sma20_value = float(latest_indicators.get('SMA20', np.nan))
                                st.metric("SMA20", f"{sma20_value:.2f}" if not np.isnan(sma20_value) else "N/A")
                        except Exception as e:
                            with indicator_cols[1]:
                                st.metric("SMA20", "N/A")
                                
                        try:
                            with indicator_cols[2]:
                                sma50_value = float(latest_indicators.get('SMA50', np.nan))
                                st.metric("SMA50", f"{sma50_value:.2f}" if not np.isnan(sma50_value) else "N/A")
                        except Exception as e:
                            with indicator_cols[2]:
                                st.metric("SMA50", "N/A")
                                
                        try:
                            with indicator_cols[3]:
                                st.metric("Volatility", f"{volatility:.2f}%" if not np.isnan(volatility) else "N/A")
                        except Exception as e:
                            with indicator_cols[3]:
                                st.metric("Volatility", "N/A")
                    else:
                        # Use enhanced model
                        analysis = EnhancedMultiModalAnalysis(historical_data)
                        results = analysis.calculate_all_indicators()
                        
                        # Split data into historical and future predictions
                        current_date = datetime.now()
                        historical_mask = results.index <= current_date
                        future_mask = results.index > current_date
                        
                        # Display market regime
                        st.subheader("Market Regime Analysis")
                        st.write(f"Current Market Regime: {analysis.market_regime}")
                        
                        # Display news sentiment if available
                        if analysis.news_data is not None and not analysis.news_data.empty:
                            st.subheader("News Sentiment Analysis")
                            latest_news = analysis.news_data.head(5)
                            for _, news in latest_news.iterrows():
                                st.write(f"- {news['title']} (Sentiment: {news['sentiment_score']:.2f})")
                        
                        # Plot predictions
                        fig = go.Figure()
                        
                        # Add historical prices
                        fig.add_trace(go.Scatter(
                            x=results.index[historical_mask],
                            y=results['Close'][historical_mask],
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add historical predictions (dashed)
                        fig.add_trace(go.Scatter(
                            x=results.index[historical_mask],
                            y=results['DL_Prediction'][historical_mask],
                            name='Historical Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Add future predictions (solid)
                        fig.add_trace(go.Scatter(
                            x=results.index[future_mask],
                            y=results['DL_Prediction'][future_mask],
                            name='Future Prediction',
                            line=dict(color='red')
                        ))
                        
                        fig.update_layout(
                            title=f"{symbol} Price Prediction ({st.session_state.model_type})",
                            yaxis_title="Price",
                            xaxis_title="Date",
                            height=600,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display prediction metrics
                        last_price = float(results['Close'][historical_mask].iloc[-1])
                        final_pred_price = float(results['DL_Prediction'][future_mask].iloc[-1])
                        change = ((final_pred_price - last_price) / last_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"${last_price:.2f}")
                        col2.metric("Predicted Price", f"${final_pred_price:.2f}")
                        col3.metric("Predicted Change", f"{change:.2f}%")
                        
                        # Calculate and display confidence metrics
                        st.subheader("Prediction Confidence Metrics")
                        
                        # Calculate prediction accuracy based on historical predictions
                        historical_accuracy = 1 - np.mean(np.abs(
                            results['Close'][historical_mask] - results['DL_Prediction'][historical_mask]
                        ) / results['Close'][historical_mask])
                        
                        # Calculate trend strength
                        trend_strength = abs(change) / 10  # Normalize the change percentage
                        
                        # Combined confidence score
                        confidence = min(100, max(0, (historical_accuracy * 70 + trend_strength * 30)))
                        st.progress(confidence/100)
                        st.write(f"Overall Confidence: {confidence:.1f}%")
                        
                        # Display technical indicators
                        st.subheader("Technical Indicators")
                        latest_indicators = results[historical_mask].iloc[-1]
                        indicator_cols = st.columns(4)
                        
                        # Calculate volatility correctly
                        returns = results['Close'].pct_change()
                        volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility
                        
                        try:
                            with indicator_cols[0]:
                                rsi_value = float(latest_indicators.get('RSI', np.nan))
                                st.metric("RSI", f"{rsi_value:.1f}" if not np.isnan(rsi_value) else "N/A")
                        except Exception as e:
                            with indicator_cols[0]:
                                st.metric("RSI", "N/A")
                                
                        try:
                            with indicator_cols[1]:
                                sma20_value = float(latest_indicators.get('SMA20', np.nan))
                                st.metric("SMA20", f"{sma20_value:.2f}" if not np.isnan(sma20_value) else "N/A")
                        except Exception as e:
                            with indicator_cols[1]:
                                st.metric("SMA20", "N/A")
                                
                        try:
                            with indicator_cols[2]:
                                sma50_value = float(latest_indicators.get('SMA50', np.nan))
                                st.metric("SMA50", f"{sma50_value:.2f}" if not np.isnan(sma50_value) else "N/A")
                        except Exception as e:
                            with indicator_cols[2]:
                                st.metric("SMA50", "N/A")
                                
                        try:
                            with indicator_cols[3]:
                                st.metric("Volatility", f"{volatility:.2f}%" if not np.isnan(volatility) else "N/A")
                        except Exception as e:
                            with indicator_cols[3]:
                                st.metric("Volatility", "N/A")
                        
                except Exception as e:
                    print(f"Prediction failed with error: {str(e)}")
                    st.error(f"Prediction failed: {str(e)}")

    with tab_portfolio:
        if enable_portfolio:
            st.subheader("⚖️ Portfolio Optimization")
            selected_assets = st.multiselect(
                "Select Assets",
                trading_engine.asset_options[asset_type],
                default=trading_engine.asset_options[asset_type][:3]
            )
            if st.button("Optimize Portfolio"):
                try:
                    # Download data and handle multi-index DataFrame
                    data = yf.download(selected_assets, period="1y")
                    
                    # Print debug information
                    print("Downloaded data shape:", data.shape)
                    print("Data columns:", data.columns.tolist())
                    
                    # Create returns DataFrame with single-level columns and handle extreme values
                    returns = pd.DataFrame()
                    for asset in selected_assets:
                        if ('Close', asset) in data.columns:
                            # Calculate returns and handle extreme values
                            price_series = data['Close'][asset].ffill()
                            asset_returns = price_series.pct_change()
                            
                            # Remove first row (NaN from pct_change)
                            asset_returns = asset_returns.iloc[1:]
                            
                            # Winsorize returns at 5th and 95th percentiles
                            lower = np.percentile(asset_returns, 5)
                            upper = np.percentile(asset_returns, 95)
                            asset_returns = np.clip(asset_returns, lower, upper)
                            
                            returns[asset] = asset_returns
                    
                    # Drop any remaining NaN values
                    returns = returns.dropna()
                    
                    # Convert very small numbers to 0 to avoid numerical issues
                    returns[abs(returns) < 1e-10] = 0
                    
                    print("Returns data shape after cleaning:", returns.shape)
                    print("Returns statistics:")
                    print(returns.describe())
                    print("Any NaN values remaining:", returns.isna().any().any())
                    print("Max return value:", returns.max().max())
                    print("Min return value:", returns.min().min())
                    
                    if returns.empty or len(returns.columns) < 2:
                        st.error("Insufficient data available for selected assets. Please select at least 2 assets with valid data.")
                        return
                        
                    if len(returns) < 30:  # Ensure we have enough data points
                        st.error("Insufficient historical data for optimization. Please try different assets or a longer time period.")
                        return
                    
                    # Check for extreme values in covariance
                    cov_matrix = returns.cov()
                    if np.any(np.abs(cov_matrix) > 1):
                        print("Warning: Large values in covariance matrix, scaling returns")
                        returns = returns / returns.std()
                    
                    # Calculate optimal portfolio
                    result = portfolio_optimizer.optimize_allocation(returns)
                    
                    # Display results in a more organized way
                    st.write("### 📊 Portfolio Optimization Results")
                    
                    # Create three columns for better organization
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("#### Asset Allocation")
                        for asset, weight in result['weights'].items():
                            st.metric(asset, f"{weight:.1%}")
                    
                    with col2:
                        st.write("#### Risk Metrics")
                        metrics = result['metrics']
                        st.metric("Expected Return", f"{metrics['expected_return']:.1%}")
                        st.metric("Portfolio Volatility", f"{metrics['volatility']:.1%}")
                    
                    with col3:
                        st.write("#### Performance")
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                        st.metric("Risk-Adjusted Return", f"{metrics['expected_return']/metrics['volatility']:.2f}")
                    
                    # Add a pie chart for visual representation
                    fig = go.Figure(data=[go.Pie(
                        labels=list(result['weights'].keys()),
                        values=list(result['weights'].values()),
                        hole=.3
                    )])
                    fig.update_layout(
                        title="Portfolio Allocation",
                        height=400,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
                    print(f"Detailed error: {str(e)}")
                    if 'returns' in locals():
                        print("\nReturns data sample:")
                        print(returns.head())
                        print("\nReturns statistics:")
                        print(returns.describe())
                        if 'cov_matrix' in locals():
                            print("\nCovariance matrix:")
                            print(cov_matrix)

    with tab_learn:
        st.subheader("📚 Trading Education")
        st.markdown("""
        ### How to Use This App
        1. **Select Asset**: Choose the asset class and specific symbol you want to analyze.
        2. **Run Analysis**: Click "Run Analysis" to see historical data and technical indicators.
        3. **Generate Predictions**: Use the "Prediction" tab to generate future price predictions.
        4. **Optimize Portfolio**: Select multiple assets and optimize your portfolio allocation.
        5. **Simulate Trades**: Use the simulation feature to test your strategies without risking real money.
        """)

if __name__ == "__main__":
    main()