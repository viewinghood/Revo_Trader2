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

    def _load_deep_learning_model(self):
        try:
            # Use tf.compat.v1 for backwards compatibility
            tf.compat.v1.reset_default_graph()
            return tf.keras.models.load_model('lstm_model.h5')
        except:
            return None

    def calculate_all_indicators(self) -> pd.DataFrame:
        try:
            bb = ta.bbands(self.data['Close'], length=20, std=2)
            if bb is not None:
                self.data = pd.concat([self.data, bb], axis=1)
            if len(self.data.index) >= 14:
                self.data['RSI'] = ta.rsi(self.data['Close'], length=14)
            else:
                self.data['RSI'] = np.nan
            if len(self.data.index) >= 20:
                self.data['SMA20'] = ta.sma(self.data['Close'], length=20)
            if len(self.data.index) >= 50:
                self.data['SMA50'] = ta.sma(self.data['Close'], length=50)
            if self.model:
                preds = self.model.predict(self.data[['Close']].values)
                self.data['DL_Prediction'] = preds.flatten()
            self.data['Volatility_Cluster'] = self._detect_volatility_clusters()
            return self.data.dropna(axis=1, how='all')
        except Exception as e:
            st.error(f"Technical analysis failed: {str(e)}")
            return self.data

    def _detect_volatility_clusters(self) -> pd.Series:
        returns = self.data['Close'].pct_change()
        return (returns.rolling(20).std() > returns.std()).astype(int)

# Market Data Integrator
class MarketDataIntegrator:
    """Enhanced market data integration with fixed market movers"""
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_market_movers(self) -> Dict:
        try:
            # Using Yahoo Finance API for more reliable data
            gainers = yf.Ticker("^GSPC").info.get('most_actively_traded', [])[:5]
            losers = yf.Ticker("^GSPC").info.get('most_actively_traded', [])[-5:]
            active = yf.Ticker("^GSPC").info.get('most_actively_traded', [])[:5]
            if not gainers:  # Fallback data if API fails
                gainers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
                losers = ["META", "NFLX", "TSLA", "AMD", "INTC"]
                active = ["SPY", "QQQ", "IWM", "DIA", "VIX"]
            return {
                "gainers": gainers,
                "losers": losers,
                "active": active
            }
        except Exception as e:
            st.warning(f"Using backup market data due to: {str(e)}")
            return {
                "gainers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                "losers": ["META", "NFLX", "TSLA", "AMD", "INTC"],
                "active": ["SPY", "QQQ", "IWM", "DIA", "VIX"]
            }

    def get_market_sentiment(self, symbol: str) -> float:
        try:
            url = f"https://finviz.com/quote.ashx?t={symbol}"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._analyze_sentiment(soup.get_text())
        except:
            return 0.5

    def _analyze_sentiment(self, text: str) -> float:
        positive_words = ['buy', 'strong', 'growth', 'bullish', 'positive']
        negative_words = ['sell', 'weak', 'decline', 'bearish', 'negative']
        positive_count = sum(text.lower().count(word) for word in positive_words)
        negative_count = sum(text.lower().count(word) for word in negative_words)
        total = positive_count + negative_count
        return 0.5 if total == 0 else positive_count / total

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
            analysis = EnhancedTechnicalAnalysis(data).calculate_all_indicators()
            last_close = float(analysis['Close'].iloc[-1])
            rsi = float(analysis.get('RSI', pd.Series([50])).iloc[-1])
            sma20 = float(analysis.get('SMA20', pd.Series([last_close])).iloc[-1])
            sma50 = float(analysis.get('SMA50', pd.Series([last_close])).iloc[-1])
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
                    "Volatility_Cluster": int(analysis.get('Volatility_Cluster', pd.Series([0])).iloc[-1])
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
                    "Volatility_Cluster": 0
                }
            }

def get_stock_data(symbol: str, period: str, max_retries: int = 3) -> pd.DataFrame:
    """Helper function to download stock data with retries"""
    for attempt in range(max_retries):
        try:
            data = yf.download(symbol, period=period, progress=False)
            if not data.empty:
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
            
        # Prepare data
        df = pd.DataFrame({
            'Close': data['Close', symbol]
        }, index=data.index)
        
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

# Main Application
def main():
    st.set_page_config(page_title="AI Trading Pro 2", layout="wide", page_icon="📈")
    st.title("🚀 AI Trading Pro 2 - Enhanced Viewing & Prediction")
    
    market_data = MarketDataIntegrator()
    trading_engine = TradingAssistant()
    portfolio_optimizer = PortfolioEngine()

    with st.sidebar:
        st.header("⚙️ Configuration")
        asset_type = st.selectbox("Asset Class", list(trading_engine.asset_options.keys()))
        symbol = st.selectbox("Symbol", trading_engine.asset_options[asset_type])
        timeframe = st.select_slider("Analysis Period", options=["1D", "5D", "1MO", "3MO", "1Y", "YTD"], value="1MO")
        with st.expander("🔧 Advanced Tools"):
            enable_portfolio = st.checkbox("Portfolio Optimization")
            show_education = st.checkbox("Tutorial Mode")
            risk_level = st.select_slider("Risk Profile", ["Low", "Medium", "High"])

    tab_analysis, tab_predict, tab_portfolio, tab_learn  = st.tabs(["Analysis", "Predictions", "Portfolio", "Academy"])

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
                        
                    print("Generating trading signal...")
                    analysis = trading_engine.generate_signal(symbol, data)
                    print(f"Analysis result: {analysis}")
                    
                    print("Getting market movers...")
                    movers = market_data.get_market_movers()
                    print(f"Market movers: {movers}")
                    
                    print("Getting market sentiment...")
                    sentiment = market_data.get_market_sentiment(symbol)
                    print(f"Sentiment score: {sentiment}")
                    
                    # Get the last close price from the analysis dictionary
                    try:
                        last_close = analysis['details']['Last_Close']
                        print(f"Last close price from analysis: {last_close}")
                    except Exception as e:
                        print(f"Error getting last close price from analysis: {str(e)}")
                        last_close = None
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${last_close:.2f}" if last_close is not None else "N/A")
                    col2.metric("Signal", analysis['signal'])
                    col3.metric("RSI", f"{analysis['details']['RSI']:.1f}" if analysis['details']['RSI'] is not None else "N/A")
                    col4.metric("Sentiment", f"{sentiment:.2f}" if sentiment is not None else "N/A")
                    
                    print("Creating candlestick chart...")
                    try:
                        print(f"Data columns available: {data.columns.tolist()}")
                        print(f"Data shape for chart: {data.shape}")
                        print(f"First few rows of data:\n{data.head()}")
                        
                        # Create a new DataFrame with single-level columns
                        chart_data = pd.DataFrame({
                            'Open': data['Open', symbol],
                            'High': data['High', symbol],
                            'Low': data['Low', symbol],
                            'Close': data['Close', symbol]
                        }, index=data.index)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=chart_data.index,
                            open=chart_data['Open'],
                            high=chart_data['High'],
                            low=chart_data['Low'],
                            close=chart_data['Close'],
                            name="Price"
                        ))
                        
                        # Add layout settings
                        fig.update_layout(
                            title=f"{symbol} Price Chart",
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
                except Exception as e:
                    print(f"Analysis failed with error: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    st.error(f"Analysis failed: {str(e)}")

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

    with tab_predict:
        st.subheader("🔮 Price Predictions")
        
        # Add model training section
        with st.expander("🔄 Train New Model"):
            st.write("Train a new LSTM model for predictions")
            if st.button("Train Model"):
                with st.spinner("Training LSTM model..."):
                    try:
                        train_lstm_model(symbol)
                        st.success("Model trained successfully!")
                    except Exception as e:
                        st.error(f"Model training failed: {str(e)}")
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating price predictions..."):
                try:
                    print(f"Starting prediction for symbol: {symbol}")
                    # Check if model exists
                    if not os.path.exists('lstm_model.h5'):
                        st.warning("No trained model found. Training new model...")
                        train_lstm_model(symbol)
                    
                    # Get historical data for training
                    historical_data = get_stock_data(symbol, "1y")
                    if historical_data.empty:
                        st.error("No historical data available for predictions.")
                        return

                    # Prepare data for prediction
                    try:
                        # Create a new DataFrame with single-level columns
                        pred_data = pd.DataFrame({
                            'Close': historical_data['Close', symbol]
                        }, index=historical_data.index)
                        
                        # Calculate technical indicators
                        pred_data['SMA20'] = ta.sma(pred_data['Close'], length=20)
                        pred_data['RSI'] = ta.rsi(pred_data['Close'], length=14)
                        
                        # Prepare data for LSTM model
                        sequence_length = 60  # Number of days to look back
                        data = pred_data['Close'].values.reshape(-1, 1)
                        
                        # Normalize the data
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(data)
                        
                        # Create sequences for prediction
                        X = []
                        # Only use the last sequence for prediction
                        last_sequence = scaled_data[-sequence_length:]
                        X.append(last_sequence)
                        X = np.array(X)
                        
                        # Load and make predictions
                        model = tf.keras.models.load_model('lstm_model.h5')
                        
                        # Create future predictions
                        future_predictions = []
                        current_sequence = last_sequence.copy()
                        
                        # Get the last actual price for continuity
                        last_actual_price = pred_data['Close'].iloc[-1]
                        
                        for _ in range(30):
                            # Predict the next value
                            next_pred = model.predict(current_sequence.reshape(1, sequence_length, 1))
                            future_predictions.append(next_pred[0, 0])
                            # Update the sequence
                            current_sequence = np.roll(current_sequence, -1)
                            current_sequence[-1] = next_pred[0, 0]
                        
                        # Convert predictions to the same scale as the original data
                        future_predictions = np.array(future_predictions).reshape(-1, 1)
                        future_predictions = scaler.inverse_transform(future_predictions)
                        
                        # Adjust the first prediction to start from the last actual price
                        price_diff = last_actual_price - future_predictions[0][0]
                        future_predictions = future_predictions + price_diff
                        
                        # Create future dates starting from the last historical date
                        last_date = pred_data.index[-1]
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
                        
                        # Create prediction DataFrame
                        pred_df = pd.DataFrame({
                            'Predicted_Price': future_predictions.flatten()
                        }, index=future_dates)
                        
                        # Plot historical and predicted prices
                        fig = go.Figure()
                        
                        # Add historical prices first
                        fig.add_trace(go.Scatter(
                            x=pred_data.index,
                            y=pred_data['Close'],
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add predicted prices second
                        fig.add_trace(go.Scatter(
                            x=pred_df.index,
                            y=pred_df['Predicted_Price'],
                            name='Predicted',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{symbol} Price Prediction",
                            yaxis_title="Price",
                            xaxis_title="Date",
                            height=600,
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display prediction metrics
                        last_price = pred_data['Close'].iloc[-1]
                        # Use the last predicted price instead of the first one for the change calculation
                        final_pred_price = future_predictions[-1][0]  # Last predicted price
                        change = ((final_pred_price - last_price) / last_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"${last_price:.2f}")
                        col2.metric("Predicted Price (30 days)", f"${final_pred_price:.2f}")
                        col3.metric("Predicted Change", f"{change:.2f}%")
                        
                        # Display prediction confidence based on the predicted trend
                        confidence = min(100, max(0, abs(change) * 2))  # Simple confidence metric
                        st.progress(confidence/100)
                        st.write(f"Prediction Confidence: {confidence:.1f}%")
                        
                    except Exception as e:
                        print(f"Error in prediction process: {str(e)}")
                        st.error("Failed to generate predictions. Please try again.")
                        
                except Exception as e:
                    print(f"Prediction failed with error: {str(e)}")
                    print(f"Error type: {type(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    st.error(f"Prediction failed: {str(e)}")

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