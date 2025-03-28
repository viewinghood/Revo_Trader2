# 🚀 AI Trading Pro 2 - Enhanced Viewing & Prediction

An advanced trading analysis and portfolio optimization platform built with Streamlit, featuring real-time market data analysis, LSTM-based price predictions, and portfolio optimization.

## 🌟 Features

### 📊 Technical Analysis
- Real-time candlestick charts
- Technical indicators (RSI, SMA, Bollinger Bands)
- Volatility clustering detection
- Market sentiment analysis

### 🔮 Price Predictions
- LSTM-based price forecasting
- 30-day future price predictions
- Confidence metrics
- Model training interface

### ⚖️ Portfolio Optimization
- Multi-asset portfolio optimization
- Risk-adjusted return maximization
- Minimum volatility optimization
- Interactive portfolio allocation visualization

### 📈 Market Data
- Real-time market data integration
- Top gainers and losers tracking
- Most active stocks monitoring
- Sentiment analysis from Finviz

## 🛠️ Requirements

```txt
streamlit>=1.24.0
yfinance>=0.2.18
pandas>=1.5.3
numpy>=1.24.3
plotly>=5.13.1
beautifulsoup4>=4.12.2
pandas-ta>=0.3.14b
pypfopt>=1.5.5
tensorflow>=2.12.0
scikit-learn>=1.2.2
requests>=2.28.2
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/viewinghood/Revo_Trader.git
cd Revo_Trader
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Select your desired asset class and symbol from the sidebar

4. Use the tabs to access different features:
   - 📊 Analysis: Technical analysis and market data
   - 🔮 Predictions: Price forecasting
   - ⚖️ Portfolio: Portfolio optimization
   - 📚 Academy: Trading education

## 🔧 Advanced Features

### Portfolio Optimization
- Select multiple assets
- Choose optimization strategy (Max Sharpe or Min Volatility)
- View optimal allocation and performance metrics
- Interactive portfolio visualization

### Price Predictions
- Train custom LSTM models
- Generate 30-day price forecasts
- View prediction confidence metrics
- Historical vs. predicted price comparison

## 📝 Notes

- The LSTM model is derived on request
- Portfolio optimization requires at least 2 assets
- Market sentiment analysis requires internet connection
- Some features may require additional API keys

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
(Generation of this file out of Cursor)
Personal note: Thanks to the original author and the lot of bugs I have learned so much...THX :) 