# 🚀 AI Trading Pro 2

An advanced trading analysis and prediction platform powered by machine learning and technical analysis.

## 🌟 Features

### 1. Enhanced Technical Analysis
- 'Real-time' market data analysis
- Advanced technical indicators (RSI, MACD, Bollinger Bands)
- Volatility clustering detection
- Multi-timeframe analysis
- Market regime identification

### 2. AI-Powered Predictions
- Basic LSTM model for price prediction
- Enhanced multi-modal model with:
  - Technical indicators integration
  - News sentiment analysis
  - Market regime detection
  - Volume analysis
  - Volatility indicators

### 3. Portfolio Optimization
- Modern portfolio theory implementation
- Risk-adjusted return optimization
- Asset allocation recommendations
- Performance metrics calculation
- Visual portfolio analysis

### 4. Market Sentiment Analysis
- News sentiment integration via Alpha Vantage API
- Social sentiment analysis
- Market mood indicators
- Sentiment-based trading signals

### 5. User Interface
- Interactive charts with Plotly
- Real-time market data updates
- Customizable analysis parameters

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Revo_Trader2.git
cd Revo_Trader
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API keys:
   - Copy `config.py.example` to `config.py`
   - Get an Alpha Vantage API key from https://www.alphavantage.co/
   - Replace `YOUR_API_KEY_HERE` in `config.py` with your actual API key

5. Run the application:
```bash
streamlit run app.py
```

## 📊 Usage

1. **Analysis Tab**
   - Choose between Basic LSTM and Enhanced Multi-Modal models
   - Select asset class and symbol
   - Choose analysis timeframe
   - View technical indicators and market sentiment
   - Get trading signals

2. **Predictions Tab**
   - Choose between Basic LSTM and Enhanced Multi-Modal models
   - Train models with historical data
   - Generate price predictions
   - View confidence metrics

3. **Portfolio Tab**
   - Select multiple assets
   - Optimize portfolio allocation
   - View risk metrics and performance indicators
   - Analyze portfolio composition

4. **Academy Tab**
   - Learn about trading strategies
   - Understand technical indicators
   - Access educational resources
   - Get started with trading

## 🔒 Security

- API keys are stored securely in `config.py`
- `config.py` is excluded from version control
- Sensitive data is never exposed in logs or error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational purposes only. Always do your own research and never invest more than you can afford to lose.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

(Generation of this file out of Cursor)
Personal note: Thanks to the original author and the lot of bugs - I have learned so much...THX :) 