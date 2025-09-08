# 💰 Stablecoin Health Monitor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stablecoin-health-monitor.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive real-time dashboard for monitoring stablecoin stability, supply dynamics, and market health across major stablecoins.

## 🌟 Overview

Stablecoins promise $1.00 stability but face constant market pressures. This dashboard provides professional-grade analytics to track peg deviations, supply changes, and overall health metrics across the stablecoin ecosystem.

**Live Dashboard:** [stablecoin-health-monitor.streamlit.app](https://stablecoin-health-monitor.streamlit.app)

## 📊 Key Features

### 🎯 Peg Stability Monitoring
- Real-time deviation tracking from $1.00 target
- 30-day historical volatility analysis  
- Risk assessment with automated alerts
- Stability ranking system

### 📦 Supply Dynamics Analysis
- Market cap distribution and dominance metrics
- 24-hour supply change tracking
- Concentration analysis (top 3 control metrics)
- Growth/contraction trend identification

### 🏥 Health Scoring System
- Composite health scores (0-100) based on multiple factors:
  - Peg deviation penalties
  - Market cap stability
  - Trading volume considerations  
  - Volatility assessments
- Color-coded health status (Excellent/Good/Warning/Critical)

### ⛓️ On-Chain Activity (Ethereum)
- Historical supply evolution trends
- Week-over-week mint/burn analysis
- On-chain supply distribution
- Integration with Dune Analytics

## 🎯 Supported Stablecoins

| Stablecoin | Symbol | Type | Backing |
|------------|--------|------|---------|
| Tether | USDT | Centralized | Cash/Equivalents |
| USD Coin | USDC | Centralized | Cash/Equivalents |
| Dai | DAI | Decentralized | Crypto-Collateral |
| Binance USD | BUSD | Centralized | Cash/Equivalents |
| TrueUSD | TUSD | Centralized | Cash/Equivalents |
| Frax | FRAX | Algorithmic | Hybrid Model |
| Liquity USD | LUSD | Decentralized | ETH-Collateral |
| sUSD | SUSD | Synthetic | SNX-Collateral |
| PayPal USD | PYUSD | Centralized | Cash/Equivalents |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CoinGecko Pro API key
- Dune Analytics API key (optional, for on-chain data)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stablecoin-health-monitor.git
cd stablecoin-health-monitor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
COINGECKO_PRO_API_KEY=your_coingecko_pro_api_key_here
DEFI_JOSH_DUNE_QUERY_API_KEY=your_dune_api_key_here
```

4. **Run the application**
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## 🔧 Configuration

### API Setup

#### CoinGecko Pro API (Required)
1. Sign up at [CoinGecko Pro](https://www.coingecko.com/en/api/pricing)
2. Get your API key from the dashboard
3. Add to `.env` file or Streamlit secrets

#### Dune Analytics (Optional)
1. Create account at [dune.com](https://dune.com)
2. Get API key from account settings
3. Add to environment variables

**Note:** The app includes fallback data and will function without API keys, but with limited real-time capabilities.

### Deployment on Streamlit Cloud

1. Fork this repository
2. Connect to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard:
   - `COINGECKO_PRO_API_KEY`
   - `DEFI_JOSH_DUNE_QUERY_API_KEY`

## 🏗️ Architecture

### Data Flow
```
CoinGecko Pro API → Market Data → Health Scoring → Dashboard
Dune Analytics → On-chain Data → Supply Analysis → Visualizations
```

### Caching Strategy
- **24-hour cache cycle** - automatic refresh every 24 hours
- **Global cache with threading locks** - prevents multiple API calls from concurrent users
- **API credit optimization** - maximum 1 call per API per day regardless of traffic
- **Fallback data system** - embedded backup data ensures availability

### Key Components

- **Health Scoring Algorithm**: Multi-factor composite scoring system
- **Real-time Data Pipeline**: Optimized API fetching with error handling
- **Professional UI**: Modern gradient design with interactive visualizations
- **Supply Analytics**: Market cap analysis and dominance calculations

## 📈 Dashboard Sections

### 1. Market Overview
- Key market metrics and live insights
- Market dominance analysis (bar charts & pie charts)
- Comprehensive health analysis table with rankings

### 2. Peg Stability
- Real-time peg status grid
- Historical deviation trends (30 days)
- Volatility rankings and stability scores
- Risk assessment alerts

### 3. Supply Analysis
- Current supply intelligence metrics
- Distribution and change visualizations  
- Detailed supply breakdown table

### 4. On-Chain Activity
- 12-week supply evolution trends
- Current week breakdown
- Week-over-week change analysis

## 🔍 Health Scoring Methodology

The health score (0-100) considers multiple factors:

**Peg Deviation (0-50 points penalty)**
- >10% deviation: -50 points (Critical)
- >5% deviation: -35 points (Severe)
- >2% deviation: -20 points (Moderate)
- >1% deviation: -10 points (Mild)
- >0.5% deviation: -5 points (Minor)

**Market Cap Stability (0-25 points penalty)**
- Based on 24-hour market cap percentage changes
- Higher volatility = higher penalties

**Volume & Liquidity (0-15 points penalty)**
- Low trading volume penalty for coins <$100M daily volume

**Score Ranges:**
- 90-100: Excellent 🟢
- 75-89: Good 🟡
- 60-74: Warning 🟠
- 0-59: Critical 🔴

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Visualizations**: Plotly
- **Data Processing**: Pandas, NumPy
- **APIs**: CoinGecko Pro, Dune Analytics
- **Styling**: Custom CSS with modern gradients
- **Deployment**: Streamlit Cloud

## 📊 Data Sources

- **Market Data**: CoinGecko Pro API
- **On-Chain Data**: Dune Analytics
- **Global Market Data**: CoinGecko Global API
- **Backup Data**: Embedded fallback dataset

## 🚦 Rate Limiting & Optimization

- **API Calls**: Maximum 2 calls per day (1 CoinGecko + 1 Dune)
- **Caching**: Thread-safe 24-hour cache system
- **Error Handling**: Comprehensive fallback mechanisms
- **Performance**: Optimized for high-traffic scenarios

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This dashboard is for informational purposes only and should not be considered as financial advice. Stablecoin investments carry risks, and past performance does not guarantee future stability. Always conduct your own research and consult with financial advisors before making investment decisions.

## 🔗 Links

- **Live Dashboard**: [stablecoin-health-monitor.streamlit.app](https://stablecoin-health-monitor.streamlit.app)
- **Dune Dashboard**: [Stablecoin Supply Analysis](https://dune.com/defi__josh/stablecoin-health-monitor)
- **CoinGecko API**: [pro-api.coingecko.com](https://www.coingecko.com/en/api)

## 📧 Contact

Created by **Jo$h** - DeFi Analytics Specialist

For questions, suggestions, or collaboration opportunities, please send a DM on [Telegram](https://t.me/joshuatochinwachi) or [X](https://x.com/defi__josh).

---

**Remember**: Past performance doesn't guarantee future stability. Use this tool as part of a comprehensive analysis approach.