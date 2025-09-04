import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Stablecoin Health Monitor Pro", 
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stRadio > div {
        display: flex;
        justify-content: center;
        margin-bottom: 25px;
        gap: 10px;
    }
    .stRadio > div > label {
        margin: 0 10px;
        font-size: 16px;
        font-weight: 500;
        color: #FFFFFF;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 12px 24px;
        border-radius: 12px;
        transition: all 0.3s ease;
        cursor: pointer;
        border: 2px solid transparent;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    .stRadio > div > label:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border: 2px solid #00FFF0;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .health-score-excellent {
        color: #00FF88;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    .health-score-good {
        color: #4ECDC4;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
    }
    .health-score-warning {
        color: #FFD93D;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 217, 61, 0.5);
    }
    .health-score-critical {
        color: #FF6B6B;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
    }
    
    .status-indicator {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-excellent {
        background: rgba(0, 255, 136, 0.2);
        color: #00FF88;
        border: 1px solid #00FF88;
    }
    .status-good {
        background: rgba(78, 205, 196, 0.2);
        color: #4ECDC4;
        border: 1px solid #4ECDC4;
    }
    .status-warning {
        background: rgba(255, 217, 61, 0.2);
        color: #FFD93D;
        border: 1px solid #FFD93D;
    }
    .status-critical {
        background: rgba(255, 107, 107, 0.2);
        color: #FF6B6B;
        border: 1px solid #FF6B6B;
    }
    
    .data-freshness {
        background: rgba(102, 126, 234, 0.1);
        padding: 8px 16px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        font-size: 14px;
        color: #B8BCC8;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 15px 0;
    }
    
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    
    .alert-banner {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
@st.cache_data(ttl=86400)  # Cache for 24 hours to conserve API credits
def get_api_keys():
    return {
        'coingecko': os.getenv("COINGECKO_PRO_API_KEY"),
        'dune': os.getenv("DEFI_JOSH_DUNE_QUERY_API_KEY")
    }

# Data fetching functions
@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_stablecoin_data():
    """Fetch current stablecoin data from CoinGecko"""
    api_keys = get_api_keys()
    if not api_keys['coingecko']:
        # Fallback to static data if no API key
        try:
            return joblib.load("data/stablecoins_filtered.joblib")
        except:
            return pd.DataFrame()
    
    url = "https://pro-api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "category": "stablecoins",
        "order": "market_cap_desc",
        "per_page": 50,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h,7d,30d"
    }
    headers = {"x-cg-pro-api-key": api_keys['coingecko']}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        
        # Filter for major stablecoins
        major_stables = ['usdt', 'usdc', 'dai', 'busd', 'tusd', 'frax', 'lusd', 'susd', 'pyusd', 'rlusd', 'usds']
        filtered_df = df[df['symbol'].isin(major_stables)].copy()
        
        # Save to cache
        try:
            joblib.dump(filtered_df, "data/stablecoins_filtered.joblib")
        except:
            pass
            
        return filtered_df
    except Exception as e:
        st.error(f"Error fetching CoinGecko data: {e}")
        # Fallback to cached data
        try:
            return joblib.load("data/stablecoins_filtered.joblib")
        except:
            return pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_dune_onchain_data():
    """Fetch on-chain mint/burn data from Dune Analytics"""
    api_keys = get_api_keys()
    if not api_keys['dune']:
        try:
            return joblib.load("data/dune_onchain_data.joblib")
        except:
            return pd.DataFrame()
    
    # Dune API endpoint for query execution
    query_id = 5681885  # Your Dune query ID
    
    try:
        url = f"https://api.dune.com/api/v1/query/{query_id}/results"
        headers = {"X-Dune-API-Key": api_keys['dune']}
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'result' in data and 'rows' in data['result']:
            df = pd.DataFrame(data['result']['rows'])
            
            # Process the data
            if 'week' in df.columns:
                df['week'] = pd.to_datetime(df['week'])
                df = df.sort_values('week', ascending=False)
            
            # Save to cache
            try:
                joblib.dump(df, "data/dune_onchain_data.joblib")
            except:
                pass
                
            return df
        else:
            st.error("Unexpected response format from Dune API")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching Dune data: {e}")
        # Fallback to cached data
        try:
            return joblib.load("data/dune_onchain_data.joblib")
        except:
            return pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_dominance_data():
    """Calculate market dominance for stablecoins"""
    df = fetch_stablecoin_data()
    if df.empty:
        try:
            return joblib.load("data/current_dominance_df.joblib")
        except:
            return pd.DataFrame()
    
    # Get total crypto market cap
    api_keys = get_api_keys()
    if api_keys['coingecko']:
        try:
            global_url = "https://pro-api.coingecko.com/api/v3/global"
            headers = {"x-cg-pro-api-key": api_keys['coingecko']}
            response = requests.get(global_url, headers=headers, timeout=10)
            global_data = response.json()
            total_market_cap = global_data['data']['total_market_cap']['usd']
        except:
            total_market_cap = 2.5e12  # Fallback estimate
    else:
        total_market_cap = 2.5e12
    
    # Calculate dominance
    dominance_data = []
    total_stable_cap = df['market_cap'].sum()
    
    for _, row in df.iterrows():
        if pd.notna(row['market_cap']) and row['market_cap'] > 0:
            crypto_dominance = (row['market_cap'] / total_market_cap) * 100
            stable_dominance = (row['market_cap'] / total_stable_cap) * 100
            
            dominance_data.append({
                'Stablecoin': row['symbol'].upper(),
                'Crypto Dominance (%)': crypto_dominance,
                'Stable Dominance (%)': stable_dominance,
                'Market Cap': row['market_cap']
            })
    
    dominance_df = pd.DataFrame(dominance_data)
    return dominance_df.sort_values('Market Cap', ascending=False)

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_historical_peg_data():
    """Fetch historical peg deviation data"""
    api_keys = get_api_keys()
    if not api_keys['coingecko']:
        try:
            return joblib.load("data/stablecoins_historical_deviation.joblib")
        except:
            return {}
    
    coingecko_ids = {
        'USDT': 'tether',
        'USDC': 'usd-coin',
        'DAI': 'dai',
        'BUSD': 'binance-usd',
        'TUSD': 'true-usd',
        'FRAX': 'frax',
        'LUSD': 'liquity-usd',
        'SUSD': 'nusd',
        'PYUSD': 'paypal-usd'
    }
    
    historical_data = {}
    headers = {"x-cg-pro-api-key": api_keys['coingecko']}
    
    for symbol, coin_id in coingecko_ids.items():
        try:
            url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {"vs_currency": "usd", "days": 30}  # Last 30 days for faster loading
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            prices = data.get("prices", [])
            if prices:
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["peg_deviation_usd"] = df["price"] - 1
                df["peg_deviation_pct"] = (df["price"] - 1) * 100
                historical_data[symbol] = df
        except Exception as e:
            st.warning(f"Could not fetch historical data for {symbol}: {e}")
    
    # Save to cache
    try:
        joblib.dump(historical_data, "data/stablecoins_historical_deviation.joblib")
    except:
        pass
        
    return historical_data

def calculate_health_score(price, market_cap_change_24h, volatility_30d=None):
    """Calculate a composite health score for stablecoins"""
    score = 100
    
    # Peg deviation penalty (0-40 points)
    peg_deviation = abs(price - 1.0)
    if peg_deviation > 0.1:  # >10% deviation
        score -= 40
    elif peg_deviation > 0.05:  # >5% deviation
        score -= 25
    elif peg_deviation > 0.01:  # >1% deviation
        score -= 10
    elif peg_deviation > 0.005:  # >0.5% deviation
        score -= 5
    
    # Market cap stability (0-30 points)
    if pd.notna(market_cap_change_24h):
        abs_change = abs(market_cap_change_24h)
        if abs_change > 10:
            score -= 30
        elif abs_change > 5:
            score -= 20
        elif abs_change > 2:
            score -= 10
    
    # Volatility penalty if available (0-30 points)
    if volatility_30d and volatility_30d > 5:
        score -= 30
    elif volatility_30d and volatility_30d > 2:
        score -= 15
    
    return max(0, min(100, score))

def get_health_status(score):
    """Get health status and color based on score"""
    if score >= 80:
        return "Excellent", "health-score-good"
    elif score >= 60:
        return "Warning", "health-score-warning"
    else:
        return "Critical", "health-score-critical"

def generate_market_insights(stablecoin_df, dominance_df, onchain_df):
    """Generate market insights based on current data"""
    insights = []
    
    if stablecoin_df.empty:
        return ["⚠️ No data available for analysis"]
    
    # Market concentration analysis
    top_3_cap = stablecoin_df.nlargest(3, 'market_cap')['market_cap'].sum()
    total_cap = stablecoin_df['market_cap'].sum()
    concentration = (top_3_cap / total_cap) * 100
    
    if concentration > 80:
        insights.append(f"🎯 **High Market Concentration**: Top 3 stablecoins control {concentration:.1f}% of the market, indicating potential systemic risk.")
    
    # Peg deviation analysis
    deviations = [(row['current_price'] - 1) * 100 for _, row in stablecoin_df.iterrows()]
    avg_deviation = np.mean([abs(d) for d in deviations])
    
    if avg_deviation > 0.5:
        insights.append(f"⚠️ **Elevated Peg Risk**: Average deviation is {avg_deviation:.3f}%, above normal ranges.")
    elif avg_deviation < 0.1:
        insights.append(f"✅ **Strong Peg Stability**: Average deviation only {avg_deviation:.3f}%, excellent stability.")
    
    # Volatility insights
    volatile_coins = stablecoin_df[stablecoin_df['price_change_percentage_24h'].abs() > 2]
    if len(volatile_coins) > 0:
        insights.append(f"📈 **Volatility Alert**: {len(volatile_coins)} stablecoin(s) showing >2% price movement in 24h.")
    
    # Market cap insights
    growing_coins = stablecoin_df[stablecoin_df['market_cap_change_percentage_24h'] > 5]
    shrinking_coins = stablecoin_df[stablecoin_df['market_cap_change_percentage_24h'] < -5]
    
    if len(growing_coins) > len(shrinking_coins):
        insights.append("📊 **Net Supply Expansion**: More stablecoins are growing than shrinking, indicating market confidence.")
    elif len(shrinking_coins) > len(growing_coins):
        insights.append("📉 **Net Supply Contraction**: More stablecoins shrinking than growing, potential market stress.")
    
    return insights if insights else ["📊 All metrics within normal ranges"]

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 16px; margin-bottom: 25px; text-align: center;">
        <h2 style="color: white; margin: 0; font-weight: 700;">🚀 Stablecoin Monitor Pro</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 14px;">Advanced Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration status
    api_keys = get_api_keys()
    config_messages = []
    
    if not api_keys['coingecko']:
        config_messages.append("⚠️ CoinGecko Pro API key not found")
    
    if not api_keys['dune']:
        config_messages.append("⚠️ Dune API key not found")
    
    if config_messages:
        st.markdown("### ⚙️ Configuration Status")
        for message in config_messages:
            st.markdown(f"<div class='data-freshness'>{message}</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Why Monitor Stablecoin Health?

    Stablecoins promise $1.00 stability, but **depegging events can be catastrophic**:

    **🚨 Historical Failures:**
    - **UST**: Collapsed from $1.00 to $0.10 in May 2022
    - **BUSD**: Regulatory pressure caused mass redemptions  
    - **USDC**: Briefly depegged during Silicon Valley Bank crisis

    **📊 What We Track:**
    - **Real-time Peg Deviations** - Distance from $1.00
    - **On-chain Supply Changes** - Mints, burns, and flows
    - **Market Dominance** - Concentration risks
    - **Health Scoring** - Composite risk assessment

    ---

    **🔍 Monitored Stablecoins:**
    - **USDT, USDC** (Market Dominants)
    - **DAI, FRAX** (DeFi Protocols) 
    - **BUSD, TUSD** (Traditional Backed)
    - **PYUSD, RLUSD, USDS** (Next Generation)
    - **LUSD, sUSD** (Specialized Mechanisms)

    Each uses different backing mechanisms - from cash reserves to algorithmic protocols to over-collateralization.

    **📡 Data Sources:**
    - 🦎 **CoinGecko Pro** - Market data
    - ⚡ **Dune Analytics** - On-chain metrics
    - 🔄 **Auto-refresh**: Every 24 hours
    - 💾 **Fallback cache** for reliability
    """)
    
    # Data freshness indicator
    st.markdown("""
    <div class='data-freshness'>
        <strong>🕐 Data Refresh:</strong><br>
        Next update: 24 hours from last fetch<br>
        <small>Optimized for API credit conservation</small>
    </div>
    """, unsafe_allow_html=True)

# Enhanced main header
current_time = datetime.now().strftime('%B %d, %Y at %H:%M UTC')
st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); 
           padding: 30px; border-radius: 20px; margin-bottom: 30px; text-align: center;
           box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
    <h1 style="color: white; margin: 0; font-size: 3em; font-weight: 800; 
               text-shadow: 0 4px 20px rgba(0,0,0,0.3); letter-spacing: -1px;">
        💰 Stablecoin Health Monitor Pro
    </h1>
    <p style="color: rgba(255,255,255,0.95); margin: 15px 0 5px 0; font-size: 1.3em; font-weight: 500;">
        Advanced Real-time Stability Analysis & Market Intelligence
    </p>
    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 1em;">
        📅 {current_time}
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced navigation
section = st.radio(
    "",
    ["📊 Market Overview", "⚖️ Peg Stability", "📦 Supply Analysis", "⚡ On-Chain Activity"],
    horizontal=True,
    key="nav_radio"
)

# Load data
with st.spinner("🔄 Loading comprehensive market data..."):
    progress_bar = st.progress(0)
    
    progress_bar.progress(25)
    stablecoin_df = fetch_stablecoin_data()
    
    progress_bar.progress(50)
    dominance_df = fetch_dominance_data()
    
    progress_bar.progress(75)
    historical_peg_data = fetch_historical_peg_data()
    
    progress_bar.progress(90)
    onchain_df = fetch_dune_onchain_data()
    
    progress_bar.progress(100)
    progress_bar.empty()

# Data quality alerts
if stablecoin_df.empty:
    st.markdown("""
    <div class='alert-banner'>
        🚨 <strong>Data Alert:</strong> Unable to fetch fresh market data. Showing cached information.
    </div>
    """, unsafe_allow_html=True)

# === MARKET OVERVIEW SECTION ===
if section == "📊 Market Overview":
    st.markdown("## 📊 Market Overview")
    
    if not stablecoin_df.empty:
        # Generate insights
        insights = generate_market_insights(stablecoin_df, dominance_df, onchain_df)
        
        # Insights panel
        st.markdown("### 🧠 Market Insights")
        insight_container = st.container()
        with insight_container:
            cols = st.columns(len(insights) if len(insights) <= 3 else 3)
            for i, insight in enumerate(insights[:3]):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class='insight-box'>
                        <p style="margin: 0; font-size: 14px; line-height: 1.5;">{insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_market_cap = stablecoin_df['market_cap'].sum()
        avg_price = stablecoin_df['current_price'].mean()
        total_stablecoins = len(stablecoin_df)
        avg_deviation = np.mean([abs(price - 1) * 100 for price in stablecoin_df['current_price']])
        
        # Calculate overall health score
        health_scores = []
        for _, row in stablecoin_df.iterrows():
            health_score = calculate_health_score(
                row['current_price'],
                row.get('market_cap_change_percentage_24h'),
            )
            health_scores.append(health_score)
        
        avg_health_score = np.mean(health_scores) if health_scores else 0
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Total Market Cap</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #00FF88;">${total_market_cap/1e9:.1f}B</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Average Price</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #4ECDC4;">${avg_price:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Tracked Assets</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #FFD93D;">{total_stablecoins}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Avg. Deviation</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #FF6B6B;">{avg_deviation:.3f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            if avg_health_score >= 80:
                score_class = "health-score-good"
            elif avg_health_score >= 60:
                score_class = "health-score-warning"
            else:
                score_class = "health-score-critical"
                
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Avg. Health Score</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700;" class="{score_class}">{avg_health_score:.1f}/100</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Market dominance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            if not dominance_df.empty:
                fig_bar = px.bar(
                    dominance_df.head(8),
                    x='Stablecoin',
                    y='Stable Dominance (%)',
                    title="Market Dominance by Stablecoin",
                    color='Stable Dominance (%)',
                    color_continuous_scale=['#8A4AF3', '#00FFF0', '#9945FF']
                )
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            if not dominance_df.empty:
                fig_pie = px.pie(
                    dominance_df.head(6),
                    names='Stablecoin',
                    values='Stable Dominance (%)',
                    title="Market Share Distribution",
                    hole=0.4,
                    color_discrete_sequence=['#00FFF0', '#8A4AF3', '#9945FF', '#FF6B6B', '#4ECDC4', '#45B7D1']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed stablecoin table with health scores
        st.markdown("### 🏥 Health Status Dashboard")
        
        health_data = []
        for _, row in stablecoin_df.iterrows():
            score = calculate_health_score(
                row['current_price'],
                row.get('market_cap_change_percentage_24h'),
            )
            status, css_class = get_health_status(score)
            
            health_data.append({
                'Symbol': row['symbol'].upper(),
                'Name': row['name'],
                'Price': row['current_price'],
                'Peg Deviation': (row['current_price']-1)*100,
                'Market Cap': row['market_cap'],
                '24h Change': row.get('market_cap_change_percentage_24h', 0),
                'Health Score': score,
                'Status': status
            })
        
        health_df = pd.DataFrame(health_data)
        health_df = health_df.sort_values('Health Score', ascending=False)
        
        # Format for display
        display_df = health_df.copy()
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.4f}")
        display_df['Peg Deviation'] = display_df['Peg Deviation'].apply(lambda x: f"{x:+.3f}%")
        display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B")
        display_df['24h Change'] = display_df['24h Change'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
        display_df['Health Score'] = display_df['Health Score'].apply(lambda x: f"{x:.1f}/100")
        
        # Reset index to start from 1 instead of 0
        display_df.index = range(1, len(display_df) + 1)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Symbol": "Symbol",
                "Name": "Name",
                "Price": "Price ($)",
                "Peg Deviation": "Peg Deviation",
                "Market Cap": "Market Cap",
                "24h Change": "24h Change",
                "Health Score": "Health Score",
                "Status": "Status"
            }
        )

# === PEG STABILITY SECTION ===
elif section == "⚖️ Peg Stability":
    st.markdown("## ⚖️ Peg Stability Analysis")
    
    if historical_peg_data and stablecoin_df is not None and not stablecoin_df.empty:
        # Current peg deviation overview
        st.markdown("### 🎯 Current Peg Deviations")
        
        peg_cols = st.columns(min(4, len(stablecoin_df)))
        for i, (_, coin) in enumerate(stablecoin_df.head(4).iterrows()):
            if i < len(peg_cols):
                deviation = (coin['current_price'] - 1) * 100
                
                with peg_cols[i]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">{coin['symbol'].upper()}</h3>
                        <p style="margin: 0; font-size: 24px; font-weight: 700; color: #4ECDC4;">${coin['current_price']:.4f}</p>
                        <p style="margin: 5px 0 0 0; font-size: 14px; color: {'#FF6B6B' if abs(deviation) > 0.5 else '#FFD93D' if abs(deviation) > 0.1 else '#00FF88'};">
                            {deviation:+.3f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Historical peg deviation chart
        st.markdown("### 📊 Historical Peg Deviation Trends")
        
        if historical_peg_data:
            fig = go.Figure()
            
            colors = ['#00FFF0', '#8A4AF3', '#9945FF', '#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for i, (symbol, df_hist) in enumerate(historical_peg_data.items()):
                if not df_hist.empty:
                    fig.add_trace(go.Scatter(
                        x=df_hist['date'],
                        y=df_hist['peg_deviation_pct'],
                        mode='lines',
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            # Add peg line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Perfect Peg ($1.00)")
            fig.add_hline(y=1, line_dash="dot", line_color="orange", opacity=0.7)
            fig.add_hline(y=-1, line_dash="dot", line_color="orange", opacity=0.7)
            
            fig.update_layout(
                title="Peg Deviation Over Time (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="Deviation from $1.00 (%)",
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility analysis
            st.markdown("### 📈 Volatility Analysis")
            
            volatility_data = []
            for symbol, df_hist in historical_peg_data.items():
                if not df_hist.empty and len(df_hist) > 1:
                    volatility = df_hist['peg_deviation_pct'].std()
                    max_deviation = df_hist['peg_deviation_pct'].abs().max()
                    
                    volatility_data.append({
                        'Stablecoin': symbol,
                        'Volatility (%)': volatility,
                        'Max Deviation (%)': max_deviation,
                        'Stability Rank': 'High' if volatility < 0.1 else 'Medium' if volatility < 0.5 else 'Low'
                    })
            
            if volatility_data:
                vol_df = pd.DataFrame(volatility_data).sort_values('Volatility (%)')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_vol = px.bar(
                        vol_df,
                        x='Stablecoin',
                        y='Volatility (%)',
                        title="Price Volatility by Stablecoin",
                        color='Volatility (%)',
                        color_continuous_scale=['#00FF88', '#FFAA00', '#FF4444']
                    )
                    fig_vol.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white'
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                with col2:
                    # Format for display
                    vol_display_df = vol_df.copy()
                    vol_display_df['Volatility (%)'] = vol_display_df['Volatility (%)'].apply(lambda x: f"{x:.3f}%")
                    vol_display_df['Max Deviation (%)'] = vol_display_df['Max Deviation (%)'].apply(lambda x: f"{x:.3f}%")
                    
                    # Reset index to start from 1 instead of 0
                    vol_display_df.index = range(1, len(vol_display_df) + 1)
                    
                    st.dataframe(
                        vol_display_df,
                        use_container_width=True,
                        column_config={
                            "Stablecoin": "Stablecoin",
                            "Volatility (%)": "Volatility (%)",
                            "Max Deviation (%)": "Max Deviation (%)",
                            "Stability Rank": "Stability Rank"
                        }
                    )

# === SUPPLY ANALYSIS SECTION ===
elif section == "📦 Supply Analysis":
    st.markdown("## 📦 Supply Analysis")
    
    if not stablecoin_df.empty:
        # Current supply metrics
        st.markdown("### 💰 Current Supply Metrics")
        
        # Calculate supply metrics
        total_supply_value = stablecoin_df['market_cap'].sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Total Stablecoin Supply</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #00FF88;">${total_supply_value/1e9:.1f}B</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            supply_concentration = dominance_df.head(3)['Stable Dominance (%)'].sum() if not dominance_df.empty else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Top 3 Concentration</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #4ECDC4;">{supply_concentration:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_24h_supply_change = stablecoin_df['market_cap_change_percentage_24h'].mean()
            if pd.notna(avg_24h_supply_change):
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Avg 24h Supply Change</h3>
                    <p style="margin: 0; font-size: 24px; font-weight: 700; color: {'#FF6B6B' if avg_24h_supply_change < 0 else '#00FF88'};">{avg_24h_supply_change:+.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Avg 24h Supply Change</h3>
                    <p style="margin: 0; font-size: 24px; font-weight: 700; color: #FFD93D;">N/A</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Supply distribution visualization
        st.markdown("### 📊 Supply Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Market cap comparison
            top_supplies = stablecoin_df.nlargest(8, 'market_cap')
            
            fig_supply = px.bar(
                top_supplies,
                x='symbol',
                y='market_cap',
                title="Current Supply by Market Cap",
                labels={'market_cap': 'Market Cap (USD)', 'symbol': 'Stablecoin'},
                color='market_cap',
                color_continuous_scale=['#8A4AF3', '#00FFF0', '#9945FF']
            )
            fig_supply.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_supply, use_container_width=True)
        
        with col2:
            # 24h change analysis
            change_data = stablecoin_df.dropna(subset=['market_cap_change_percentage_24h'])
            
            if not change_data.empty:
                fig_change = px.bar(
                    change_data,
                    x='symbol',
                    y='market_cap_change_percentage_24h',
                    title="24h Supply Changes",
                    labels={'market_cap_change_percentage_24h': '24h Change (%)', 'symbol': 'Stablecoin'},
                    color='market_cap_change_percentage_24h',
                    color_continuous_scale=['#FF4444', '#FFAA00', '#00FF88']
                )
                fig_change.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_change, use_container_width=True)
        
        # Detailed supply table
        st.markdown("### 📋 Detailed Supply Information")
        
        supply_table_data = []
        for _, row in stablecoin_df.iterrows():
            supply_table_data.append({
                'Stablecoin': row['symbol'].upper(),
                'Name': row['name'],
                'Market Cap': row['market_cap'],
                'Market Cap Rank': int(row['market_cap_rank']) if pd.notna(row['market_cap_rank']) else 'N/A',
                '24h Change': row.get('market_cap_change_percentage_24h', 0),
                'Price': row['current_price'],
                'Last Updated': pd.to_datetime(row['last_updated']).strftime('%Y-%m-%d %H:%M UTC')
            })
        
        supply_df_display = pd.DataFrame(supply_table_data)
        supply_df_display = supply_df_display.sort_values('Market Cap Rank')
        
        # Format for display
        supply_df_display['Market Cap'] = supply_df_display['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B")
        supply_df_display['24h Change'] = supply_df_display['24h Change'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
        supply_df_display['Price'] = supply_df_display['Price'].apply(lambda x: f"${x:.4f}")
        
        # Reset index to start from 1 instead of 0
        supply_df_display.index = range(1, len(supply_df_display) + 1)
        
        st.dataframe(supply_df_display, use_container_width=True)
    
    else:
        st.error("Unable to load stablecoin data. Please check your API configuration.")

# === ON-CHAIN ACTIVITY SECTION ===
elif section == "⚡ On-Chain Activity":
    st.markdown("## ⚡ On-Chain Activity Analysis")
    
    if not onchain_df.empty:
        # Display latest on-chain data
        st.markdown("### 📊 Latest On-Chain Supply Changes")
        
        # Get the latest week's data
        if 'week' in onchain_df.columns:
            latest_data = onchain_df.iloc[0]
            week = pd.to_datetime(latest_data['week']).strftime('%B %d, %Y')
            
            st.markdown(f"#### Week of {week}")
            
            # Create metrics for major stablecoins
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'USDC supply' in latest_data:
                    usdc_supply = latest_data['USDC supply']
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">USDC Supply</h3>
                        <p style="margin: 0; font-size: 24px; font-weight: 700; color: #4ECDC4;">${usdc_supply/1e9:.2f}B</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if 'DAI supply' in latest_data:
                    dai_supply = latest_data['DAI supply']
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">DAI Supply</h3>
                        <p style="margin: 0; font-size: 24px; font-weight: 700; color: #FFD93D;">${dai_supply/1e9:.2f}B</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if 'LUSD supply' in latest_data:
                    lusd_supply = latest_data['LUSD supply']
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">LUSD Supply</h3>
                        <p style="margin: 0; font-size: 24px; font-weight: 700; color: #00FF88;">${lusd_supply/1e9:.2f}B</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Historical supply trends
        st.markdown("### 📈 Historical Supply Trends")
        
        # Prepare data for plotting
        if 'week' in onchain_df.columns:
            plot_df = onchain_df.head(12).copy()  # Last 12 weeks
            plot_df = plot_df.sort_values('week')  # Sort chronologically
            
            # Create line chart
            fig = go.Figure()
            
            # Add traces for each stablecoin
            if 'USDC supply' in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df['week'], 
                    y=plot_df['USDC supply']/1e9,
                    name='USDC',
                    line=dict(color='#4ECDC4', width=3),
                    mode='lines+markers'
                ))
            
            if 'DAI supply' in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df['week'], 
                    y=plot_df['DAI supply']/1e9,
                    name='DAI',
                    line=dict(color='#FFD93D', width=3),
                    mode='lines+markers'
                ))
            
            if 'LUSD supply' in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df['week'], 
                    y=plot_df['LUSD supply']/1e9,
                    name='LUSD',
                    line=dict(color='#00FF88', width=3),
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                height=500,
                title='Historical Stablecoin Supply (Billions USD)',
                xaxis_title="Week",
                yaxis_title="Supply (Billions USD)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed on-chain data table
        st.markdown("### 📋 Historical On-Chain Data")
        
        # Prepare data for display
        onchain_display_df = onchain_df.copy()
        
        # Format columns
        for col in onchain_display_df.columns:
            if col != 'week' and 'supply' in col:
                onchain_display_df[col] = onchain_display_df[col].apply(lambda x: f"${x/1e9:.2f}B")
        
        # Format date
        if 'week' in onchain_display_df.columns:
            onchain_display_df['week'] = onchain_display_df['week'].dt.strftime('%Y-%m-%d')
        
        # Reset index to start from 1 instead of 0
        onchain_display_df.index = range(1, len(onchain_display_df) + 1)
        
        # Display the table
        st.dataframe(
            onchain_display_df,
            use_container_width=True
        )
    else:
        st.warning("No on-chain data available. Please check your Dune API configuration.")

# Footer
st.markdown("---")
last_updated = datetime.now().strftime('%B %d, %Y at %H:%M:%S UTC')
st.markdown(f"""
<div style="text-align: center; color: #B8BCC8; font-size: 14px; margin-top: 30px;">
    <p>📊 <strong>Stablecoin Health Monitor Pro</strong> | Real-time data from CoinGecko Pro API & Dune Analytics</p>
    <p>⚡ Updates every 24 hours | 🔄 Last updated: {last_updated}</p>
    <p style="font-size: 0.9em;">💡 Remember: Past performance doesn't guarantee future stability</p>
</div>
""", unsafe_allow_html=True)