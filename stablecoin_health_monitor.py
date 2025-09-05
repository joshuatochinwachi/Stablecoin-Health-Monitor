#!/usr/bin/env python3
"""
Stablecoin Health Monitor Dashboard
A comprehensive dashboard for monitoring stablecoin stability, supply dynamics, and market health.

Features:
- Real-time market data from CoinGecko Pro API
- On-chain supply data from Dune Analytics  
- Historical peg deviation analysis
- Health scoring system
- Supply trend monitoring
- Professional UI with enhanced visualizations

Author: Jo$h
Date: 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    page_title="Stablecoin Health Monitor", 
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .stRadio > div {
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
        gap: 15px;
    }
    
    .stRadio > div > label {
        font-size: 16px;
        font-weight: 600;
        color: #FFFFFF;
        background: linear-gradient(135deg, #8A4AF3 0%, #9945FF 100%);
        padding: 12px 24px;
        border-radius: 25px;
        border: 2px solid transparent;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(138, 74, 243, 0.3);
    }
    
    .stRadio > div > label:hover {
        background: linear-gradient(135deg, #00FFF0 0%, #8A4AF3 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 255, 240, 0.4);
        border-color: #00FFF0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(138, 74, 243, 0.3);
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #00FFF0;
        box-shadow: 0 12px 40px rgba(0, 255, 240, 0.2);
        transform: translateY(-2px);
    }
    
    .health-score-excellent {
        color: #00FF88;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .health-score-good {
        color: #4ECDC4;
        font-weight: 700;
    }
    
    .health-score-warning {
        color: #FFB347;
        font-weight: 700;
    }
    
    .health-score-critical {
        color: #FF6B6B;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(138, 74, 243, 0.1) 0%, rgba(0, 255, 240, 0.1) 100%);
        border-left: 4px solid #00FFF0;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-style: italic;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 181, 71, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border-left: 4px solid #FFB347;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stDataFrame > div {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration with 24-hour caching to preserve API credits
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_api_keys():
    """Get API keys from environment variables"""
    keys = {
        'coingecko': os.getenv("COINGECKO_PRO_API_KEY"),
        'dune': os.getenv("DEFI_JOSH_DUNE_QUERY_API_KEY")
    }
    return keys

# Data fetching functions
@st.cache_data(ttl=86400)  # 24-hour cache
def fetch_stablecoin_data():
    """Fetch current stablecoin data from CoinGecko with enhanced error handling"""
    api_keys = get_api_keys()
    if not api_keys['coingecko']:
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
    headers = {
        "x-cg-pro-api-key": api_keys['coingecko'],
        "User-Agent": "StablecoinHealthMonitor/1.0"
    }
    
    try:
        with st.spinner("Fetching latest market data..."):
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data)
            major_stables = ['usdt', 'usdc', 'dai', 'busd', 'tusd', 'frax', 'lusd', 'susd', 'pyusd', 'rlusd', 'usds']
            filtered_df = df[df['symbol'].isin(major_stables)].copy()
            
            return filtered_df
    except Exception as e:
        st.warning(f"Using cached data due to API error: {str(e)[:100]}...")
        try:
            return joblib.load("data/stablecoins_filtered.joblib")
        except:
            return pd.DataFrame()

@st.cache_data(ttl=86400)  # 24-hour cache
def fetch_dune_supply_data():
    """Fetch on-chain supply data from Dune Analytics"""
    api_keys = get_api_keys()
    if not api_keys['dune']:
        st.warning("Dune API key not found. Please check your .env file.")
        return pd.DataFrame()
    
    try:
        from dune_client.client import DuneClient
        
        with st.spinner("Fetching on-chain supply data from Dune Analytics..."):
            dune = DuneClient(api_keys['dune'])
            query_result = dune.get_latest_result(5681885)
            
            if query_result and query_result.result and query_result.result.rows:
                df = pd.DataFrame(query_result.result.rows)
                
                # Process the data
                if 'week' in df.columns:
                    df['week'] = pd.to_datetime(df['week'])
                    df = df.sort_values('week', ascending=False)
                
                return df
            else:
                st.warning("No data returned from Dune query")
                return pd.DataFrame()
                
    except Exception as e:
        st.error(f"Error fetching Dune data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)  # 24-hour cache
def fetch_dominance_data():
    """Calculate market dominance with enhanced calculations"""
    df = fetch_stablecoin_data()
    if df.empty:
        try:
            return joblib.load("data/current_dominance_df.joblib")
        except:
            return pd.DataFrame()
    
    api_keys = get_api_keys()
    if api_keys['coingecko']:
        try:
            global_url = "https://pro-api.coingecko.com/api/v3/global"
            headers = {"x-cg-pro-api-key": api_keys['coingecko']}
            response = requests.get(global_url, headers=headers, timeout=15)
            global_data = response.json()
            total_market_cap = global_data['data']['total_market_cap']['usd']
        except:
            total_market_cap = 2.8e12  # Updated market cap estimate
    else:
        total_market_cap = 2.8e12
    
    dominance_data = []
    for _, row in df.iterrows():
        if pd.notna(row['market_cap']) and row['market_cap'] > 0:
            dominance = (row['market_cap'] / total_market_cap) * 100
            dominance_data.append({
                'Stablecoin': row['symbol'].upper(),
                'Dominance (%)': dominance,
                'Market Cap': row['market_cap'],
                'Market Cap Formatted': f"${row['market_cap']/1e9:.2f}B"
            })
    
    dominance_df = pd.DataFrame(dominance_data)
    dominance_df = dominance_df.sort_values('Dominance (%)', ascending=False)
    
    return dominance_df

@st.cache_data(ttl=86400)  # 24-hour cache
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
        'FRAX': 'frax'
    }
    
    historical_data = {}
    headers = {"x-cg-pro-api-key": api_keys['coingecko']}
    
    for symbol, coin_id in coingecko_ids.items():
        try:
            url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {"vs_currency": "usd", "days": 30}
            response = requests.get(url, params=params, headers=headers, timeout=15)
            data = response.json()
            
            prices = data.get("prices", [])
            if prices:
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["peg_deviation_usd"] = df["price"] - 1
                df["peg_deviation_pct"] = (df["price"] - 1) * 100
                historical_data[symbol] = df
        except Exception as e:
            st.warning(f"Could not fetch historical data for {symbol}")
    
    return historical_data

def calculate_enhanced_health_score(price, market_cap_change_24h, volume_24h=None, volatility_30d=None):
    """Enhanced health score algorithm with multiple factors"""
    score = 100
    
    # Peg deviation penalty (0-50 points)
    peg_deviation = abs(price - 1.0)
    if peg_deviation > 0.1:  # >10% deviation - critical
        score -= 50
    elif peg_deviation > 0.05:  # >5% deviation - severe
        score -= 35
    elif peg_deviation > 0.02:  # >2% deviation - moderate
        score -= 20
    elif peg_deviation > 0.01:  # >1% deviation - mild
        score -= 10
    elif peg_deviation > 0.005:  # >0.5% deviation - minor
        score -= 5
    
    # Market cap stability (0-25 points)
    if pd.notna(market_cap_change_24h):
        abs_change = abs(market_cap_change_24h)
        if abs_change > 15:
            score -= 25
        elif abs_change > 10:
            score -= 20
        elif abs_change > 5:
            score -= 15
        elif abs_change > 2:
            score -= 8
    
    # Volume consideration (0-15 points)
    if volume_24h and volume_24h < 100000000:  # Low volume penalty
        score -= 15
    
    # Volatility penalty (0-10 points)
    if volatility_30d:
        if volatility_30d > 5:
            score -= 10
        elif volatility_30d > 2:
            score -= 5
    
    return max(0, min(100, score))

def get_enhanced_health_status(score):
    """Enhanced health status with more granular categories"""
    if score >= 90:
        return "Excellent", "health-score-excellent", "🟢"
    elif score >= 75:
        return "Good", "health-score-good", "🟡"
    elif score >= 60:
        return "Warning", "health-score-warning", "🟠"
    else:
        return "Critical", "health-score-critical", "🔴"

def create_market_insights(df, dominance_df):
    """Generate automated market insights"""
    insights = []
    
    if not df.empty:
        # Price deviation insights
        max_deviation = df['current_price'].apply(lambda x: abs(x - 1)).max()
        if max_deviation > 0.05:
            worst_coin = df.loc[df['current_price'].apply(lambda x: abs(x - 1)).idxmax(), 'symbol'].upper()
            insights.append(f"⚠️ {worst_coin} shows significant peg deviation ({max_deviation*100:.2f}%)")
        
        # Market cap insights
        if not dominance_df.empty:
            top_two_dominance = dominance_df.head(2)['Dominance (%)'].sum()
            if top_two_dominance > 80:
                insights.append(f"📊 Market highly concentrated: Top 2 stablecoins control {top_two_dominance:.1f}%")
            
        # Stability insights
        stable_count = sum(1 for _, row in df.iterrows() if abs(row['current_price'] - 1) < 0.01)
        total_count = len(df)
        stability_ratio = stable_count / total_count
        
        if stability_ratio > 0.8:
            insights.append(f"✅ Strong market stability: {stable_count}/{total_count} coins within 1% peg")
        elif stability_ratio < 0.5:
            insights.append(f"🚨 Market instability detected: Only {stable_count}/{total_count} coins stable")
    
    return insights

# Enhanced sidebar with better organization
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #8A4AF3 0%, #9945FF 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 25px; text-align: center;">
        <h2 style="color: #FFFFFF; margin: 0; font-weight: 700;">💰 Stablecoin Monitor</h2>
        <p style="color: #E0E0E0; margin: 5px 0 0 0; font-size: 0.9em;">Real-time Health Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Are stablecoins truly stable? 🤔
    
    In reality, Stablecoins promise $1.00 stability but face constant market pressures.
    
    **How They Maintain Peg:**
    - 📈 **Above $1:** New tokens minted → Supply increases
    - 📉 **Below $1:** Tokens burned/redeemed → Supply decreases
    
    **Different Approaches:**
    - **💵 Cash-Backed:** USDC, USDT (reserve-based)
    - **🏦 Algorithmic:** DAI (collateralized)
    - **🆕 Hybrid:** FRAX (partial collateral)
    
    ---

    ### 📊 This dashboard Tracks:

    **🎯 Peg Stability**
    - Real-time deviation from $1.00
    - Historical volatility analysis
    - Stability trend monitoring
    
    **📦 Supply Dynamics**  
    - On-chain mint/burn activity. Check [Dune dashboard](https://dune.com/defi__josh/stablecoin-health-monitor) for historical mint/burn and supply trends of each of these stablecoins.
    - Market cap fluctuations
    - Concentration analysis
    
    **🏥 Health Scoring**
    - Composite stability metrics
    - Risk assessment algorithms  
    - Performance rankings
    
    **⛓️ On-Chain Activity**
    - Ethereum contract interactions
    - Supply change patterns
    - DeFi integration levels
    
    ---
    
    **💡 Key Insight:** Each stablecoin reacts differently to stress. Some rely on cash reserves, others on algorithmic mechanisms. Understanding these differences is crucial for risk assessment.
    """)
    
    # Add real-time status indicator
    st.markdown("""
    <div style="background: linear-gradient(90deg, #00FFF0 0%, #8A4AF3 100%); 
                padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;">
        <p style="color: white; margin: 0; font-weight: 600;">🔄 Data Updates</p>
        <p style="color: #E0E0E0; margin: 5px 0 0 0; font-size: 0.85em;">Every 24 hours</p>
        <p style="color: #E0E0E0; margin: 0; font-size: 0.8em;"> </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced main header
st.markdown("""
<div style="background: linear-gradient(135deg, #00FFF0 0%, #8A4AF3 30%, #9945FF 70%, #FF6B6B 100%); 
           padding: 30px; border-radius: 20px; margin-bottom: 40px; text-align: center;
           box-shadow: 0 10px 40px rgba(138, 74, 243, 0.4);">
    <h1 style="color: white; margin: 0; font-size: 3em; font-weight: 700; 
               text-shadow: 2px 2px 8px rgba(0,0,0,0.5);">
        💰 Stablecoin Health Monitor
    </h1>
    <p style="color: #E0E0E0; margin: 15px 0 0 0; font-size: 1.3em; font-weight: 400;">
        Advanced Stability Analysis & Market Intelligence Platform
    </p>
    <p style="color: #B0B0B0; margin: 10px 0 0 0; font-size: 1em;">
        Real-time monitoring • Historical analysis • Risk assessment • Market insights
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation
section = st.radio(
    "",
    ["Market Overview", "Peg Stability", "Supply Analysis", "On-Chain Activity"],
    horizontal=True,
    key="nav_radio"
)

# Load all data with better error handling
with st.spinner("🔄 Loading comprehensive market data..."):
    stablecoin_df = fetch_stablecoin_data()
    dominance_df = fetch_dominance_data()
    historical_peg_data = fetch_historical_peg_data()
    dune_supply_data = fetch_dune_supply_data()

# Generate market insights
market_insights = create_market_insights(stablecoin_df, dominance_df)

# === MARKET OVERVIEW SECTION ===
if section == "Market Overview":
    st.markdown("## 📈 Market Overview & Health Dashboard")
    
    if not stablecoin_df.empty:
        # Display market insights
        if market_insights:
            st.markdown("### 🔍 Live Market Insights")
            for insight in market_insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Enhanced key metrics
        st.markdown("### 📊 Key Market Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_market_cap = stablecoin_df['market_cap'].sum()
        avg_price = stablecoin_df['current_price'].mean()
        total_stablecoins = len(stablecoin_df)
        avg_24h_change = stablecoin_df['price_change_percentage_24h'].mean()
        healthy_count = sum(1 for _, row in stablecoin_df.iterrows() 
                          if abs(row['current_price'] - 1) < 0.01)
        
        with col1:
            st.metric(
                "🏦 Total Market Cap",
                f"${total_market_cap/1e9:.1f}B",
                delta=f"{avg_24h_change:.2f}%" if pd.notna(avg_24h_change) else None
            )
        
        with col2:
            deviation_from_peg = abs(avg_price - 1) * 100
            st.metric(
                "⚖️ Avg Price",
                f"${avg_price:.4f}",
                delta=f"{deviation_from_peg:.3f}% from peg"
            )
        
        with col3:
            st.metric(
                "🎯 Tracked Coins",
                total_stablecoins,
                delta="Major stablecoins"
            )
        
        with col4:
            health_ratio = healthy_count / total_stablecoins
            st.metric(
                "✅ Stable (±1%)",
                f"{healthy_count}/{total_stablecoins}",
                delta=f"{health_ratio:.1%} healthy"
            )
        
        with col5:
            if not dominance_df.empty:
                top_dominance = dominance_df.iloc[0]['Dominance (%)']
                leader = dominance_df.iloc[0]['Stablecoin']
                st.metric(
                    f"👑 Market Leader",
                    leader,
                    delta=f"{top_dominance:.1f}% dominance"
                )
        
        st.markdown("---")
        
        # Enhanced dominance visualization
        st.markdown("### 📊 Market Dominance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not dominance_df.empty:
                fig_bar = px.bar(
                    dominance_df.head(8),
                    x='Stablecoin',
                    y='Dominance (%)',
                    title="🏆 Market Dominance Rankings",
                    color='Dominance (%)',
                    color_continuous_scale=['#FF6B6B', '#FFB347', '#4ECDC4', '#00FF88'],
                    text='Market Cap Formatted'
                )
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_size=16,
                    showlegend=False,
                    xaxis_tickangle=-45,
                    height=500
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            if not dominance_df.empty:
                colors = ['#00FFF0', '#8A4AF3', '#9945FF', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFB347', '#98D8E8']
                fig_pie = px.pie(
                    dominance_df.head(8),
                    names='Stablecoin',
                    values='Dominance (%)',
                    title="🥧 Market Share Distribution",
                    hole=0.4,
                    color_discrete_sequence=colors
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Dominance: %{value:.2f}%<br>Market Cap: %{customdata}<extra></extra>',
                    customdata=dominance_df.head(8)['Market Cap Formatted']
                )
                fig_pie.update_layout(
                    font_color='white',
                    title_font_size=16,
                    height=500
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Health status dashboard
        st.markdown("### 🏥 Comprehensive Health Analysis")
        
        health_data = []
        for i, (_, row) in enumerate(stablecoin_df.iterrows(), 1):
            score = calculate_enhanced_health_score(
                row['current_price'],
                row.get('market_cap_change_percentage_24h'),
                row.get('total_volume')
            )
            status, css_class, emoji = get_enhanced_health_status(score)
            
            health_data.append({
                'Rank': i,
                'Symbol': row['symbol'].upper(),
                'Name': row['name'],
                'Price ($)': f"{row['current_price']:.4f}",
                'Peg Deviation (%)': f"{(row['current_price']-1)*100:+.3f}",
                'Market Cap ($B)': f"{row['market_cap']/1e9:.2f}",
                '24h Change (%)': f"{row.get('market_cap_change_percentage_24h', 0):+.2f}",
                'Volume ($M)': f"{row.get('total_volume', 0)/1e6:.1f}",
                'Health Score': score,
                'Status': f"{emoji} {status}",
                'Last Updated': pd.to_datetime(row['last_updated']).strftime('%H:%M UTC')
            })
        
        health_df = pd.DataFrame(health_data)
        health_df = health_df.sort_values('Health Score', ascending=False)
        
        # Reset ranking after sorting
        health_df['Rank'] = range(1, len(health_df) + 1)
        
        st.dataframe(
            health_df,
            use_container_width=True,
            column_config={
                "Health Score": st.column_config.ProgressColumn(
                    "Health Score",
                    help="Composite health score (0-100) based on peg stability, market cap changes, and volume",
                    min_value=0,
                    max_value=100,
                ),
                "Rank": st.column_config.NumberColumn("Rank", help="Health ranking"),
                "Price ($)": st.column_config.TextColumn("Price ($)"),
                "Market Cap ($B)": st.column_config.TextColumn("Market Cap ($B)"),
            },
            hide_index=True
        )

# === PEG STABILITY SECTION ===
elif section == "Peg Stability":
    st.markdown("## ⚖️ Peg Stability & Volatility Analysis")
    
    if historical_peg_data and stablecoin_df is not None and not stablecoin_df.empty:
        # Current peg deviation overview
        st.markdown("### 🎯 Real-Time Peg Status")
        
        deviation_data = []
        for _, coin in stablecoin_df.iterrows():
            deviation_pct = (coin['current_price'] - 1) * 100
            deviation_data.append({
                'symbol': coin['symbol'].upper(),
                'price': coin['current_price'],
                'deviation': deviation_pct,
                'status': 'Stable' if abs(deviation_pct) < 0.1 else 'Moderate' if abs(deviation_pct) < 0.5 else 'High Risk'
            })
        
        # Display top 6 in grid format
        cols = st.columns(3)
        for i, coin_data in enumerate(deviation_data[:6]):
            col_idx = i % 3
            with cols[col_idx]:
                deviation = coin_data['deviation']
                if abs(deviation) < 0.1:
                    color = "green"
                    status_emoji = "✅"
                elif abs(deviation) < 0.5:
                    color = "orange"
                    status_emoji = "⚠️"
                else:
                    color = "red"
                    status_emoji = "🚨"
                
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <h3 style="color: #00FFF0; margin: 0;">{coin_data['symbol']}</h3>
                    <h2 style="color: white; margin: 10px 0;">${coin_data['price']:.4f}</h2>
                    <p style="color: {color}; font-size: 1.2em; margin: 0;">
                        {status_emoji} {deviation:+.3f}%
                    </p>
                    <p style="color: #888; font-size: 0.9em; margin: 5px 0 0 0;">
                        {coin_data['status']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Historical peg deviation analysis
        st.markdown("### 📊 Historical Peg Deviation Trends (30 Days)")
        
        if historical_peg_data:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Price Deviation from $1.00", "Absolute Deviation (Log Scale)"),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            colors = ['#00FFF0', '#8A4AF3', '#9945FF', '#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for i, (symbol, df_hist) in enumerate(historical_peg_data.items()):
                if not df_hist.empty:
                    # Main deviation chart
                    fig.add_trace(
                        go.Scatter(
                            x=df_hist['date'],
                            y=df_hist['peg_deviation_pct'],
                            mode='lines',
                            name=symbol,
                            line=dict(color=colors[i % len(colors)], width=2.5),
                            hovertemplate=f'<b>{symbol}</b><br>Date: %{{x}}<br>Price: $%{{customdata:.4f}}<br>Deviation: %{{y:.3f}}%<extra></extra>',
                            customdata=df_hist['price']
                        ),
                        row=1, col=1
                    )
                    
                    # Absolute deviation (log scale)
                    fig.add_trace(
                        go.Scatter(
                            x=df_hist['date'],
                            y=df_hist['peg_deviation_pct'].abs(),
                            mode='lines',
                            name=f"{symbol} (abs)",
                            line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                            showlegend=False,
                            hovertemplate=f'<b>{symbol} Absolute</b><br>Date: %{{x}}<br>Abs Deviation: %{{y:.3f}}%<extra></extra>'
                        ),
                        row=2, col=1
                    )
            
            # Add reference lines
            fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.8, 
                         annotation_text="Perfect Peg ($1.00)", row=1)
            fig.add_hline(y=1, line_dash="dot", line_color="orange", opacity=0.7, row=1)
            fig.add_hline(y=-1, line_dash="dot", line_color="orange", opacity=0.7, row=1)
            fig.add_hline(y=0.1, line_dash="dot", line_color="green", opacity=0.7, row=2)
            fig.add_hline(y=0.5, line_dash="dot", line_color="orange", opacity=0.7, row=2)
            fig.add_hline(y=1, line_dash="dot", line_color="red", opacity=0.7, row=2)
            
            fig.update_layout(
                height=700,
                hovermode='x unified',
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
            
            fig.update_yaxes(title_text="Deviation (%)", row=1, col=1)
            fig.update_yaxes(title_text="Absolute Deviation (%)", type="log", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility analysis
            st.markdown("### 📈 Volatility & Stability Rankings")
            
            volatility_data = []
            for symbol, df_hist in historical_peg_data.items():
                if not df_hist.empty and len(df_hist) > 1:
                    volatility = df_hist['peg_deviation_pct'].std()
                    max_deviation = df_hist['peg_deviation_pct'].abs().max()
                    mean_abs_deviation = df_hist['peg_deviation_pct'].abs().mean()
                    
                    stability_score = max(0, 100 - (volatility * 10) - (mean_abs_deviation * 5))
                    
                    volatility_data.append({
                        'Stablecoin': symbol,
                        'Volatility (%)': volatility,
                        'Max Deviation (%)': max_deviation,
                        'Avg Abs Deviation (%)': mean_abs_deviation,
                        'Stability Score': stability_score,
                        'Stability Grade': 'A' if stability_score >= 90 else 'B' if stability_score >= 80 else 'C' if stability_score >= 70 else 'D'
                    })
            
            if volatility_data:
                vol_df = pd.DataFrame(volatility_data).sort_values('Stability Score', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_vol = px.bar(
                        vol_df,
                        x='Stablecoin',
                        y='Volatility (%)',
                        title="📊 30-Day Price Volatility Comparison",
                        color='Stability Score',
                        color_continuous_scale=['#FF4444', '#FFAA00', '#4ECDC4', '#00FF88'],
                        text='Stability Grade'
                    )
                    fig_vol.update_traces(textposition='outside')
                    fig_vol.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        height=400
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🏆 Stability Rankings")
                    display_vol_df = vol_df.copy()
                    display_vol_df['Rank'] = range(1, len(display_vol_df) + 1)
                    
                    st.dataframe(
                        display_vol_df[['Rank', 'Stablecoin', 'Stability Score', 'Stability Grade', 'Volatility (%)']],
                        use_container_width=True,
                        column_config={
                            "Stability Score": st.column_config.ProgressColumn(
                                "Stability Score",
                                help="Higher scores indicate better stability",
                                min_value=0,
                                max_value=100,
                            ),
                            "Volatility (%)": st.column_config.NumberColumn(
                                "Volatility (%)",
                                format="%.3f"
                            )
                        },
                        hide_index=True
                    )
        
        # Risk assessment section
        st.markdown("### ⚠️ Risk Assessment & Alerts")
        
        risk_alerts = []
        for _, coin in stablecoin_df.iterrows():
            deviation = abs(coin['current_price'] - 1) * 100
            if deviation > 1:
                risk_level = "🔴 High Risk" if deviation > 2 else "🟠 Medium Risk"
                risk_alerts.append(f"{risk_level}: {coin['symbol'].upper()} deviation at {deviation:.2f}%")
        
        if risk_alerts:
            for alert in risk_alerts:
                st.markdown(f'<div class="warning-box">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">✅ All monitored stablecoins are within acceptable deviation ranges</div>', 
                       unsafe_allow_html=True)

# === SUPPLY ANALYSIS SECTION ===
elif section == "Supply Analysis":
    st.markdown("## 📦 Supply Dynamics & Market Analysis")
    
    if not stablecoin_df.empty:
        # Enhanced supply metrics dashboard
        st.markdown("### 💰 Current Supply Intelligence")
        
        total_supply_value = stablecoin_df['market_cap'].sum()
        supply_changes = stablecoin_df['market_cap_change_percentage_24h'].dropna()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🏦 Total Supply Value",
                f"${total_supply_value/1e9:.1f}B",
                delta=f"{supply_changes.mean():.2f}% (24h avg)" if not supply_changes.empty else None
            )
        
        with col2:
            if not dominance_df.empty:
                concentration = dominance_df.head(3)['Dominance (%)'].sum()
                st.metric(
                    "📊 Top 3 Concentration",
                    f"{concentration:.1f}%",
                    delta="Market control"
                )
        
        with col3:
            expanding_count = sum(1 for change in supply_changes if change > 2)
            contracting_count = sum(1 for change in supply_changes if change < -2)
            st.metric(
                "📈 Supply Expanding",
                f"{expanding_count}/{len(supply_changes)}",
                delta=f"{contracting_count} contracting"
            )
        
        with col4:
            if not supply_changes.empty:
                volatility = supply_changes.std()
                st.metric(
                    "🌊 Supply Volatility",
                    f"{volatility:.2f}%",
                    delta="24h std dev"
                )
        
        st.markdown("---")
        
        # Enhanced supply visualization
        st.markdown("### 📊 Supply Distribution & Changes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_supplies = stablecoin_df.nlargest(8, 'market_cap')
            
            fig_supply = px.bar(
                top_supplies,
                x='symbol',
                y='market_cap',
                title="🏦 Current Market Cap Distribution",
                labels={'market_cap': 'Market Cap (USD)', 'symbol': 'Stablecoin'},
                color='market_cap',
                color_continuous_scale=['#8A4AF3', '#00FFF0', '#9945FF'],
                text=top_supplies['market_cap'].apply(lambda x: f'${x/1e9:.1f}B')
            )
            fig_supply.update_traces(textposition='outside')
            fig_supply.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_supply, use_container_width=True)
        
        with col2:
            change_data = stablecoin_df.dropna(subset=['market_cap_change_percentage_24h'])
            
            if not change_data.empty:
                fig_change = px.bar(
                    change_data.head(8),
                    x='symbol',
                    y='market_cap_change_percentage_24h',
                    title="📈 24h Supply Changes",
                    labels={'market_cap_change_percentage_24h': '24h Change (%)', 'symbol': 'Stablecoin'},
                    color='market_cap_change_percentage_24h',
                    color_continuous_scale=['#FF4444', '#FFAA00', '#FFFFFF', '#4ECDC4', '#00FF88'],
                    text=change_data.head(8)['market_cap_change_percentage_24h'].apply(lambda x: f'{x:+.2f}%')
                )
                fig_change.update_traces(textposition='outside')
                fig_change.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_change.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_change, use_container_width=True)
        
        # Comprehensive supply table
        st.markdown("### 📋 Detailed Supply Analysis")
        
        supply_analysis_data = []
        for i, (_, row) in enumerate(stablecoin_df.iterrows(), 1):
            mcap_change = row.get('market_cap_change_percentage_24h', 0)
            trend = "📈 Growing" if mcap_change > 1 else "📉 Shrinking" if mcap_change < -1 else "➡️ Stable"
            
            supply_analysis_data.append({
                'Rank': i,
                'Stablecoin': row['symbol'].upper(),
                'Full Name': row['name'],
                'Market Cap': f"${row['market_cap']/1e9:.2f}B",
                'Market Share': f"{(row['market_cap']/total_supply_value)*100:.1f}%",
                'Global Rank': int(row['market_cap_rank']) if pd.notna(row['market_cap_rank']) else 'N/A',
                '24h Change': f"{mcap_change:+.2f}%",
                'Trend': trend,
                'Price': f"${row['current_price']:.4f}",
                'Volume': f"${row.get('total_volume', 0)/1e6:.1f}M",
                'Last Update': pd.to_datetime(row['last_updated']).strftime('%H:%M UTC')
            })
        
        supply_analysis_df = pd.DataFrame(supply_analysis_data)
        
        st.dataframe(
            supply_analysis_df,
            use_container_width=True,
            column_config={
                "Market Cap": st.column_config.TextColumn("Market Cap"),
                "Market Share": st.column_config.TextColumn("Market Share"),
                "24h Change": st.column_config.TextColumn("24h Change"),
                "Volume": st.column_config.TextColumn("Volume (24h)"),
            },
            hide_index=True
        )

# === ON-CHAIN ACTIVITY SECTION ===
elif section == "On-Chain Activity":
    st.markdown("## ⛓️ On-Chain Supply Dynamics")
    
    if not dune_supply_data.empty:
        st.markdown("### 📊 Historical Supply Trends on Ethereum")
        
        # Process the supply data for better visualization
        supply_columns = [col for col in dune_supply_data.columns if 'supply' in col.lower()]
        
        if supply_columns:
            # Create time series chart for supply evolution
            if 'week' in dune_supply_data.columns:
                recent_data = dune_supply_data.head(12)  # Last 12 weeks
                
                fig_supply_evolution = go.Figure()
                
                colors = ['#00FFF0', '#8A4AF3', '#9945FF', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFB347', '#98D8E8', '#F06292', '#AED581']
                
                for i, col in enumerate(supply_columns[:10]):  # Limit to top 10
                    if col in recent_data.columns:
                        stablecoin_name = col.replace(' supply', '').upper()
                        fig_supply_evolution.add_trace(
                            go.Scatter(
                                x=recent_data['week'],
                                y=recent_data[col],
                                mode='lines+markers',
                                name=stablecoin_name,
                                line=dict(color=colors[i % len(colors)], width=3),
                                hovertemplate=f'<b>{stablecoin_name}</b><br>Date: %{{x}}<br>Supply: $%{{y:,.0f}}<extra></extra>'
                            )
                        )
                
                fig_supply_evolution.update_layout(
                    title="📈 12-Week Supply Evolution Trends",
                    xaxis_title="Date",
                    yaxis_title="Supply (USD)",
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=600,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                st.plotly_chart(fig_supply_evolution, use_container_width=True)
            
            # Current week supply comparison
            st.markdown("### 💰 Current Week Supply Breakdown")
            
            if len(dune_supply_data) > 0:
                current_week = dune_supply_data.iloc[0]
                
                # Extract supply data for current week
                current_supplies = []
                for col in supply_columns:
                    if col in current_week and pd.notna(current_week[col]):
                        coin_name = col.replace(' supply', '').upper()
                        supply_value = current_week[col]
                        current_supplies.append({
                            'Stablecoin': coin_name,
                            'Supply': supply_value,
                            'Supply_Formatted': f"${supply_value/1e9:.2f}B" if supply_value > 0 else "$0"
                        })
                
                if current_supplies:
                    current_supply_df = pd.DataFrame(current_supplies)
                    current_supply_df = current_supply_df.sort_values('Supply', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_current = px.bar(
                            current_supply_df.head(8),
                            x='Stablecoin',
                            y='Supply',
                            title="🏦 Current Supply Distribution",
                            color='Supply',
                            color_continuous_scale=['#8A4AF3', '#00FFF0', '#9945FF'],
                            text='Supply_Formatted'
                        )
                        fig_current.update_traces(textposition='outside')
                        fig_current.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white',
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_current, use_container_width=True)
                    
                    with col2:
                        fig_pie_supply = px.pie(
                            current_supply_df.head(6),
                            names='Stablecoin',
                            values='Supply',
                            title="🥧 Supply Market Share",
                            hole=0.4,
                            color_discrete_sequence=['#00FFF0', '#8A4AF3', '#9945FF', '#FF6B6B', '#4ECDC4', '#45B7D1']
                        )
                        fig_pie_supply.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate='<b>%{label}</b><br>Supply: $%{value:,.0f}<br>Share: %{percent}<extra></extra>'
                        )
                        fig_pie_supply.update_layout(
                            font_color='white',
                            height=400
                        )
                        st.plotly_chart(fig_pie_supply, use_container_width=True)
            
            # Supply change analysis
            if len(dune_supply_data) > 1:
                st.markdown("### 📊 Week-over-Week Supply Changes")
                
                current_week = dune_supply_data.iloc[0]
                previous_week = dune_supply_data.iloc[1]
                
                supply_changes = []
                for col in supply_columns:
                    if col in current_week and col in previous_week:
                        coin_name = col.replace(' supply', '').upper()
                        current_val = current_week[col] if pd.notna(current_week[col]) else 0
                        previous_val = previous_week[col] if pd.notna(previous_week[col]) else 0
                        
                        if previous_val > 0:
                            change_pct = ((current_val - previous_val) / previous_val) * 100
                            change_abs = current_val - previous_val
                            
                            supply_changes.append({
                                'Stablecoin': coin_name,
                                'Previous Week': previous_val,
                                'Current Week': current_val,
                                'Change ($)': change_abs,
                                'Change (%)': change_pct,
                                'Trend': '📈 Mint' if change_pct > 0.1 else '📉 Burn' if change_pct < -0.1 else '➡️ Stable'
                            })
                
                if supply_changes:
                    changes_df = pd.DataFrame(supply_changes)
                    changes_df = changes_df.sort_values('Change (%)', ascending=False)
                    
                    # Format the display
                    changes_display = changes_df.copy()
                    changes_display['Previous Week'] = changes_display['Previous Week'].apply(lambda x: f"${x/1e9:.2f}B")
                    changes_display['Current Week'] = changes_display['Current Week'].apply(lambda x: f"${x/1e9:.2f}B")
                    changes_display['Change ($)'] = changes_display['Change ($)'].apply(lambda x: f"${x/1e9:+.2f}B")
                    changes_display['Change (%)'] = changes_display['Change (%)'].apply(lambda x: f"{x:+.2f}%")
                    changes_display['Rank'] = range(1, len(changes_display) + 1)
                    
                    st.dataframe(
                        changes_display[['Rank', 'Stablecoin', 'Previous Week', 'Current Week', 'Change ($)', 'Change (%)', 'Trend']],
                        use_container_width=True,
                        column_config={
                            "Change ($)": st.column_config.TextColumn("Weekly Change ($)"),
                            "Change (%)": st.column_config.TextColumn("Weekly Change (%)"),
                        },
                        hide_index=True
                    )
        else:
            st.info("Supply column data not available in the current dataset")
    else:
        st.warning("On-chain supply data is not available. Please check your Dune Analytics API configuration.")
        
        # Show instructions for Dune setup
        st.markdown("""
        ### 🔧 Setup Instructions for Dune Analytics
        
        To enable on-chain supply data:
        
        1. **Get Dune API Key:** Sign up at [dune.com](https://dune.com) and get your API key
        2. **Add to Environment:** Add `DEFI_JOSH_DUNE_QUERY_API_KEY=your_key_here` to your `.env` file
        3. **Install Dune Client:** Run `pip install dune-client`
        4. **Restart App:** Restart the Streamlit app to load the new configuration
        
        The dashboard will automatically fetch on-chain mint/burn data once configured.
        """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>📊 <strong>Stablecoin Health Monitor</strong> | Real-time data from CoinGecko Pro API & Dune Analytics</p>
    <p>⚡ Cache: 24 hours | 🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    <p style="font-size: 0.9em;">💡 Remember: Past performance doesn't guarantee future stability</p>
    <p style="font-size: 0.8em;">🛠️ Built with Streamlit • Plotly • CoinGecko Pro API • Dune Analytics</p>
</div>
""", unsafe_allow_html=True)