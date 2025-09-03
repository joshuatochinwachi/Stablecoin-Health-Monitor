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

# -------------------------
# Stablecoin Health Monitor
# Extended: On-chain Mint/Burn, Liquidity Depth (Uniswap/Curve), Whale Flows
# Maintains original UI styling and sections
# -------------------------

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Stablecoin Health Monitor", 
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (kept from original)
st.markdown("""
<style>
    .stRadio > div {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .stRadio > div > label {
        margin: 0 15px;
        font-size: 16px;
        color: #00FFF0;
        background-color: #8A4AF3;
        padding: 8px 16px;
        border-radius: 5px;
        transition: all 0.3s;
        cursor: pointer;
    }
    .stRadio > div > label:hover {
        background-color: #9945FF;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #8A4AF3;
        margin: 10px 0;
    }
    .health-score-good {
        color: #00FF88;
        font-weight: bold;
    }
    .health-score-warning {
        color: #FFAA00;
        font-weight: bold;
    }
    .health-score-critical {
        color: #FF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# API Configuration
# -------------------------
@st.cache_data(ttl=300)
def get_api_keys():
    return {
        'coingecko': os.getenv("COINGECKO_PRO_API_KEY"),
        'dune': os.getenv("DEFI_JOSH_DUNE_QUERY_API_KEY"),
        'etherscan': os.getenv("ETHERSCAN_API_KEY"),
        'thegraph': os.getenv("THEGRAPH_API_KEY", "")
    }

# Known mainnet contract addresses (ethereum) for major stables - used for onchain lookups
STABLECOIN_CONTRACTS = {
    'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
    'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
    'DAI':  '0x6b175474e89094c44da98b954eedeac495271d0f',
    'BUSD': '0x4fabb145d64652a948d72533023f6e7a623c7c53'
}

ETHERSCAN_API_URL = 'https://api.etherscan.io/api'
UNISWAP_V2_SUBGRAPH = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2'
CURVE_SUBGRAPH = 'https://api.thegraph.com/subgraphs/name/curvefi/curve'

# -------------------------
# CoinGecko: market data (unchanged logic)
# -------------------------
@st.cache_data(ttl=300)
def fetch_stablecoin_data():
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
    headers = {"x-cg-pro-api-key": api_keys['coingecko']}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        major_stables = ['usdt', 'usdc', 'dai', 'busd', 'tusd', 'frax', 'lusd', 'susd', 'pyusd', 'rlusd', 'usds']
        filtered_df = df[df['symbol'].isin(major_stables)].copy()
        return filtered_df
    except Exception as e:
        st.error(f"Error fetching CoinGecko data: {e}")
        try:
            return joblib.load("data/stablecoins_filtered.joblib")
        except:
            return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_dominance_data():
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
            response = requests.get(global_url, headers=headers, timeout=10)
            global_data = response.json()
            total_market_cap = global_data['data']['total_market_cap']['usd']
        except:
            total_market_cap = 2.5e12
    else:
        total_market_cap = 2.5e12

    dominance_data = []
    for _, row in df.iterrows():
        if pd.notna(row['market_cap']) and row['market_cap'] > 0:
            dominance = (row['market_cap'] / total_market_cap) * 100
            dominance_data.append({
                'Stablecoin': row['symbol'].upper(),
                'Dominance (%)': dominance,
                'Market Cap': row['market_cap']
            })

    dominance_df = pd.DataFrame(dominance_data)
    return dominance_df.sort_values('Dominance (%)', ascending=False)

@st.cache_data(ttl=300)
def fetch_historical_peg_data():
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

    return historical_data

# -------------------------
# On-chain: Etherscan token transfers
# -------------------------
@st.cache_data(ttl=120)
def fetch_token_transfers(contract_address, startblock=0, endblock=99999999, page=1, offset=10000, sort='desc'):
    api_keys = get_api_keys()
    key = api_keys.get('etherscan')
    if not key:
        st.warning("ETHERSCAN_API_KEY not set — on-chain features will be limited.")
        return []

    params = {
        'module': 'account',
        'action': 'tokentx',
        'contractaddress': contract_address,
        'startblock': startblock,
        'endblock': endblock,
        'page': page,
        'offset': offset,
        'sort': sort,
        'apikey': key
    }

    try:
        r = requests.get(ETHERSCAN_API_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get('status') == '1' and data.get('result'):
            return data['result']
        else:
            return []
    except Exception as e:
        st.error(f"Etherscan request failed: {e}")
        return []

def summarize_mint_burn(transfers_df, lookback_days=7):
    if len(transfers_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    zero = '0x0000000000000000000000000000000000000000'
    df = transfers_df.copy()
    df = df.assign(value_float = lambda d: d['value'].astype(float) / (10 ** d['tokenDecimal'].astype(int)),
                   timeStamp = pd.to_datetime(d['timeStamp'].astype(int), unit='s'))
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    df = df[df['timeStamp'] >= cutoff]

    mints = df[df['from'].str.lower() == zero]
    burns = df[df['to'].str.lower() == zero]

    mint_summary = mints.groupby('contractAddress').agg({
        'value_float': 'sum',
        'hash': 'count'
    }).rename(columns={'value_float': 'total_minted', 'hash': 'mint_txs'}).reset_index()

    burn_summary = burns.groupby('contractAddress').agg({
        'value_float': 'sum',
        'hash': 'count'
    }).rename(columns={'value_float': 'total_burned', 'hash': 'burn_txs'}).reset_index()

    return mint_summary, burn_summary

# -------------------------
# Whale flows: large transfers detection
# -------------------------
def get_large_transfers(transfers_df, threshold_usd=1_000_000, price_lookup=1.0):
    if len(transfers_df) == 0:
        return pd.DataFrame()
    df = transfers_df.copy()
    df['value_float'] = df['value'].astype(float) / (10 ** df['tokenDecimal'].astype(int))
    df['usd_value'] = df['value_float'] * price_lookup
    df['timeStamp'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s')
    large = df[df['usd_value'] >= threshold_usd]
    return large.sort_values('usd_value', ascending=False)

# -------------------------
# Liquidity depth via The Graph (Uniswap V2 and Curve)
# -------------------------
def query_the_graph(subgraph_url, query, variables=None):
    try:
        r = requests.post(subgraph_url, json={'query': query, 'variables': variables}, timeout=15)
        r.raise_for_status()
        res = r.json()
        return res.get('data')
    except Exception as e:
        st.warning(f"The Graph query failed: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_uniswap_pair_by_token(token_address):
    token = token_address.lower()
    query = '''
    query($token: Bytes!) {
      pairs0: pairs(where: {token0: $token}, first: 10, orderBy: reserveUSD, orderDirection: desc) {
        id reserve0 reserve1 reserveUSD token0 {id symbol decimals} token1 {id symbol decimals}
      }
      pairs1: pairs(where: {token1: $token}, first: 10, orderBy: reserveUSD, orderDirection: desc) {
        id reserve0 reserve1 reserveUSD token0 {id symbol decimals} token1 {id symbol decimals}
      }
    }
    '''
    data = query_the_graph(UNISWAP_V2_SUBGRAPH, query, {'token': token})
    if not data:
        return []
    pairs = (data.get('pairs0', []) or []) + (data.get('pairs1', []) or [])
    out = []
    for p in pairs:
        out.append({
            'pair_id': p['id'],
            'token0': p['token0']['symbol'],
            'token1': p['token1']['symbol'],
            'reserveUSD': float(p['reserveUSD']) if p['reserveUSD'] else 0.0
        })
    return sorted(out, key=lambda x: x['reserveUSD'], reverse=True)

@st.cache_data(ttl=300)
def fetch_curve_pool_liquidity(token_address):
    token = token_address.lower()
    query = '''
    query($token: String!) {
      pools(first: 10, where: {coins_contains: [$token]}) {
        id name coins
        virtual_price
        totalSupply
      }
    }
    '''
    data = query_the_graph(CURVE_SUBGRAPH, query, {'token': token})
    if not data:
        return []
    pools = data.get('pools', [])
    out = []
    for p in pools:
        out.append({'pool_id': p.get('id'), 'name': p.get('name'), 'coins': p.get('coins'), 'virtual_price': p.get('virtual_price')})
    return out

# -------------------------
# Health scoring (unchanged)
# -------------------------
def calculate_health_score(price, market_cap_change_24h, volatility_30d=None):
    score = 100
    peg_deviation = abs(price - 1.0)
    if peg_deviation > 0.1:
        score -= 40
    elif peg_deviation > 0.05:
        score -= 25
    elif peg_deviation > 0.01:
        score -= 10
    elif peg_deviation > 0.005:
        score -= 5

    if pd.notna(market_cap_change_24h):
        abs_change = abs(market_cap_change_24h)
        if abs_change > 10:
            score -= 30
        elif abs_change > 5:
            score -= 20
        elif abs_change > 2:
            score -= 10

    if volatility_30d and volatility_30d > 5:
        score -= 30
    elif volatility_30d and volatility_30d > 2:
        score -= 15

    return max(0, min(100, score))


def get_health_status(score):
    if score >= 80:
        return "Healthy", "health-score-good"
    elif score >= 60:
        return "Warning", "health-score-warning"
    else:
        return "Critical", "health-score-critical"

# -------------------------
# Sidebar (kept content)
# -------------------------
with st.sidebar:
    st.markdown("""
    <div style="background-color: #8A4AF3; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #00FFF0; text-align: center; margin: 0;">📊 Stablecoin Health Monitor</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Are stablecoins actually stable? 🤔

    Stablecoins were designed to solve crypto's volatility problem by pegging to the dollar at $1.00. 
    But **stability is never guaranteed.**

    This dashboard monitors:
    - **Peg Deviation** - How far from $1.00?
    - **Supply Changes** - Mints, burns, and trends
    - **Market Dominance** - Who controls the market?
    - **Health Scores** - Composite stability metrics

    ---

    **Major Stablecoins Tracked:**
    - USDT, USDC (Market Leaders)
    - DAI, FRAX (DeFi Natives) 
    - BUSD, TUSD (Traditional)
    - PYUSD, RLUSD, USDS (New Players)
    - LUSD, sUSD (Specialized)

    Each reacts differently to market stress. Some use cash reserves (USDC), 
    others algorithmic mechanisms (DAI), testing new approaches and risks.

    **Data Sources:**
    - 🦎 CoinGecko Pro API
    - ⚡ Dune Analytics
    - 📡 Real-time updates every 5 minutes
    """, unsafe_allow_html=True)

# -------------------------
# Main header
# -------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #00FFF0 0%, #8A4AF3 50%, #9945FF 100%); 
           padding: 20px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
    <h1 style="color: white; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        💰 Stablecoin Health Monitor
    </h1>
    <p style="color: #E0E0E0; margin: 10px 0 0 0; font-size: 1.2em;">
        Real-time stability analysis and market intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation
section = st.radio(
    "",
    ["Market Overview", "Peg Stability", "Supply Analysis", "On-chain Flows"],
    horizontal=True,
    key="nav_radio"
)

# Load data
with st.spinner("Loading latest stablecoin data..."):
    stablecoin_df = fetch_stablecoin_data()
    dominance_df = fetch_dominance_data()
    historical_peg_data = fetch_historical_peg_data()

# === MARKET OVERVIEW ===
if section == "Market Overview":
    st.markdown("## 📈 Market Overview")
    if not stablecoin_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        total_market_cap = stablecoin_df['market_cap'].sum()
        avg_price = stablecoin_df['current_price'].mean()
        total_stablecoins = len(stablecoin_df)
        avg_24h_change = stablecoin_df['price_change_percentage_24h'].mean()

        with col1:
            st.metric("Total Market Cap", f"${total_market_cap/1e9:.1f}B", delta=f"{avg_24h_change:.2f}%" if pd.notna(avg_24h_change) else None)
        with col2:
            st.metric("Average Price", f"${avg_price:.4f}", delta=f"{abs(avg_price-1)*100:.3f}% from peg")
        with col3:
            st.metric("Tracked Stablecoins", total_stablecoins)
        with col4:
            healthy_count = sum(1 for _, row in stablecoin_df.iterrows() if abs(row['current_price'] - 1) < 0.01)
            st.metric("Healthy (±1%)", f"{healthy_count}/{total_stablecoins}")

        st.markdown('---')
        col1, col2 = st.columns(2)
        with col1:
            if not dominance_df.empty:
                fig_bar = px.bar(dominance_df.head(8), x='Stablecoin', y='Dominance (%)', title="Market Dominance by Stablecoin", color='Dominance (%)')
                fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            if not dominance_df.empty:
                fig_pie = px.pie(dominance_df.head(6), names='Stablecoin', values='Dominance (%)', title="Market Share Distribution", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('### 🏥 Health Status Dashboard')
        health_data = []
        for _, row in stablecoin_df.iterrows():
            score = calculate_health_score(row['current_price'], row.get('market_cap_change_percentage_24h'))
            status, css_class = get_health_status(score)
            health_data.append({
                'Symbol': row['symbol'].upper(),
                'Name': row['name'],
                'Price': f"${row['current_price']:.4f}",
                'Peg Deviation': f"{(row['current_price']-1)*100:+.3f}%",
                'Market Cap': f"${row['market_cap']/1e9:.2f}B",
                '24h Change': f"{row.get('market_cap_change_percentage_24h', 0):+.2f}%",
                'Health Score': score,
                'Status': status
            })
        health_df = pd.DataFrame(health_data).sort_values('Health Score', ascending=False)
        st.dataframe(health_df, use_container_width=True, column_config={
            "Health Score": st.column_config.ProgressColumn("Health Score", help="Composite health score (0-100)", min_value=0, max_value=100),
            "Price": st.column_config.TextColumn("Price ($)"),
            "Market Cap": st.column_config.TextColumn("Market Cap")
        })
    else:
        st.error("Unable to load CoinGecko data. Set COINGECKO_PRO_API_KEY or provide cached joblib files.")

# === PEG STABILITY ===
elif section == "Peg Stability":
    st.markdown("## ⚖️ Peg Stability Analysis")
    if historical_peg_data and stablecoin_df is not None and not stablecoin_df.empty:
        st.markdown("### 🎯 Current Peg Deviations")
        peg_cols = st.columns(len(stablecoin_df.head(4)))
        for i, (_, coin) in enumerate(stablecoin_df.head(4).iterrows()):
            if i < len(peg_cols):
                deviation = (coin['current_price'] - 1) * 100
                with peg_cols[i]:
                    st.metric(coin['symbol'].upper(), f"${coin['current_price']:.4f}", delta=f"{deviation:+.3f}%", delta_color="inverse")
        st.markdown('---')
        st.markdown('### 📊 Historical Peg Deviation Trends')
        fig = go.Figure()
        colors = ['#00FFF0', '#8A4AF3', '#9945FF', '#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, (symbol, df_hist) in enumerate(historical_peg_data.items()):
            if not df_hist.empty:
                fig.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['peg_deviation_pct'], mode='lines', name=symbol, line=dict(color=colors[i % len(colors)], width=2)))
        fig.add_hline(y=0, line_dash='dash', line_color='gray', annotation_text='Perfect Peg ($1.00)')
        fig.add_hline(y=1, line_dash='dot', line_color='orange', opacity=0.7)
        fig.add_hline(y=-1, line_dash='dot', line_color='orange', opacity=0.7)
        fig.update_layout(title='Peg Deviation Over Time (Last 30 Days)', xaxis_title='Date', yaxis_title='Deviation from $1.00 (%)', hovermode='x unified', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=500)
        st.plotly_chart(fig, use_container_width=True)

        volatility_data = []
        for symbol, df_hist in historical_peg_data.items():
            if not df_hist.empty and len(df_hist) > 1:
                volatility = df_hist['peg_deviation_pct'].std()
                max_deviation = df_hist['peg_deviation_pct'].abs().max()
                volatility_data.append({'Stablecoin': symbol, 'Volatility (%)': volatility, 'Max Deviation (%)': max_deviation, 'Stability Rank': 'High' if volatility < 0.1 else 'Medium' if volatility < 0.5 else 'Low'})
        if volatility_data:
            vol_df = pd.DataFrame(volatility_data).sort_values('Volatility (%)')
            col1, col2 = st.columns(2)
            with col1:
                fig_vol = px.bar(vol_df, x='Stablecoin', y='Volatility (%)', title='Price Volatility by Stablecoin', color='Volatility (%)')
                fig_vol.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_vol, use_container_width=True)
            with col2:
                st.dataframe(vol_df, use_container_width=True, column_config={"Volatility (%)": st.column_config.NumberColumn("Volatility (%)", format="%.3f%%"), "Max Deviation (%)": st.column_config.NumberColumn("Max Deviation (%)", format="%.3f%%")})

# === SUPPLY ANALYSIS ===
elif section == "Supply Analysis":
    st.markdown("## 📦 Supply Analysis")
    if not stablecoin_df.empty:
        total_supply_value = stablecoin_df['market_cap'].sum()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stablecoin Supply", f"${total_supply_value/1e9:.1f}B")
        with col2:
            supply_concentration = dominance_df.head(3)['Dominance (%)'].sum() if not dominance_df.empty else 0
            st.metric("Top 3 Concentration", f"{supply_concentration:.1f}%")
        with col3:
            avg_24h_supply_change = stablecoin_df['market_cap_change_percentage_24h'].mean()
            if pd.notna(avg_24h_supply_change):
                st.metric("Avg 24h Supply Change", f"{avg_24h_supply_change:+.2f}%")
        st.markdown('---')
        col1, col2 = st.columns(2)
        with col1:
            top_supplies = stablecoin_df.nlargest(8, 'market_cap')
            fig_supply = px.bar(top_supplies, x='symbol', y='market_cap', title='Current Supply by Market Cap', labels={'market_cap': 'Market Cap (USD)', 'symbol': 'Stablecoin'})
            fig_supply.update_traces(marker_color='#00FFF0')
            fig_supply.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_supply, use_container_width=True)
        with col2:
            change_data = stablecoin_df.dropna(subset=['market_cap_change_percentage_24h'])
            if not change_data.empty:
                fig_change = px.bar(change_data, x='symbol', y='market_cap_change_percentage_24h', title='24h Supply Changes', labels={'market_cap_change_percentage_24h': '24h Change (%)', 'symbol': 'Stablecoin'}, color='market_cap_change_percentage_24h')
                fig_change.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_change, use_container_width=True)
        st.markdown('### 📋 Detailed Supply Information')
        supply_table_data = []
        for _, row in stablecoin_df.iterrows():
            supply_table_data.append({'Stablecoin': row['symbol'].upper(), 'Name': row['name'], 'Market Cap': f"${row['market_cap']/1e9:.2f}B", 'Market Cap Rank': int(row['market_cap_rank']) if pd.notna(row['market_cap_rank']) else 'N/A', '24h Change': f"{row.get('market_cap_change_percentage_24h', 0):+.2f}%", 'Price': f"${row['current_price']:.4f}", 'Last Updated': pd.to_datetime(row['last_updated']).strftime('%Y-%m-%d %H:%M UTC')})
        supply_df_display = pd.DataFrame(supply_table_data).sort_values('Market Cap Rank')
        st.dataframe(supply_df_display, use_container_width=True)
    else:
        st.error('Unable to load stablecoin data. Please check your API configuration.')

# === ON-CHAIN FLOWS ===
elif section == 'On-chain Flows':
    st.markdown('## 🔗 On-chain Mint/Burn, Liquidity & Whale Flows')
    api_keys = get_api_keys()
    st.info('On-chain features require ETHERSCAN_API_KEY (Etherscan) and The Graph public subgraphs for liquidity checks.')

    col1, col2 = st.columns([2,1])
    with col2:
        token_choice = st.selectbox('Select token', list(STABLECOIN_CONTRACTS.keys()))
        contract = STABLECOIN_CONTRACTS[token_choice]
        lookback = st.slider('Lookback days', 1, 30, 7)
        large_threshold = st.number_input('Large transfer threshold (USD)', value=1_000_000, step=100_000)

    with col1:
        st.markdown(f'### {token_choice} ({contract}) — On-chain summary')
        with st.spinner('Fetching token transfers...'):
            transfers = fetch_token_transfers(contract, page=1, offset=10000)
            transfers_df = pd.DataFrame(transfers)
        if transfers_df.empty:
            st.warning('No transfer data available (empty response or API key missing)')
        else:
            transfers_df['timeStamp'] = pd.to_datetime(transfers_df['timeStamp'].astype(int), unit='s')
            st.write(f"Total transfers fetched: {len(transfers_df)} — showing last {lookback} days")
            mint_summary, burn_summary = summarize_mint_burn(transfers_df, lookback_days=lookback)
            st.markdown('#### Mint / Burn Summary')
            c1, c2 = st.columns(2)
            with c1:
                if not mint_summary.empty:
                    st.dataframe(mint_summary)
                else:
                    st.write('No mints in lookback window')
            with c2:
                if not burn_summary.empty:
                    st.dataframe(burn_summary)
                else:
                    st.write('No burns in lookback window')

            # Whale detection
            token_price = 1.0
            large_df = get_large_transfers(transfers_df, threshold_usd=large_threshold, price_lookup=token_price)
            st.markdown('#### Large Transfers (Whale movements)')
            if not large_df.empty:
                st.dataframe(large_df[['timeStamp','from','to','value_float','usd_value','hash']].rename(columns={'value_float':'amount'}))
            else:
                st.write('No transfers exceeded threshold in lookback window')

    st.markdown('---')
    st.markdown('## 💧 Liquidity Depth (Uniswap V2 & Curve)')
    with st.spinner('Querying Uniswap + Curve...'):
        pairs = fetch_uniswap_pair_by_token(contract)
        curve_pools = fetch_curve_pool_liquidity(contract)
    if pairs:
        st.markdown('### Top Uniswap V2 pairs by reserve (USD)')
        st.dataframe(pd.DataFrame(pairs))
    else:
        st.write('No Uniswap V2 pairs found or subgraph query failed')
    if curve_pools:
        st.markdown('### Curve pools containing token')
        st.dataframe(pd.DataFrame(curve_pools))
    else:
        st.write('No Curve pools found or subgraph query failed')

# Footer
st.markdown('---')
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>📊 <strong>Stablecoin Health Monitor</strong> | Real-time data from CoinGecko Pro API</p>
    <p>⚡ Updates every 5 minutes | 🔄 Last updated: {}</p>
    <p style="font-size: 0.9em;">💡 Remember: Past performance doesn't guarantee future stability</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')), unsafe_allow_html=True)
