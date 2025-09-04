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
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Dune client for on-chain data
try:
    from dune_client.client import DuneClient
    DUNE_AVAILABLE = True
except ImportError:
    DUNE_AVAILABLE = False
    st.warning("⚠️ Dune client not installed. On-chain data features will be limited.")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🚀 Stablecoin Health Monitor Pro", 
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for professional styling
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

# Configuration Management
class ConfigManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.cache_duration = 86400  # 24 hours in seconds
        self.major_stables = [
            'usdt', 'usdc', 'dai', 'busd', 'tusd', 'frax', 
            'lusd', 'susd', 'pyusd', 'rlusd', 'usds'
        ]
        self.dune_query_id = 5681885
    
    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        return {
            'coingecko': os.getenv("COINGECKO_PRO_API_KEY"),
            'dune': os.getenv("DEFI_JOSH_DUNE_QUERY_API_KEY")
        }
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate configuration and return status with messages"""
        messages = []
        is_valid = True
        
        if not self.api_keys['coingecko']:
            messages.append("⚠️ CoinGecko Pro API key not found")
            is_valid = False
        
        if not self.api_keys['dune']:
            messages.append("⚠️ Dune API key not found")
            is_valid = False
        
        if not DUNE_AVAILABLE:
            messages.append("⚠️ Dune client library not installed")
        
        return is_valid, messages

# Enhanced Data Fetcher with better error handling and validation
class DataFetcher:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.session = None
    
    def _validate_response_data(self, data: any, source: str) -> bool:
        """Validate API response data structure"""
        try:
            if source == 'coingecko':
                return isinstance(data, list) and len(data) > 0 and 'symbol' in data[0]
            elif source == 'dune':
                return hasattr(data, 'result') and hasattr(data.result, 'rows')
            return False
        except Exception:
            return False
    
    @st.cache_data(ttl=86400)  # Cache for 24 hours
    def fetch_stablecoin_data(_self) -> pd.DataFrame:
        """Fetch current stablecoin data from CoinGecko with enhanced error handling"""
        if not _self.config.api_keys['coingecko']:
            return _self._load_fallback_data('stablecoins_filtered.joblib')
        
        url = "https://pro-api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "category": "stablecoins",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "1h,24h,7d,30d"
        }
        headers = {"x-cg-pro-api-key": _self.config.api_keys['coingecko']}
        
        try:
            with st.spinner("🦎 Fetching latest market data from CoinGecko..."):
                response = requests.get(url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                if not _self._validate_response_data(data, 'coingecko'):
                    raise ValueError("Invalid response structure from CoinGecko")
                
                df = pd.DataFrame(data)
                
                # Enhanced filtering and data cleaning
                filtered_df = df[df['symbol'].isin(_self.config.major_stables)].copy()
                
                # Data validation and cleaning
                filtered_df = filtered_df.dropna(subset=['current_price', 'market_cap'])
                filtered_df = filtered_df[filtered_df['market_cap'] > 0]
                
                # Save successful fetch
                try:
                    joblib.dump(filtered_df, "data/stablecoins_filtered.joblib")
                except:
                    pass  # Silent fail for cache save
                
                return filtered_df
                
        except requests.exceptions.Timeout:
            st.error("🕐 Request timed out. Using cached data.")
            return _self._load_fallback_data('stablecoins_filtered.joblib')
        except requests.exceptions.RequestException as e:
            st.error(f"🌐 Network error: {str(e)[:100]}...")
            return _self._load_fallback_data('stablecoins_filtered.joblib')
        except Exception as e:
            st.error(f"📊 Data processing error: {str(e)[:100]}...")
            return _self._load_fallback_data('stablecoins_filtered.joblib')
    
    @st.cache_data(ttl=86400)  # Cache for 24 hours
    def fetch_dune_onchain_data(_self) -> pd.DataFrame:
        """Fetch on-chain mint/burn data from Dune Analytics"""
        if not _self.config.api_keys['dune'] or not DUNE_AVAILABLE:
            return _self._load_fallback_data('dune_onchain_data.joblib')
        
        try:
            with st.spinner("⚡ Fetching on-chain data from Dune Analytics..."):
                dune = DuneClient(_self.config.api_keys['dune'])
                query_result = dune.get_latest_result(_self.config.dune_query_id)
                
                if not _self._validate_response_data(query_result, 'dune'):
                    raise ValueError("Invalid response from Dune Analytics")
                
                rows = query_result.result.rows
                df = pd.DataFrame(rows)
                
                if df.empty:
                    raise ValueError("Empty dataset from Dune")
                
                # Data processing and validation
                df['week'] = pd.to_datetime(df['week'])
                df = df.sort_values('week', ascending=False)
                
                # Validate required columns
                required_cols = ['week', 'USDC supply', 'DAI supply', 'LUSD supply']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"⚠️ Missing columns in Dune data: {missing_cols}")
                
                # Save successful fetch
                try:
                    joblib.dump(df, "data/dune_onchain_data.joblib")
                except:
                    pass
                
                return df
                
        except Exception as e:
            st.error(f"⚡ Dune Analytics error: {str(e)[:100]}...")
            return _self._load_fallback_data('dune_onchain_data.joblib')
    
    def _load_fallback_data(self, filename: str) -> pd.DataFrame:
        """Load fallback data with proper error handling"""
        try:
            return joblib.load(f"data/{filename}")
        except FileNotFoundError:
            st.warning(f"📁 Cache file '{filename}' not found. Creating empty dataset.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"💾 Cache loading error: {str(e)[:50]}...")
            return pd.DataFrame()
    
    @st.cache_data(ttl=86400)
    def fetch_dominance_data(_self) -> pd.DataFrame:
        """Calculate market dominance with enhanced accuracy"""
        df = _self.fetch_stablecoin_data()
        if df.empty:
            return _self._load_fallback_data('current_dominance_df.joblib')
        
        try:
            # Get total crypto market cap
            if _self.config.api_keys['coingecko']:
                global_url = "https://pro-api.coingecko.com/api/v3/global"
                headers = {"x-cg-pro-api-key": _self.config.api_keys['coingecko']}
                response = requests.get(global_url, headers=headers, timeout=10)
                global_data = response.json()
                total_market_cap = global_data['data']['total_market_cap']['usd']
            else:
                total_market_cap = 2.8e12  # Updated fallback estimate
            
            # Calculate dominance with validation
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
                        'Market Cap': row['market_cap'],
                        'Name': row['name']
                    })
            
            dominance_df = pd.DataFrame(dominance_data)
            return dominance_df.sort_values('Market Cap', ascending=False)
            
        except Exception as e:
            st.error(f"📊 Dominance calculation error: {str(e)[:50]}...")
            return pd.DataFrame()

# Enhanced Analytics Engine
class AnalyticsEngine:
    def __init__(self):
        self.volatility_thresholds = {
            'low': 0.1,
            'medium': 0.5,
            'high': 1.0
        }
    
    def calculate_advanced_health_score(self, row: pd.Series, historical_data: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate comprehensive health score with multiple factors"""
        base_score = 100
        factors = {}
        
        # 1. Peg Deviation Analysis (40% weight)
        peg_deviation = abs(row['current_price'] - 1.0)
        if peg_deviation > 0.1:  # >10%
            peg_penalty = 40
            peg_status = "Critical"
        elif peg_deviation > 0.05:  # >5%
            peg_penalty = 25
            peg_status = "Poor"
        elif peg_deviation > 0.01:  # >1%
            peg_penalty = 12
            peg_status = "Fair"
        elif peg_deviation > 0.005:  # >0.5%
            peg_penalty = 6
            peg_status = "Good"
        else:
            peg_penalty = 0
            peg_status = "Excellent"
        
        factors['peg_deviation'] = {
            'penalty': peg_penalty,
            'value': peg_deviation * 100,
            'status': peg_status
        }
        
        # 2. Market Cap Stability (25% weight)
        market_cap_change = row.get('market_cap_change_percentage_24h', 0)
        if pd.notna(market_cap_change):
            abs_change = abs(market_cap_change)
            if abs_change > 15:
                cap_penalty = 25
                cap_status = "Highly Volatile"
            elif abs_change > 10:
                cap_penalty = 18
                cap_status = "Volatile"
            elif abs_change > 5:
                cap_penalty = 12
                cap_status = "Moderate"
            elif abs_change > 2:
                cap_penalty = 6
                cap_status = "Stable"
            else:
                cap_penalty = 0
                cap_status = "Very Stable"
        else:
            cap_penalty = 5  # Penalty for missing data
            cap_status = "Unknown"
        
        factors['market_cap_stability'] = {
            'penalty': cap_penalty,
            'value': market_cap_change,
            'status': cap_status
        }
        
        # 3. Price Volatility (20% weight)
        price_changes = [
            row.get('price_change_percentage_1h', 0),
            row.get('price_change_percentage_24h', 0),
            row.get('price_change_percentage_7d', 0)
        ]
        
        volatility_score = np.std([x for x in price_changes if pd.notna(x)])
        if volatility_score > 2:
            vol_penalty = 20
            vol_status = "High"
        elif volatility_score > 1:
            vol_penalty = 12
            vol_status = "Moderate"
        elif volatility_score > 0.5:
            vol_penalty = 6
            vol_status = "Low"
        else:
            vol_penalty = 0
            vol_status = "Very Low"
        
        factors['volatility'] = {
            'penalty': vol_penalty,
            'value': volatility_score,
            'status': vol_status
        }
        
        # 4. Market Position (15% weight)
        market_cap = row.get('market_cap', 0)
        if market_cap > 50e9:  # >$50B
            position_bonus = 5
            position_status = "Dominant"
        elif market_cap > 10e9:  # >$10B
            position_bonus = 3
            position_status = "Major"
        elif market_cap > 1e9:  # >$1B
            position_bonus = 0
            position_status = "Established"
        else:
            position_bonus = -10
            position_status = "Small"
        
        factors['market_position'] = {
            'penalty': -position_bonus,  # Bonus becomes negative penalty
            'value': market_cap / 1e9,
            'status': position_status
        }
        
        # Calculate final score
        final_score = base_score - sum(factor['penalty'] for factor in factors.values())
        final_score = max(0, min(100, final_score))
        
        # Determine overall status
        if final_score >= 90:
            overall_status = "Excellent"
            status_class = "excellent"
        elif final_score >= 75:
            overall_status = "Good"
            status_class = "good"
        elif final_score >= 50:
            overall_status = "Warning"
            status_class = "warning"
        else:
            overall_status = "Critical"
            status_class = "critical"
        
        return {
            'score': final_score,
            'status': overall_status,
            'status_class': status_class,
            'factors': factors
        }
    
    def generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable insights from the data"""
        insights = []
        
        if df.empty:
            return ["⚠️ No data available for analysis"]
        
        # Market concentration analysis
        top_3_cap = df.nlargest(3, 'market_cap')['market_cap'].sum()
        total_cap = df['market_cap'].sum()
        concentration = (top_3_cap / total_cap) * 100
        
        if concentration > 80:
            insights.append(f"🎯 **High Market Concentration**: Top 3 stablecoins control {concentration:.1f}% of the market, indicating potential systemic risk.")
        
        # Peg deviation analysis
        deviations = [(row['current_price'] - 1) * 100 for _, row in df.iterrows()]
        avg_deviation = np.mean([abs(d) for d in deviations])
        
        if avg_deviation > 0.5:
            insights.append(f"⚠️ **Elevated Peg Risk**: Average deviation is {avg_deviation:.3f}%, above normal ranges.")
        elif avg_deviation < 0.1:
            insights.append(f"✅ **Strong Peg Stability**: Average deviation only {avg_deviation:.3f}%, excellent stability.")
        
        # Volatility insights
        volatile_coins = df[df['price_change_percentage_24h'].abs() > 2]
        if len(volatile_coins) > 0:
            insights.append(f"📈 **Volatility Alert**: {len(volatile_coins)} stablecoin(s) showing >2% price movement in 24h.")
        
        # Market cap insights
        growing_coins = df[df['market_cap_change_percentage_24h'] > 5]
        shrinking_coins = df[df['market_cap_change_percentage_24h'] < -5]
        
        if len(growing_coins) > len(shrinking_coins):
            insights.append("📊 **Net Supply Expansion**: More stablecoins are growing than shrinking, indicating market confidence.")
        elif len(shrinking_coins) > len(growing_coins):
            insights.append("📉 **Net Supply Contraction**: More stablecoins shrinking than growing, potential market stress.")
        
        return insights if insights else ["📊 All metrics within normal ranges"]

# Initialize components
config = ConfigManager()
data_fetcher = DataFetcher(config)
analytics = AnalyticsEngine()

# Sidebar with enhanced information
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 16px; margin-bottom: 25px; text-align: center;">
        <h2 style="color: white; margin: 0; font-weight: 700;">🚀 Stablecoin Monitor Pro</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 14px;">Advanced Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration status
    is_valid, config_messages = config.validate_config()
    
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
        🚀 Stablecoin Health Monitor Pro
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

# Load all data with progress tracking
with st.spinner("🔄 Loading comprehensive market data..."):
    progress_bar = st.progress(0)
    
    progress_bar.progress(25)
    stablecoin_df = data_fetcher.fetch_stablecoin_data()
    
    progress_bar.progress(50)
    dominance_df = data_fetcher.fetch_dominance_data()
    
    progress_bar.progress(75)
    onchain_df = data_fetcher.fetch_dune_onchain_data()
    
    progress_bar.progress(100)
    progress_bar.empty()

# Data quality alerts
if stablecoin_df.empty:
    st.markdown("""
    <div class='alert-banner'>
        🚨 <strong>Data Alert:</strong> Unable to fetch fresh market data. Showing cached information.
    </div>
    """, unsafe_allow_html=True)

# === ENHANCED MARKET OVERVIEW SECTION ===
if section == "📊 Market Overview":
    st.markdown("## 📊 Comprehensive Market Overview")
    
    if not stablecoin_df.empty:
        # Generate insights
        insights = analytics.generate_insights(stablecoin_df)
        
        # Insights panel
        st.markdown("### 🧠 AI-Powered Market Insights")
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
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Avg. Price</h3>
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
            # Calculate overall health score
            health_scores = []
            for _, row in stablecoin_df.iterrows():
                health_score = analytics.calculate_advanced_health_score(row)
                health_scores.append(health_score['score'])
            
            avg_health_score = np.mean(health_scores) if health_scores else 0
            
            if avg_health_score >= 90:
                score_class = "health-score-excellent"
            elif avg_health_score >= 75:
                score_class = "health-score-good"
            elif avg_health_score >= 50:
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
        
        # Market dominance chart
        st.markdown("### 🏆 Market Dominance")
        
        if not dominance_df.empty:
            fig = px.pie(
                dominance_df, 
                values='Market Cap', 
                names='Stablecoin',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Market Cap: $%{value:,.0f}<br>Dominance: %{percent}'
            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No dominance data available")
        
        st.markdown("---")
        
        # Enhanced stablecoin table with health scores
        st.markdown("### 📋 Stablecoin Performance Details")
        
        if not stablecoin_df.empty:
            # Calculate health scores for each stablecoin
            health_data = []
            for _, row in stablecoin_df.iterrows():
                health_score = analytics.calculate_advanced_health_score(row)
                health_data.append({
                    'symbol': row['symbol'].upper(),
                    'name': row['name'],
                    'price': row['current_price'],
                    'market_cap': row['market_cap'],
                    '24h_change': row.get('price_change_percentage_24h', 0),
                    'health_score': health_score['score'],
                    'health_status': health_score['status']
                })
            
            health_df = pd.DataFrame(health_data)
            
            # Format the table for display
            display_df = health_df.copy()
            display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.2f}B")
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:.4f}")
            display_df['24h_change'] = display_df['24h_change'].apply(lambda x: f"{x:.2f}%")
            display_df['health_score'] = display_df['health_score'].apply(lambda x: f"{x:.1f}/100")
            
            # Reset index to start from 1 instead of 0
            display_df.index = range(1, len(display_df) + 1)
            
            # Display the table
            st.dataframe(
                display_df,
                column_config={
                    "symbol": "Symbol",
                    "name": "Name",
                    "price": "Price (USD)",
                    "market_cap": "Market Cap",
                    "24h_change": "24h Change",
                    "health_score": "Health Score",
                    "health_status": "Status"
                },
                use_container_width=True
            )
        else:
            st.warning("No stablecoin data available")

# === PEG STABILITY SECTION ===
elif section == "⚖️ Peg Stability":
    st.markdown("## ⚖️ Peg Stability Analysis")
    
    if not stablecoin_df.empty:
        # Create peg deviation visualization
        peg_data = []
        for _, row in stablecoin_df.iterrows():
            deviation = (row['current_price'] - 1) * 100  # Convert to percentage
            peg_data.append({
                'Stablecoin': row['symbol'].upper(),
                'Deviation (%)': deviation,
                'Price': row['current_price'],
                'Market Cap': row['market_cap']
            })
        
        peg_df = pd.DataFrame(peg_data)
        
        # Create bar chart for peg deviations
        fig = px.bar(
            peg_df, 
            x='Stablecoin', 
            y='Deviation (%)',
            color='Deviation (%)',
            color_continuous_scale=['#FF6B6B', '#FFD93D', '#4ECDC4', '#00FF88'],
            range_color=[-2, 2],
            title='Peg Deviation from $1.00'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Stablecoin",
            yaxis_title="Deviation from $1.00 (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            coloraxis_showscale=False
        )
        
        # Add horizontal lines for thresholds
        fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
        fig.add_hline(y=1, line_dash="dash", line_color="#FFD93D", line_width=1, opacity=0.7)
        fig.add_hline(y=-1, line_dash="dash", line_color="#FFD93D", line_width=1, opacity=0.7)
        fig.add_hline(y=2, line_dash="dash", line_color="#FF6B6B", line_width=1, opacity=0.7)
        fig.add_hline(y=-2, line_dash="dash", line_color="#FF6B6B", line_width=1, opacity=0.7)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Peg stability insights
        st.markdown("### 📊 Stability Insights")
        
        # Count stablecoins in different deviation ranges
        within_1_percent = len(peg_df[abs(peg_df['Deviation (%)']) <= 1])
        within_2_percent = len(peg_df[(abs(peg_df['Deviation (%)']) > 1) & (abs(peg_df['Deviation (%)']) <= 2)])
        beyond_2_percent = len(peg_df[abs(peg_df['Deviation (%)']) > 2])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Within 1%</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #00FF88;">{within_1_percent}</p>
                <p style="margin: 5px 0 0 0; font-size: 12px; color: #B8BCC8;">Excellent Stability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Within 2%</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #FFD93D;">{within_2_percent}</p>
                <p style="margin: 5px 0 0 0; font-size: 12px; color: #B8BCC8;">Moderate Deviation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Beyond 2%</h3>
                <p style="margin: 0; font-size: 24px; font-weight: 700; color: #FF6B6B;">{beyond_2_percent}</p>
                <p style="margin: 5px 0 0 0; font-size: 12px; color: #B8BCC8;">High Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed peg deviation table
        st.markdown("### 📋 Detailed Peg Analysis")
        
        # Format table for display
        peg_display_df = peg_df.copy()
        peg_display_df['Market Cap'] = peg_display_df['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B")
        peg_display_df['Price'] = peg_display_df['Price'].apply(lambda x: f"${x:.4f}")
        peg_display_df['Deviation (%)'] = peg_display_df['Deviation (%)'].apply(lambda x: f"{x:.3f}%")
        
        # Add status indicator
        def get_peg_status(deviation):
            abs_dev = abs(deviation)
            if abs_dev <= 0.5:
                return "🟢 Excellent"
            elif abs_dev <= 1.0:
                return "🟡 Good"
            elif abs_dev <= 2.0:
                return "🟠 Warning"
            else:
                return "🔴 Critical"
        
        peg_display_df['Status'] = peg_df['Deviation (%)'].apply(get_peg_status)
        
        # Reset index to start from 1 instead of 0
        peg_display_df.index = range(1, len(peg_display_df) + 1)
        
        # Display the table
        st.dataframe(
            peg_display_df,
            column_config={
                "Stablecoin": "Stablecoin",
                "Deviation (%)": "Deviation",
                "Price": "Price",
                "Market Cap": "Market Cap",
                "Status": "Status"
            },
            use_container_width=True
        )

# === SUPPLY ANALYSIS SECTION ===
elif section == "📦 Supply Analysis":
    st.markdown("## 📦 Supply Analysis")
    
    if not stablecoin_df.empty:
        # Market cap distribution
        st.markdown("### 💰 Market Capitalization Distribution")
        
        # Sort by market cap
        supply_df = stablecoin_df.sort_values('market_cap', ascending=False).copy()
        
        # Create horizontal bar chart
        fig = px.bar(
            supply_df,
            y='symbol',
            x='market_cap',
            orientation='h',
            title='Stablecoin Market Capitalization',
            color='market_cap',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Market Cap (USD)",
            yaxis_title="Stablecoin",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            coloraxis_showscale=False
        )
        
        # Format x-axis to show billions
        fig.update_xaxes(tickprefix="$", ticksuffix="B", tickformat=".2s")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market cap change analysis
        st.markdown("### 📈 Market Cap Changes (24h)")
        
        # Filter out coins with no change data
        change_df = supply_df[pd.notna(supply_df['market_cap_change_percentage_24h'])].copy()
        
        if not change_df.empty:
            # Create change visualization
            fig = px.bar(
                change_df,
                y='symbol',
                x='market_cap_change_percentage_24h',
                orientation='h',
                title='24h Market Cap Change (%)',
                color='market_cap_change_percentage_24h',
                color_continuous_scale=['#FF6B6B', '#4ECDC4', '#00FF88'],
                range_color=[-10, 10]
            )
            
            fig.update_layout(
                height=400,
                xaxis_title="24h Change (%)",
                yaxis_title="Stablecoin",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                coloraxis_showscale=False
            )
            
            # Add vertical line at zero
            fig.add_vline(x=0, line_dash="solid", line_color="white", line_width=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Supply change insights
            growing = len(change_df[change_df['market_cap_change_percentage_24h'] > 0])
            shrinking = len(change_df[change_df['market_cap_change_percentage_24h'] < 0])
            stable = len(change_df[change_df['market_cap_change_percentage_24h'] == 0])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Growing</h3>
                    <p style="margin: 0; font-size: 24px; font-weight: 700; color: #00FF88;">{growing}</p>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #B8BCC8;">Supply Expansion</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Shrinking</h3>
                    <p style="margin: 0; font-size: 24px; font-weight: 700; color: #FF6B6B;">{shrinking}</p>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #B8BCC8;">Supply Contraction</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">Stable</h3>
                    <p style="margin: 0; font-size: 24px; font-weight: 700; color: #4ECDC4;">{stable}</p>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #B8BCC8;">No Change</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed supply table
        st.markdown("### 📋 Detailed Supply Metrics")
        
        # Prepare data for display
        supply_display_df = supply_df[['symbol', 'name', 'market_cap', 'market_cap_change_percentage_24h', 'total_volume']].copy()
        supply_display_df['market_cap'] = supply_display_df['market_cap'].apply(lambda x: f"${x/1e9:.2f}B")
        supply_display_df['market_cap_change_percentage_24h'] = supply_display_df['market_cap_change_percentage_24h'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        )
        supply_display_df['total_volume'] = supply_display_df['total_volume'].apply(lambda x: f"${x/1e6:.2f}M")
        
        # Reset index to start from 1 instead of 0
        supply_display_df.index = range(1, len(supply_display_df) + 1)
        
        # Display the table
        st.dataframe(
            supply_display_df,
            column_config={
                "symbol": "Symbol",
                "name": "Name",
                "market_cap": "Market Cap",
                "market_cap_change_percentage_24h": "24h Change",
                "total_volume": "24h Volume"
            },
            use_container_width=True
        )

# === ON-CHAIN ACTIVITY SECTION ===
elif section == "⚡ On-Chain Activity":
    st.markdown("## ⚡ On-Chain Activity Analysis")
    
    if not onchain_df.empty:
        # Display latest on-chain data
        st.markdown("### 📊 Latest On-Chain Supply Changes")
        
        # Get the latest week's data
        latest_data = onchain_df.iloc[0]
        week = latest_data['week'].strftime('%B %d, %Y')
        
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
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly changes analysis
        st.markdown("### 🔄 Weekly Supply Changes")
        
        if len(onchain_df) > 1:
            # Calculate week-over-week changes
            changes = {}
            current_week = onchain_df.iloc[0]
            previous_week = onchain_df.iloc[1]
            
            for coin in ['USDC supply', 'DAI supply', 'LUSD supply']:
                if coin in current_week and coin in previous_week:
                    current_val = current_week[coin]
                    previous_val = previous_week[coin]
                    change = ((current_val - previous_val) / previous_val) * 100
                    changes[coin.replace(' supply', '')] = change
            
            # Display changes
            if changes:
                col1, col2, col3 = st.columns(3)
                
                for i, (coin, change) in enumerate(changes.items()):
                    with [col1, col2, col3][i]:
                        color = "#00FF88" if change > 0 else "#FF6B6B"
                        icon = "📈" if change > 0 else "📉"
                        
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style="margin: 0 0 10px 0; color: #B8BCC8; font-size: 14px;">{coin} Weekly Change</h3>
                            <p style="margin: 0; font-size: 24px; font-weight: 700; color: {color};">
                                {icon} {change:.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Detailed on-chain data table
        st.markdown("### 📋 Historical On-Chain Data")
        
        # Prepare data for display
        onchain_display_df = onchain_df.copy()
        
        # Format columns
        for col in onchain_display_df.columns:
            if col != 'week' and 'supply' in col:
                onchain_display_df[col] = onchain_display_df[col].apply(lambda x: f"${x/1e9:.2f}B")
        
        # Format date
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

# Footer with data freshness information
st.markdown("---")
last_updated = datetime.now().strftime('%B %d, %Y at %H:%M:%S UTC')
st.markdown(f"""
<div style="text-align: center; color: #B8BCC8; font-size: 14px; margin-top: 30px;">
    <p>🕐 Data last updated: {last_updated}</p>
    <p>🔁 Automatic refresh every 24 hours | 💾 Fallback caching enabled</p>
    <p>Built with ❤️ using Streamlit, CoinGecko API, and Dune Analytics</p>
</div>
""", unsafe_allow_html=True)