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
        avg_