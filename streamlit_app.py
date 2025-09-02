# from pathlib import Path
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.express as px
# import matplotlib.pyplot as plt
# import streamlit as st

# # Page configuration
# st.set_page_config(page_title="Solana Validators Dashboard", layout="wide")

# # Load the data
# df_validators = joblib.load('notebook_and_app/df_validators.joblib')
# df_expanded = joblib.load('notebook_and_app/df_expanded.joblib')
# df_cleaned = joblib.load('notebook_and_app/df_cleaned.joblib')
# df_tps = joblib.load('notebook_and_app/df_tps.joblib')
# df_supply = joblib.load('notebook_and_app/df_supply.joblib')
# df_fees = joblib.load('notebook_and_app/df_fees.joblib')
# df_inflation = joblib.load('notebook_and_app/df_inflation.joblib')
# df_epochs = joblib.load('notebook_and_app/df_epochs.joblib')

# # Prepare merged DataFrame
# df_expanded['vote_account'] = df_expanded['votePubkey']
# df_merge = df_validators.merge(df_expanded, how = 'left', on = ['vote_account', 'epoch'] )
# df_merge["active_stake_SOL"] = df_merge["active_stake"] / 1e9
# df_merge['name'] = df_merge['name'].replace([None, 'None'], 'Unknown')


# # Sidebar with About Section
# with st.sidebar:
#     st.markdown(
#         """
#         <div style="background-color: #8A4AF3; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
#             <h3 style="color: #00FFF0; text-align: center; margin: 0;">About Solana Validators Dashboard</h3>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
#     st.markdown(
#         """
#         ### 
#         Welcome to the Solana Validators Dashboard, a simple and user-friendly web-based tool designed to help you explore and understand the Solana blockchain like never before! This dashboard lets you dive into validator performance, track staking rewards, and get a clear picture of key network metrics—all in one place.

#         Built with ease of use in mind, it offers an interactive interface where you can:
#         - Check out how validators are performing across different epochs.
#         - See detailed staking rewards and active stake trends with handy charts.
#         - Get a quick overview of the network, including total validators, transactions per second (TPS), and SOL supply breakdown.

#         Whether you're a blockchain enthusiast, a validator operator, or just curious about Solana, this dashboard makes it simple to stay informed. Created with love using Streamlit, Pandas, Plotly, and Matplotlib, it’s designed to bring Solana’s data to life with a sleek, Solana-inspired look. Enjoy exploring!
#         """,
#         unsafe_allow_html=True
#     )

# # # Sidebar
# # st.sidebar.title("Navigation")
# # section = st.sidebar.radio("Go to", ["Overview", "Validator Performance", "Rewards"])

# # Navigation at the top of the page
# st.markdown(
#     """
#     <div style="background: linear-gradient(90deg, #00FFF0 0%, #8A4AF3 100%); padding: 10px; border-radius: 5px; margin-bottom: 20px;">
#         <h2 style="color: white; text-align: center; margin: 0;">Solana Validators Dashboard</h2>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # Navigation buttons (horizontal radio buttons)
# section = st.radio(
#     "Navigate to Section",
#     ["Overview", "Validator Performance", "Staking Reward"],
#     index=0,  # Default to Overview
#     format_func=lambda x: x,  # Display labels as-is
#     horizontal=True,  # Horizontal layout
#     key="nav_radio",
#     help="Select a section to explore the dashboard."
# )

# # Add some styling to the radio buttons using CSS
# st.markdown(
#     """
#     <style>
#     div[role="radiogroup"] {
#         display: flex;
#         justify-content: center;
#         margin-bottom: 20px;
#     }
#     div[role="radiogroup"] label {
#         margin: 0 15px;
#         font-size: 18px;
#         color: #00FFF0;
#         background-color: #8A4AF3;
#         padding: 8px 16px;
#         border-radius: 5px;
#         transition: background-color 0.3s;
#     }
#     div[role="radiogroup"] label:hover {
#         background-color: #9945FF;
#     }
#     div[role="radiogroup"] input[type="radio"]:checked + label {
#         background-color: #00FFF0;
#         color: #8A4AF3;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --- Overview ---
# if section == "Overview":
#     with st.container():
#         st.markdown(
#         """
#         <div style="border: 1px solid #CCC; padding: 10px; border-radius: 5px; background-color: #8A4AF3;">
#         """,
#         unsafe_allow_html=True,
#         )
        
#         st.title("Network Overview")

#         total_validators = df_expanded['vote_account'].nunique()

#         # Circulating and non-circulating supply
#         # Use numbers for calculations and display-formatted strings for metrics only
#         circulating_val = int(df_supply['circulating_sol'][0])
#         non_circulating_val = int(df_supply['nonCirculating_sol'][0])

#         circulating = f"{circulating_val:,}"
#         non_circulating = f"{non_circulating_val:,}"

#         # TPS and fees
#         tps = f"{int(df_tps['tps'][0]):,}"
#         avg_fee_usd = df_fees['avg_fee_usd'][0]

#         # Latest stake
#         df_epochs['total_reward_SOL'] = df_epochs['total_rewards'] / 1e9
#         df_epochs['total_active_stake_SOL'] = df_epochs['total_active_stake'] / 1e9
#         df_epochs['total_reward_SOL'] = df_epochs['total_reward_SOL'].astype(object)
#         df_epochs['total_active_stake_SOL'] = df_epochs['total_active_stake_SOL'].astype(object)
#         df_epochs.loc[0, ['total_reward_SOL', 'total_active_stake_SOL']] = 'ongoing'
#         concluded_epochs = df_epochs[df_epochs['total_active_stake_SOL'] != 'ongoing']
#         latest_concluded = concluded_epochs.sort_values(by='epoch', ascending=False).iloc[0]
#         latest_active_stake_SOL = f"{int(latest_concluded['total_active_stake_SOL']):,}"

#         latest_epoch = latest_concluded['epoch']

    
#         col1,col2,col3 = st.columns(3)
#         col1.metric("Validators", total_validators)
#         col2.metric("Epoch", f"{latest_epoch}")
#         col3.metric("TPS", f"{tps}")

#         col4, col5 = st.columns(2)
        
#         col4.metric("Avg Fee (USD)", f"${avg_fee_usd:.6f}")
#         col5.metric("Total Active Stake (SOL)", f"{latest_active_stake_SOL} SOL")
        
#         # Spacer
#     st.markdown("---")

#     with st.container():
#         st.markdown(
#         """
#         <div style="border: 1px solid #CCC; padding: 10px; border-radius: 5px; background-color: #8A4AF3;">
#         """,
#         unsafe_allow_html=True,
#     )

#         # Token Supply Pie Chart Section
#         st.subheader("SOL Supply Breakdown")

#         col6, col7 = st.columns(2)
#         #col1.metric("Validators", total_validators)
#         col6.metric("Circulating Supply", f"{circulating} SOL")
#         col7.metric("Non-Circulating Supply", f"{non_circulating} SOL")


#         supply_data = pd.DataFrame({
#         'Supply Type': ['Circulating', 'Non-Circulating'],
#         'Amount': [circulating_val, non_circulating_val]
#         })

#         fig = px.pie(
#             supply_data,
#             names='Supply Type',
#             values='Amount',
#             title="Solana Token Supply Distribution",
#             hole=0.4,
#         )

#         fig.update_traces(
#             textinfo='label+percent',
#             hovertemplate='%{label}: %{value:,.0f} SOL<br>(%{percent})',
#             marker=dict(colors=['#00FFF0', '#8A4AF3'])
#         )

#         st.plotly_chart(fig, use_container_width=True)

        
# # --- Validator Performance ---
# elif section == "Validator Performance":
#     with st.container():
#         st.markdown(
#             """
#             <div style="border: 1px solid #CCC; padding: 10px; border-radius: 5px; background-color: #8A4AF3;">
#             """,
#             unsafe_allow_html=True,
#         )
        
#         st.title("Validator Performance")

#         # Prepare data for filtering
#         latest_epoch = df_cleaned['epoch'].max()
#         df_cleaned = df_cleaned.dropna()
#         df_cleaned = df_cleaned[~((df_cleaned['epoch'] == latest_epoch) & df_cleaned['total_rewards'].isna() & df_cleaned['total_active_stake'].isna())].reset_index(drop=True)

#         # Search inputs
#         vote_account_search = st.text_input("Search by Vote Account", key="vote_account_search")
#         name_search = st.text_input("Search by Name", key="name_search")

#         # Filter the data
#         filtered_data = df_cleaned.copy()
#         if vote_account_search:
#             filtered_data = filtered_data[filtered_data['vote_account'].str.contains(vote_account_search, case=False, na=False)]
#         if name_search:
#             filtered_data = filtered_data[filtered_data['name'].str.contains(name_search, case=False, na=False)]

#         # Create previous_validator_performance from filtered_data with selected columns
#         previous_validator_performance = filtered_data[
#             ['name', 'activatedStake_SOL', 'commission', 'credits_earned', 'details']
#         ]

#         # Display previous_validator_performance as the first table with pagination
#         st.subheader(f"Previous Validator Performance (Epoch {latest_epoch})")

#         # Pagination
#         rows_per_page = 100
#         total_rows = len(previous_validator_performance)
#         total_pages = (total_rows - 1) // rows_per_page + 1
#         page_options = [f"{i*rows_per_page + 1}-{min((i+1)*rows_per_page, total_rows)}" for i in range(total_pages)]
        
#         if total_rows > rows_per_page:
#             selected_range = st.selectbox(
#                 "Select Range of Validators to View",
#                 page_options,
#                 index=0,
#                 key="validator_page_select"
#             )
#             start_idx = int(selected_range.split('-')[0]) - 1
#             end_idx = int(selected_range.split('-')[1])
#             display_data = previous_validator_performance.iloc[start_idx:end_idx]
#         else:
#             display_data = previous_validator_performance

#         # Format the table for display
#         display_data = display_data.copy()
#         display_data['activatedStake_SOL'] = display_data['activatedStake_SOL'].round(2)
#         display_data = display_data.rename(columns={
#             'name': 'Name',
#             'activatedStake_SOL': 'Active Stake (SOL)',
#             'commission': 'Commission (%)',
#             'credits_earned': 'Epoch Credits',
#             'details': 'Details'
#         })

#         # Display the table
#         st.dataframe(display_data, use_container_width=True)


#                 # Spacer
#         st.markdown("---")

#         with st.container():
#             st.markdown(
#             """
#             <div style="border: 1px solid #CCC; padding: 10px; border-radius: 5px; background-color: #8A4AF3;">
#             """,
#             unsafe_allow_html=True,
#         )

#         # Bar Chart: Top 10 Validators by Active Stake from df_merge
#         # Prepare top_10_validators
#         df_merge['name'] = df_merge['name'].replace([None, 'None'], 'Unknown')
#         df_merge['active_stake_SOL'] = pd.to_numeric(df_merge['active_stake_SOL'], errors='coerce')
#         top_10_validators = df_merge[['name', 'active_stake_SOL']].sort_values(
#             by='active_stake_SOL', ascending=False).head(10)
#         top_10_validators.reset_index(drop=True, inplace=True)

#         st.subheader("Top 10 Validators by Active Stake")
#         stake_fig = px.bar(
#             top_10_validators,
#             x='name',
#             y='active_stake_SOL',
#             title="Top 10 Validators by Active Stake",
#             labels={'active_stake_SOL': 'Active Stake (SOL)', 'name': 'Validator'},
#             text_auto='.2f'
#         )
#         stake_fig.update_traces(marker_color='#00FFF0')  # Use Solana cyan
#         stake_fig.update_layout(
#             yaxis_tickformat=',.2f',
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font_color='white',
#             xaxis_tickangle=-45  # Rotate x-axis labels for readability
#         )
#         st.plotly_chart(stake_fig, use_container_width=True)


# # --- Staking Reward ---
# elif section == "Staking Reward":
#     with st.container():
#         st.markdown(
#             """
#             <div style="border: 1px solid #CCC; padding: 10px; border-radius: 5px; background-color: #8A4AF3;">
#             """,
#             unsafe_allow_html=True,
#         )
        
#         st.title("Staking Reward")

#         # Prepare staking rewards data
#         df_epochs = df_epochs.copy()
#         df_epochs['total_reward_SOL'] = df_epochs['total_rewards'] / 1e9
#         df_epochs['total_active_stake_SOL'] = df_epochs['total_active_stake'] / 1e9

#         # Convert to object type to allow string + float
#         df_epochs['total_reward_SOL'] = df_epochs['total_reward_SOL'].astype('object')
#         df_epochs['total_active_stake_SOL'] = df_epochs['total_active_stake_SOL'].astype('object')

#         df_epochs.loc[0, ['total_reward_SOL', 'total_active_stake_SOL']] = 'ongoing'

#         staking_rewards = df_epochs[
#             ['epoch', 'total_reward_SOL', 'total_active_stake_SOL']
#         ]

#          # Pagination setup
#         rows_per_page = 10
#         total_rows = len(staking_rewards)
#         total_pages = (total_rows - 1) // rows_per_page + 1

#         # Initialize session state for page navigation
#         if 'current_page' not in st.session_state:
#             st.session_state.current_page = 0

#         # Navigation buttons
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col1:
#             if st.button("Previous 10") and st.session_state.current_page > 0:
#                 st.session_state.current_page -= 1
#         with col2:
#             st.write(f"Page {st.session_state.current_page + 1} of {total_pages} (Showing rows {st.session_state.current_page * rows_per_page + 1} to {min((st.session_state.current_page + 1) * rows_per_page, total_rows)})")
#         with col3:
#             if st.button("Next 10") and (st.session_state.current_page + 1) < total_pages:
#                 st.session_state.current_page += 1

#         # Slice the data for the current page
#         start_idx = st.session_state.current_page * rows_per_page
#         end_idx = min((st.session_state.current_page + 1) * rows_per_page, total_rows)
#         display_rewards = staking_rewards.iloc[start_idx:end_idx].copy()

#         # Display staking rewards table
#         st.subheader("Staking Rewards by Epoch")
#         display_rewards = display_rewards.rename(columns={
#             'epoch': 'Epoch',
#             'total_reward_SOL': 'Total Reward (SOL)',
#             'total_active_stake_SOL': 'Total Active Stake (SOL)'
#         })

#         # Format numerical values where not 'ongoing'
#         display_rewards['Total Reward (SOL)'] = display_rewards['Total Reward (SOL)'].apply(
#             lambda x: f"{float(x):,.2f}" if x != 'ongoing' else x
#         )
#         display_rewards['Total Active Stake (SOL)'] = display_rewards['Total Active Stake (SOL)'].apply(
#             lambda x: f"{float(x):,.2f}" if x != 'ongoing' else x
#         )

#         st.dataframe(display_rewards, use_container_width=True)

        
#         # Staking Rewards and Active Stake per Epoch Graph
#         # Filter out 'ongoing' rows and convert types
#         st.subheader("Staking Rewards and Active Stake per Epoch")
#         staking_rewards_filtered = staking_rewards[
#             staking_rewards['total_active_stake_SOL'] != 'ongoing'
#         ].copy()

#         staking_rewards_filtered['total_reward_SOL'] = staking_rewards_filtered['total_reward_SOL'].astype(float)
#         staking_rewards_filtered['total_active_stake_SOL'] = staking_rewards_filtered['total_active_stake_SOL'].astype(float)

#         fig, ax1 = plt.subplots(figsize=(12, 6))
#         fig.patch.set_facecolor('#0e1117')  # Dark background for the figure
#         ax1.set_facecolor('#0e1117')        # Dark background for the bar axis

#         # Bar for rewards
#         ax1.bar(
#             staking_rewards_filtered['epoch'], 
#             staking_rewards_filtered['total_reward_SOL'], 
#             color="#00FFF0", 
#             label='Total Reward (SOL)'
#         )
#         ax1.set_xlabel("Epoch")
#         ax1.set_ylabel("Reward (SOL)", color="#00FFF0")
#         ax1.tick_params(axis='y', labelcolor="#00FFF0")

#         # Line for active stake
#         ax2 = ax1.twinx()
#         ax2.plot(
#             staking_rewards_filtered['epoch'], 
#             staking_rewards_filtered['total_active_stake_SOL'], 
#             color="#8A4AF3", 
#             marker='o', 
#             label='Total Active Stake (SOL)'
#         )
#         ax2.set_ylabel("Active Stake (SOL)", color="#8A4AF3")
#         ax2.tick_params(axis='y', labelcolor="#8A4AF3")

#         # Titles and layout
#         fig.suptitle("Staking Rewards and Active Stake per Epoch")
#         fig.tight_layout()

#         # Display in Streamlit
#         st.pyplot(fig)

#         st.markdown("---")




import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import plotly.express as px


# Page configuration
st.set_page_config(page_title="Stablecoin Health Monitor", layout="wide")

# Load the data
dominance_df = joblib.load("data/current_dominance_df.joblib")
clean_filtered_stables = joblib.load("data/stablecoins_filtered.joblib")

# Prepare merged DataFrame

# Sidebar with About Section
with st.sidebar:
    st.markdown(
        """
        <div style="background-color: #8A4AF3; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="color: #00FFF0; text-align: center; margin: 0;">About Stablecoin Health Monitor Dashboard</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        ### 
        Are stablecoins actually stable?  🤔
        
        Stablecoins were meant to fix crypto’s biggest problem — **volatility**. Pegged to the dollar, they should hold steady at $1. But the truth is, **stability is never guaranteed.**
        
        Stablecoins rise and fall with market pressure. If prices creep above 1 USD, new tokens get minted until supply cools things down. If they drop below 1 USD, supply shrinks as coins are burned or redeemed. It’s a constant tug-of-war between demand, liquidity, and trust. \
        Some fight this battle with cash reserves (USDC). Others use collateralized systems (DAI). Newcomers like PYUSD or RLUSD test new ideas — and new risks. Each reacts differently when stress hits, and that’s where things get interesting.  
        
        Stablecoins aren’t just tokens — they’re the backbone of DeFi. They dominate trading pairs, power lending markets, and act as the preferred unit of account across protocols. Their stability (or lack of it) directly shapes how capital flows.  

        ---

        Welcome to the Stablecoin Health Monitor Dashboard, a simple and user-friendly web-based tool designed to help you explore and understand the health and dominance of stablecoins in the cryptocurrency market! This dashboard lets you dive into stablecoin performance, track dominance metrics, and get a clear picture of key market indicators—all in one place.

        This dashboard is built with ease of use in mind, offering an interactive interface where you can:
        - Explore detailed stablecoin data and metrics.
        - Visualize dominance trends with interactive charts.
        - Observe token mints and burns activity and its influence on the circulating supply of these popular stable coins.
        - Access peg deviation of these major stables and how it changes over time. This tells if these stablecoins are actually stable or not.
        - Track DeFi usage of major Ethereum stablecoins **([USDC](https://etherscan.io/token/0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48), [DAI](https://etherscan.io/token/0x6b175474e89094c44da98b954eedeac495271d0f), [USDT](https://etherscan.io/token/0xdac17f958d2ee523a2206206994597c13d831ec7), [TUSD](https://etherscan.io/token/0x0000000000085d4780B73119b644AE5ecd22b376) [LUSD](https://etherscan.io/token/0x5f98805A4E8be255a32880FDeC7F6728C6568bA0), [sUSD](https://etherscan.io/token/0x57ab1ec28d129707052df4df418d58a2d46d5f51), [PYUSD](https://etherscan.io/token/0x6c3ea9036406852006290770BEdFcAbA0e23A0e8), [RLUSD](https://etherscan.io/token/0x8292bb45bf1ee4d140127049757c2e0ff06317ed), [USDS](https://etherscan.io/token/0xdc035d45d973e3ec169d2276ddab16f1e407384f), [BUSD](https://etherscan.io/token/0x4fabb145d64652a948d72533023f6e7a623c7c53), [FRAX](https://etherscan.io/token/0x853d955acef822db058eb8505911ed77f175b99e))**.
        - These indicators reveal the true **health** of each stablecoin — whether it’s resilient, fragile, or showing early signs of stress.  Stay informed about the latest market developments.

        Whether you're a crypto enthusiast, analyst, or just curious about stablecoins, this dashboard provides a comprehensive view to keep you updated. Created with love using Streamlit, Pandas, Plotly, and other powerful tools, it’s designed to bring stablecoin data to life with a sleek, intuitive look. Enjoy exploring!
        """,
        unsafe_allow_html=True
    )


# Navigation at the top of the page
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #00FFF0 0%, #8A4AF3 100%); padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center; margin: 0;">Stablecoin Health Monitor Dashboard</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation buttons (horizontal radio buttons)
section = st.radio(
    "Navigate to Section",
    ["Overview", "Validator Performance", "Staking Reward"],
    index=0,  # Default to Overview
    format_func=lambda x: x,  # Display labels as-is
    horizontal=True,  # Horizontal layout
    key="nav_radio",
    help="Select a section to explore the dashboard."
)




# st.header("Dominance Data (Raw)")
st.write(clean_filtered_stables)



# Interactive bar chart
fig = px.bar(
    dominance_df,
    x="Stablecoin",
    y="Dominance (%)",
    text="Dominance (%)",
    title="Stablecoin Dominance (%)",
    labels={"Stablecoin": "Stablecoin", "Dominance (%)": "Dominance (%)"},
    template="plotly_white"
)

# Add text on top of bars
fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

# Adjust layout
fig.update_layout(
    xaxis_title="Stablecoin",
    yaxis_title="Dominance (%)",
    height=600,
    width=800
)

fig.show()

st.plotly_chart(fig, use_container_width=True)





fig = px.pie(
    dominance_df,
    names="Stablecoin",
    values="Dominance (%)",
    title="Stablecoin Dominance (%)",
    hole=0.3,  # makes it a donut chart
    template="plotly_white"
)

fig.update_traces(textinfo="label+percent", hovertemplate="Stablecoin: %{label}<br>Dominance: %{value:.2f}%")

fig.show()

st.plotly_chart(fig, use_container_width=True)




# # Load environment variables
# load_dotenv()

# # Set API keys
# dune_api_key = os.getenv("DEFI_JOSH_DUNE_QUERY_API_KEY")

# # Create Dune client instance
# dune = DuneClient(dune_api_key)

# # Fetch data from Dune
# query_result = dune.get_latest_result(5681885)

# # Convert rows into DataFrame
# stablecoin_data = pd.DataFrame(query_result.result.rows)

# # Create dashboard
# st.title("Stablecoin Health Monitor")

# st.header("Stablecoin Data (Raw)")
# st.write(stablecoin_data)

# # Example chart (adjust column names to match your query output)
# if "name" in stablecoin_data.columns and "supply" in stablecoin_data.columns:
#     st.header("Stablecoin Supply")
#     st.bar_chart(stablecoin_data.set_index("name")["supply"])

# # Sidebar filter (only if columns exist)
# if "name" in stablecoin_data.columns:
#     stablecoin_filter = st.sidebar.selectbox("Select Stablecoin", stablecoin_data["name"].unique())
#     filtered = stablecoin_data[stablecoin_data["name"] == stablecoin_filter]
#     st.write(f"Filtered Data for {stablecoin_filter}", filtered)
