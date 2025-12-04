import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import subprocess
import time
import os
import sys
from pathlib import Path
import json
import glob
from scipy.stats import norm

# Import the training monitor
from training_monitor import TrainingMonitor

# Page configuration
st.set_page_config(
    page_title="Derivative Hedging RL",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with softer colors
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-blue: #4a9eff;
        --bg-dark: #0e1117;
        --text-light: #e0e0e0;
        --text-dim: #9ca3af;
        --border-color: #2d3748;
    }
    
    /* Global styles */
    .main {
        background-color: var(--bg-dark);
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--primary-blue) !important;
        font-weight: 600;
    }
    
    /* Text */
    p, div, span, label {
        color: var(--text-light) !important;
    }
    
    /* Metric boxes */
    [data-testid="stMetricValue"] {
        color: var(--primary-blue) !important;
        font-size: 2rem !important;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-dim) !important;
        font-size: 0.9rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: transparent;
        color: var(--primary-blue);
        border: 2px solid var(--primary-blue);
        border-radius: 5px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-blue);
        color: var(--bg-dark);
        box-shadow: 0 0 15px rgba(74, 158, 255, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: rgba(45, 55, 72, 0.3);
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-dim);
        background-color: transparent;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary-blue);
        border-bottom: 2px solid var(--primary-blue);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 24, 32, 0.95);
        border-right: 1px solid var(--border-color);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: rgba(45, 55, 72, 0.5);
        color: var(--text-light);
        border: 1px solid var(--border-color);
        border-radius: 5px;
    }
    
    /* Warning/Info boxes */
    .stAlert {
        background-color: rgba(74, 158, 255, 0.1);
        border: 1px solid var(--primary-blue);
        border-radius: 5px;
        color: var(--text-light);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: var(--primary-blue);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Stock ticker dropdown
popular_stocks = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ']
ticker = st.sidebar.selectbox(
    "Stock Ticker",
    options=popular_stocks,
    index=0,
    help="Select a stock ticker for analysis"
)

# Date range
st.sidebar.subheader("Market Data Period")
st.sidebar.caption("Select date range for fetching historical stock prices from Yahoo Finance. This is used ONLY for Market Intelligence tab analysis, NOT for training.")

end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)

start = st.sidebar.date_input(
    "Start Date", 
    start_date,
    help="Beginning date for downloading historical stock data"
)
end = st.sidebar.date_input(
    "End Date", 
    end_date,
    help="End date for downloading historical stock data"
)

# ==================== MODEL REGISTRY ====================
st.sidebar.markdown("---")
st.sidebar.subheader("Model Registry")

# Scan for models in saved_models_real folder
model_folder = "./saved_models_real/"
os.makedirs(model_folder, exist_ok=True)
model_files = glob.glob(os.path.join(model_folder, "*.zip"))

if model_files:
    # Extract just the filenames
    model_names = [os.path.basename(f) for f in model_files]
    
    # Model selection dropdown
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        options=model_names,
        help="Choose a trained model to view details or use in Man vs Machine"
    )
    
    # Get full path of selected model
    selected_model_path = os.path.join(model_folder, selected_model_name)
    
    # Display metadata
    if os.path.exists(selected_model_path):
        file_size_mb = os.path.getsize(selected_model_path) / (1024 * 1024)
        creation_time = datetime.fromtimestamp(os.path.getctime(selected_model_path))
        
        st.sidebar.caption(f"**File Size:** {file_size_mb:.2f} MB")
        st.sidebar.caption(f"**Created:** {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Store selected model in session state for Man vs Machine
        st.session_state['selected_game_model'] = selected_model_path
        
        # Deletion controls
        st.sidebar.markdown("")
        enable_deletion = st.sidebar.checkbox("Enable Deletion", value=False)
        
        if enable_deletion:
            if st.sidebar.button("Delete Model", type="secondary"):
                try:
                    # Also try to delete the corresponding normalizer file
                    norm_file = selected_model_path.replace(".zip", "").replace("ppo_", "vec_normalize_") + ".pkl"
                    
                    os.remove(selected_model_path)
                    st.sidebar.success(f"Deleted {selected_model_name}")
                    
                    # Delete normalizer if exists
                    if os.path.exists(norm_file):
                        os.remove(norm_file)
                        st.sidebar.success("Also deleted normalizer file")
                    
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to delete: {str(e)}")
        else:
            st.sidebar.caption("⚠️ Check 'Enable Deletion' to delete models")
else:
    st.sidebar.info("No models found in saved_models_real/")
    st.sidebar.caption("Train a model first using Real Market Data")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "MARKET INTELLIGENCE",
    "NEURAL TRAINING",
    "RISK MANAGER",
    "BACKTEST LABORATORY"
])

# ==================== TAB 1: MARKET INTELLIGENCE ====================
with tab1:
    st.header("Market Intelligence")
    
    try:
        # Fetch data
        with st.spinner(f"Fetching {ticker} data..."):
            data = yf.download(ticker, start=start, end=end, progress=False)
        
        if len(data) == 0:
            st.error("No data available for this ticker and date range.")
        else:
            # Clean the data first - handle multi-index columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Calculate metrics (extract scalar values properly)
            current_price = float(data['Close'].iloc[-1])
            daily_returns = data['Close'].pct_change().dropna()
            daily_vol = float(daily_returns.std())
            annual_vol = daily_vol * np.sqrt(252)
            
            # Get min/max as scalar values
            min_price = float(data['Close'].min())
            max_price = float(data['Close'].max())
            price_range = f"${min_price:.2f} - ${max_price:.2f}"
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Daily Volatility", f"{daily_vol*100:.2f}%")
            with col3:
                st.metric("Annual Volatility", f"{annual_vol*100:.2f}%")
            with col4:
                st.metric("Price Range", price_range)
            
            # Price chart
            st.subheader("Price History")
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=data.index.strftime('%Y-%m-%d'),
                y=data['Close'].values,
                mode='lines',
                name='Close Price',
                line=dict(color='#4a9eff', width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            fig_price.update_layout(
                template='plotly_dark',
                hovermode='x unified',
                height=400,
                margin=dict(l=50, r=20, t=30, b=50),
                xaxis_title='Date',
                yaxis_title='Price ($)',
                xaxis=dict(
                    tickangle=-45,
                    nticks=10
                ),
                showlegend=False
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Returns distribution and Volume in same row
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Returns Distribution")
                returns_pct = (daily_returns * 100).values
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=returns_pct,
                    nbinsx=40,
                    name='Daily Returns',
                    marker_color='#4a9eff',
                    hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
                ))
                fig_dist.update_layout(
                    template='plotly_dark',
                    xaxis_title='Daily Return (%)',
                    yaxis_title='Frequency',
                    height=350,
                    margin=dict(l=50, r=20, t=40, b=50),
                    showlegend=False
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.subheader("Volume")
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=data.index.strftime('%Y-%m-%d'),
                    y=data['Volume'].values,
                    name='Volume',
                    marker_color='#4a9eff',
                    hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
                ))
                fig_vol.update_layout(
                    template='plotly_dark',
                    xaxis_title='Date',
                    yaxis_title='Volume',
                    height=350,
                    margin=dict(l=50, r=20, t=40, b=50),
                    xaxis=dict(
                        tickangle=-45,
                        nticks=8
                    ),
                    showlegend=False
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Raw data expander
            with st.expander("View Raw Data"):
                # Simple reset_index approach - most reliable
                display_df = data.tail(100).reset_index()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

# ==================== TAB 2: NEURAL TRAINING ====================
with tab2:
    st.header("Neural Training Center")
    
    # Explanation section
    with st.expander("ℹ️ What is this?", expanded=False):
        st.markdown("""
        ### Training Data Types Explained
        
        **1. Standard (GBM - Synthetic)**
        - Uses **Geometric Brownian Motion** - a mathematical formula that simulates stock prices
        - Perfect for learning basic hedging patterns
        - Fast training (2-3 minutes)
        - No internet required
        - **Use Case**: Testing algorithms, learning basics
        
        **2. Real Market Data (Yahoo Finance)**  
        - Downloads **actual historical stock prices** from Yahoo Finance
        - Training Period: 2020-2023 (includes COVID crash, recovery, inflation)
        - Testing Period: 2024 (out-of-sample validation)
        - **Use Case**: Real-world performance validation
        
        ### Date Range (Sidebar)
        - **Start Date & End Date**: Only used for **Market Intelligence Tab**
        - Fetches historical data to display price charts and volatility
        - **NOT used for training** - training uses fixed 2020-2024 period
        
        ### How to Know Which Training is Running?
        - Look at the blue info box below showing current training mode
        - Check the terminal/command output window
        - Real Market: Downloads data and shows "REAL WORLD TRAINING"
        - Synthetic: Shows "AI TRADING SCHOOL - Training Session Starting"
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_timesteps = st.number_input(
            "Total Timesteps",
            min_value=10000,
            max_value=1000000,
            value=50000,
            step=10000,
            help="Number of training steps (higher = better but slower)"
        )
    
    with col2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=0.01,
            value=0.0003,
            step=0.0001,
            format="%.5f",
            help="Learning rate for neural network optimizer"
        )
    
    market_data_type = st.radio(
        "Training Data Source",
        ["Standard (GBM - Synthetic)", "Real Market Data (Yahoo Finance)"],
        help="Choose between synthetic data or real market data"
    )
    
    # Show additional options for Real Market Data
    if "Real Market" in market_data_type:
        st.markdown("### Real Market Data Configuration")
        
        col_stock, col_dates = st.columns([1, 2])
        
        with col_stock:
            training_stock = st.selectbox(
                "Select Stock for Training",
                options=["TSLA", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                index=0,
                help="Choose which stock to train the AI agent on"
            )
        
        with col_dates:
            st.caption("Training Date Range")
            col_start, col_end = st.columns(2)
            
            with col_start:
                train_start_date = st.date_input(
                    "Training Start",
                    value=datetime(2020, 1, 1),
                    min_value=datetime(2015, 1, 1),
                    max_value=datetime(2023, 12, 31),
                    help="Start date for training data"
                )
            
            with col_end:
                train_end_date = st.date_input(
                    "Training End",
                    value=datetime(2023, 12, 31),
                    min_value=datetime(2015, 1, 1),
                    max_value=datetime(2024, 12, 31),
                    help="End date for training data"
                )
        
        st.info(f"Training on {training_stock} stock from {train_start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    # Real-time training monitoring
    st.subheader("Training Progress Monitor")
    
    # Add explanation
    with st.expander("📖 Understanding the Training Metrics"):
        st.markdown("""
        **Chart Explanation:**
        - **Blue Line (Reward)**: The actual profit/loss the AI agent achieves at each step. 
          - For **real market data**: Rewards are typically small negative values (e.g., -0.05 to -5.0)
          - Negative values = agent is learning to minimize losses from hedging errors
          - Goal: Minimize hedging error (get closer to 0 or positive)
        
        - **Green Dashed Line (10-Step Average)**: Smoothed trend line showing overall learning progress
          - Rising trend = AI is improving 
          - Flat line = AI has plateaued (might need more training)
        
        **Metrics Explained:**
        - **Training Step**: Current progress (e.g., 50,000 / 200,000 steps completed)
        - **Current Reward**: Most recent normalized reward (NOT dollar P&L)
          - Typical range: -10 to +5 for well-trained agents
          - Values in millions indicate a problem (restart training)
        - **Total Improvement**: % change from starting performance to current
          - Positive % = getting better! Target: +20% to +80%
        - **Best Reward**: The best performance achieved during entire training session
        
        **What's Happening?**
        The agent learns to hedge derivative positions by minimizing:
        1. **P&L variance** (smooth profits/losses)
        2. **Delta tracking error** (follow Black-Scholes delta)
        3. **Transaction costs** (minimize unnecessary trading)
        
        **Expected Results:**
        - **Good training**: Rewards improve from -5 to -1 or better (+80% improvement)
        - **Bad training**: Rewards get worse or stay at -1000+ (reward scale broken)
        - **NVDA**: Often easiest to learn (consistent volatility 2020-2023)
        - **TSLA**: Harder due to wild price swings (Elon tweets, splits)
        - **AMZN**: Moderate difficulty (high price requires good normalization)
        """)
    
    # Placeholders for real-time updates
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Initialize session state for training process
    if 'training_process' not in st.session_state:
        st.session_state.training_process = None
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False
    
    # Show current training mode
    if "Real Market" in market_data_type:
        st.info("📊 Training Mode: REAL MARKET DATA (Yahoo Finance) - Uses actual Tesla/NVIDIA stock prices from 2020-2023")
    else:
        st.info("🎲 Training Mode: SYNTHETIC DATA (GBM) - Uses mathematical model for stock price simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("START TRAINING", type="primary", use_container_width=True, disabled=st.session_state.training_active):
            status_placeholder.info("Initializing training...")
            
            try:
                # Determine which script to run
                if "Real Market" in market_data_type:
                    script = "train_real_market.py"
                    cmd = [
                        sys.executable,
                        script,
                        "--timesteps", str(total_timesteps),
                        "--lr", str(learning_rate),
                        "--ticker", training_stock
                    ]
                else:
                    script = "train_agent.py"
                    cmd = [
                        sys.executable,
                        script,
                        "--timesteps", str(total_timesteps),
                        "--lr", str(learning_rate)
                    ]
                
                # Initialize monitor
                monitor = TrainingMonitor()
                
                # Debug: Show command being run
                with st.expander("🔍 Command Details"):
                    st.code(" ".join(cmd), language="bash")
                
                # Start training process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Combine stderr with stdout
                    text=True,
                    bufsize=1,
                    cwd=os.getcwd(),
                    universal_newlines=True
                )
                
                # Store in session state
                st.session_state.training_process = process
                st.session_state.training_active = True
                
                status_placeholder.success("Training started! Monitoring progress...")
                
                # Monitor training
                rewards_history = []
                timesteps_history = []
                start_time = time.time()
                last_step = 0
                console_output = []
                
                # Add console output placeholder
                console_placeholder = st.empty()
                
                while process.poll() is None:  # While process is running
                    # Read any new console output
                    if process.stdout:
                        line = process.stdout.readline()
                        if line:
                            console_output.append(line.strip())
                            # Keep only last 10 lines
                            if len(console_output) > 10:
                                console_output.pop(0)
                            
                            # Show console output
                            with console_placeholder.container():
                                with st.expander("📟 Training Console Output", expanded=False):
                                    st.code("\n".join(console_output[-5:]), language="text")
                    
                    # Get training status
                    status = monitor.get_status()
                    
                    if status:
                        current_step = status.get('current_timestep', 0)
                        total_steps = status.get('total_timesteps', total_timesteps)
                        mean_reward = status.get('mean_reward')
                        
                        # Check if we've reached target timesteps
                        if current_step >= total_timesteps:
                            status_placeholder.info(f"✅ Reached target timesteps ({total_timesteps:,}). Finalizing training...")
                            break
                        
                        # Check if training is stuck (no progress for 60 seconds)
                        if current_step == last_step and current_step > 0:
                            time_stuck = time.time() - start_time
                            if time_stuck > 60:
                                status_placeholder.warning(f"⚠️ Training appears stuck at step {current_step}. This might be normal during evaluation phases. Waiting...")
                        
                        # Calculate time estimates
                        elapsed_time = time.time() - start_time
                        
                        if total_steps > 0 and current_step > 0:
                            progress = min(current_step / total_steps, 1.0)  # Clamp to max 1.0
                            
                            # Estimate remaining time
                            time_per_step = elapsed_time / current_step
                            remaining_steps = max(0, total_steps - current_step)
                            estimated_remaining = time_per_step * remaining_steps
                            
                            # Calculate steps per second
                            if elapsed_time > 0:
                                steps_per_sec = current_step / elapsed_time
                            else:
                                steps_per_sec = 0
                            
                            # Format time
                            elapsed_str = f"{int(elapsed_time//60)}m {int(elapsed_time%60)}s"
                            remaining_str = f"{int(estimated_remaining//60)}m {int(estimated_remaining%60)}s"
                            
                            # Update progress with detailed info
                            progress_placeholder.progress(
                                progress,
                                text=f"⏱️ Progress: {current_step:,}/{total_steps:,} ({progress*100:.1f}%) | "
                                     f"Elapsed: {elapsed_str} | Remaining: ~{remaining_str} | "
                                     f"Speed: {steps_per_sec:.1f} steps/sec"
                            )
                        else:
                            progress_placeholder.info("🔄 Initializing training environment...")
                        
                        # Update reward history
                        if mean_reward is not None:
                            rewards_history.append(mean_reward)
                            timesteps_history.append(current_step)
                            
                            # Create learning visualization with dual charts
                            fig = go.Figure()
                            
                            # Main reward line
                            fig.add_trace(go.Scatter(
                                x=timesteps_history,
                                y=rewards_history,
                                mode='lines',
                                name='Reward',
                                line=dict(color='#4a9eff', width=3),
                                fill='tozeroy',
                                fillcolor='rgba(74, 158, 255, 0.2)'
                            ))
                            
                            # Add moving average if enough data
                            if len(rewards_history) >= 5:
                                window = min(10, len(rewards_history))
                                moving_avg = pd.Series(rewards_history).rolling(window=window).mean()
                                fig.add_trace(go.Scatter(
                                    x=timesteps_history,
                                    y=moving_avg,
                                    mode='lines',
                                    name=f'{window}-Step Avg',
                                    line=dict(color='#00ff00', width=2, dash='dash')
                                ))
                            
                            fig.update_layout(
                                template='plotly_dark',
                                title='🧠 AI Learning Progress - How Agent Improves Over Time',
                                xaxis_title='Training Steps',
                                yaxis_title='Average Reward ($)',
                                height=400,
                                hovermode='x unified',
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            chart_placeholder.plotly_chart(fig, use_container_width=True)
                            
                            # Update metrics with better formatting
                            col_m1, col_m2, col_m3, col_m4 = metrics_placeholder.columns(4)
                            
                            with col_m1:
                                st.metric(
                                    "Training Step", 
                                    f"{current_step:,}",
                                    delta=f"+{current_step - last_step}" if current_step > last_step else None
                                )
                            
                            with col_m2:
                                # Display reward in reasonable format
                                if mean_reward is not None:
                                    if abs(mean_reward) < 1.0:
                                        reward_display = f"{mean_reward:.4f}"
                                    elif abs(mean_reward) < 1000:
                                        reward_display = f"${mean_reward:.2f}"
                                    else:
                                        reward_display = f"${mean_reward/1000:.1f}K"
                                    
                                    st.metric(
                                        "Current Reward", 
                                        reward_display,
                                        delta=f"{mean_reward - rewards_history[-2]:.2f}" if len(rewards_history) > 1 else None
                                    )
                                else:
                                    st.metric("Current Reward", "N/A")
                            
                            with col_m3:
                                if len(rewards_history) > 1:
                                    improvement = ((rewards_history[-1] - rewards_history[0]) / abs(rewards_history[0]) * 100) if rewards_history[0] != 0 else 0
                                    st.metric("Total Improvement", f"{improvement:+.1f}%")
                                else:
                                    st.metric("Total Improvement", "0.0%")
                            
                            with col_m4:
                                best_reward = max(rewards_history)
                                st.metric("Best Reward", f"${best_reward:.2f}")
                            
                            last_step = current_step
                    
                    time.sleep(2)  # Update every 2 seconds
                
                # Training complete - wait for process to finish
                status_placeholder.info("⏳ Training complete! Saving models and finalizing...")
                
                try:
                    # Give it up to 30 seconds to save models and clean up
                    stdout, stderr = process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    # If still running after 30 seconds, force kill
                    process.kill()
                    stdout, stderr = process.communicate()
                
                # Mark training as inactive
                st.session_state.training_active = False
                st.session_state.training_process = None
                
                if process.returncode == 0:
                    status_placeholder.success("✅ Training completed successfully!")
                    st.balloons()
                else:
                    status_placeholder.error(f"❌ Training failed with exit code: {process.returncode}")
                    with st.expander("Show error details"):
                        if stderr:
                            st.error(stderr)
                        if stdout:
                            st.info("STDOUT:")
                            st.code(stdout)
            
            except Exception as e:
                st.session_state.training_active = False
                st.session_state.training_process = None
                status_placeholder.error(f"Error starting training: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    with col2:
        if st.button("STOP TRAINING", type="secondary", use_container_width=True, disabled=not st.session_state.training_active):
            if st.session_state.training_process is not None:
                try:
                    # Terminate the training process
                    st.session_state.training_process.terminate()
                    time.sleep(1)
                    
                    # Force kill if still running
                    if st.session_state.training_process.poll() is None:
                        st.session_state.training_process.kill()
                    
                    st.session_state.training_active = False
                    st.session_state.training_process = None
                    status_placeholder.warning("⚠️ Training stopped by user")
                except Exception as e:
                    st.error(f"Error stopping training: {str(e)}")
            else:
                st.info("No active training to stop")
    
    st.markdown("---")
    
    # Man vs Machine Challenge
    st.subheader("AI vs Human Challenge")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        **Test yourself against the trained AI agent!**
        
        - Use LEFT/RIGHT arrow keys to adjust your hedge position
        - Green line = AI performance
        - Red line = Your performance
        - Try to minimize losses and beat the machine
        """)
    
    with col2:
        if st.button("Launch Man vs Machine", type="primary", use_container_width=True):
            try:
                import os
                
                # Check if user selected a model from sidebar
                selected_model = st.session_state.get('selected_game_model', None)
                
                # If no model selected, check default locations
                if not selected_model or not os.path.exists(selected_model):
                    model_paths = [
                        "./saved_models/ppo_hedging_agent.zip",
                        "./saved_models_real/ppo_NVDA_real.zip",
                        "./saved_models_real/ppo_TSLA_real.zip",
                        "./saved_models_real/ppo_AAPL_real.zip"
                    ]
                    
                    for path in model_paths:
                        if os.path.exists(path):
                            selected_model = path
                            break
                
                if selected_model and os.path.exists(selected_model):
                    # Pass the selected model path to game launcher
                    env = os.environ.copy()
                    env['GAME_MODEL_PATH'] = selected_model
                    
                    subprocess.Popen([sys.executable, "game_launcher.py"], env=env)
                    
                    model_name = os.path.basename(selected_model)
                    st.success(f"Game launched with model: {model_name}")
                    st.info("Controls: LEFT/RIGHT arrows to adjust hedge")
                else:
                    st.error("No trained model found. Train an agent first or select one from the sidebar!")
            except Exception as e:
                st.error(f"Failed to launch game: {str(e)}")
    
    st.markdown("---")
    st.info("""
    **Training Information:**
    - Standard (GBM): Uses synthetic Geometric Brownian Motion data
    - Real Market Data: Uses historical Yahoo Finance data (2020-2024)
    - Training typically takes 2-5 minutes depending on timesteps
    - Progress updates every 2,048 steps
    - Model saved to: `./saved_models/ppo_hedging_agent.zip`
    """)

# ==================== TAB 3: RISK MANAGER ====================
with tab3:
    st.header("Quantitative Risk Management")
    st.markdown("""
    Visualize the mathematical 'physics' driving Option prices using the **Nobel Prize-winning Black-Scholes formula**.
    
    The "Greeks" are derivatives (calculus) that measure different types of risk:
    - **Delta (Δ):** The Speed - How much does option price change per $1 stock move?
    - **Gamma (Γ):** The Acceleration - How fast is Delta changing? High Gamma = Crash Risk!
    - **Theta (Θ):** The Melting Ice - Daily value loss due to time decay
    - **Vega (ν):** The Fear - Sensitivity to volatility changes
    """)
    
    # --- 1. INPUTS (The Controls) ---
    st.markdown("### Option Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        S = st.slider("Spot Price ($)", 50, 150, 100, help="Current stock price")
    with col2:
        sigma = st.slider("Volatility (%)", 10, 100, 20, help="Historical volatility (annualized)") / 100.0
    with col3:
        T_days = st.slider("Time to Expiry (Days)", 1, 365, 90, help="Time until option expires")
        T = T_days / 365.0  # Convert to years for Black-Scholes
    with col4:
        r = st.slider("Risk-Free Rate (%)", 0, 10, 5, help="Treasury bond rate") / 100.0
    
    K = 100  # Strike Price is fixed for this demo
    st.caption(f"Strike Price: ${K} (fixed) | Time to Expiry: {T_days} days ({T_days/30:.1f} months)")
    
    # --- 2. MATH (The Black-Scholes Engine) ---
    try:
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate Call Option Price
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        # Calculate Greeks
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        # Display Option Price
        st.markdown("### Option Valuation")
        st.metric(
            "Call Option Price", 
            f"${call_price:.2f}",
            help="Fair value calculated using Black-Scholes formula"
        )
        
        # --- 3. METRICS (The Dashboard) ---
        st.markdown("### Real-Time Greeks")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric(
            "Delta (Δ)", 
            f"{delta:.4f}",
            help="Hedge Ratio: If stock moves $1, option moves ~$" + f"{delta:.2f}"
        )
        m2.metric(
            "Gamma (Γ)", 
            f"{gamma:.4f}",
            help="Delta acceleration. High Gamma = High rebalancing cost!"
        )
        m3.metric(
            "Theta (Θ)", 
            f"${theta/365:.2f}/day",
            help="Daily time decay. Option loses this much value per day."
        )
        m4.metric(
            "Vega (ν)", 
            f"${vega/100:.2f} per 1%",
            help="Volatility sensitivity. If vol increases 1%, option gains this much."
        )
        
        # Interpretation Guide
        st.markdown("---")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Current Risk Profile:**")
            
            if delta < 0.3:
                st.info("**Low Delta** - Option is Out-of-the-Money. Low hedging needed.")
            elif delta > 0.7:
                st.warning("**High Delta** - Option is In-the-Money. High hedging needed.")
            else:
                st.success("**Medium Delta** - Option is At-the-Money. Moderate hedging.")
            
            if gamma > 0.05:
                st.error("**High Gamma Risk!** - Delta is changing rapidly. Expensive to hedge.")
            else:
                st.success("**Low Gamma** - Delta is stable. Cheaper to maintain hedge.")
        
        with col_b:
            st.markdown("**AI Agent Strategy:**")
            st.write(f"""
            - **Target Hedge:** {delta:.1%} of stock position
            - **Rebalancing Frequency:** Every time step
            - **Gamma Exposure:** {'High' if gamma > 0.05 else 'Moderate'}
            - **Time Decay:** ${abs(theta/365):.2f} lost per day
            """)
        
        # --- 4. VISUALIZATION (The 3D Surface) ---
        st.markdown("---")
        st.markdown("### Delta Surface Topology")
        st.caption("Visualizing how Hedge Ratio (Delta) changes across Stock Price and Time dimensions.")
        
        # Generate Meshgrid data
        price_range = np.linspace(max(50, S-30), min(150, S+30), 40)
        time_range_days = np.linspace(1, 365, 40)  # Days from 1 to 365
        time_range_years = time_range_days / 365.0  # Convert to years for calculation
        X, Y_days = np.meshgrid(price_range, time_range_days)
        _, Y_years = np.meshgrid(price_range, time_range_years)
        
        # Vectorized Calculation for the Surface (using years)
        Z_d1 = (np.log(X / K) + (r + 0.5 * sigma ** 2) * Y_years) / (sigma * np.sqrt(Y_years))
        Z_delta = norm.cdf(Z_d1)
        
        # Create Plotly 3D Surface (display in days)
        fig = go.Figure(data=[go.Surface(
            z=Z_delta, 
            x=X, 
            y=Y_days,  # Use days for display
            colorscale='Viridis',
            colorbar=dict(title="Delta"),
            hovertemplate='<b>Price:</b> $%{x:.2f}<br>' +
                          '<b>Time:</b> %{y:.0f} days<br>' +
                          '<b>Delta:</b> %{z:.3f}<extra></extra>'
        )])
        
        # Mark current position (in days)
        fig.add_trace(go.Scatter3d(
            x=[S],
            y=[T_days],  # Use days for display
            z=[delta],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Current Position',
            hovertemplate='<b>Current State</b><br>' +
                          f'Price: ${S}<br>' +
                          f'Time: {T_days} days<br>' +
                          f'Delta: {delta:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Delta Surface - The Hedging Landscape', 
            scene = dict(
                xaxis_title='Stock Price ($)',
                yaxis_title='Time to Expiry (Days)',
                zaxis_title='Delta (Hedge Ratio)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                ),
                bgcolor='rgba(10, 10, 20, 0.9)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width=900, 
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
            font=dict(color='#e0e0e0')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Educational Note
        st.markdown("---")
        st.info("""
        **Understanding the Surface:**
        
        - **Steep cliffs** = High Gamma regions (dangerous for hedgers)
        - **Flat plateaus** = Low Gamma regions (easy to hedge)
        - **Red diamond** = Your current market position
        - **As time decreases (Y→0):** Delta approaches 0 or 1 (binary outcome)
        - **At-the-money (X≈$100):** Maximum Gamma risk when time is short
        
        Professional traders use this visualization to:
        1. Identify when rebalancing costs will be highest
        2. Predict how their hedge ratio will evolve
        3. Avoid "Gamma squeeze" scenarios that can cause massive losses
        """)
        
    except Exception as e:
        st.error(f"Calculation error: {str(e)}")
        st.info("Try adjusting the parameters - very short time to expiry can cause numerical issues.")

# ==================== TAB 4: BACKTEST LABORATORY ====================
with tab4:
    st.header("Backtest Laboratory")
    st.markdown("""
    Test hedging strategies against **historical market events**. Compare unhedged positions vs AI-hedged positions 
    to see how well the hedging strategy would have performed during extreme market conditions.
    """)
    
    # Scenario definitions
    scenarios = {
        "COVID Crash (Feb-Apr 2020)": {
            "ticker": "TSLA",
            "start": "2020-02-01",
            "end": "2020-04-30",
            "description": "Market panic, extreme volatility, S&P 500 dropped 34%"
        },
        "Tech Bear Market (2022)": {
            "ticker": "NVDA",
            "start": "2022-01-01",
            "end": "2022-12-31",
            "description": "Fed rate hikes, tech selloff, NVDA dropped 50%"
        },
        "2023 Bull Run": {
            "ticker": "TSLA",
            "start": "2023-01-01",
            "end": "2023-06-30",
            "description": "AI hype, strong rally, TSLA up 100%+"
        },
        "NVDA AI Boom (2023-2024)": {
            "ticker": "NVDA",
            "start": "2023-06-01",
            "end": "2024-03-31",
            "description": "NVDA tripled on AI chip demand"
        }
    }
    
    # User controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_scenario = st.selectbox(
            "Select Historical Scenario",
            options=list(scenarios.keys()),
            help="Choose a historical market event to backtest"
        )
    
    with col2:
        st.metric("Ticker", scenarios[selected_scenario]["ticker"])
    
    # Display scenario info
    st.info(f"""
    **Scenario:** {selected_scenario}  
    **Period:** {scenarios[selected_scenario]['start']} to {scenarios[selected_scenario]['end']}  
    **Event:** {scenarios[selected_scenario]['description']}
    """)
    
    # Run backtest button
    if st.button("Run Backtest", type="primary", use_container_width=True):
        
        scenario = scenarios[selected_scenario]
        
        with st.spinner(f"Downloading {scenario['ticker']} data and running simulation..."):
            try:
                # 1. Try to load trained PPO model
                ticker = scenario['ticker']
                model = None
                vec_normalize = None
                model_source = "Simulation (No trained model found)"
                
                # Check for trained models
                model_paths = [
                    (f"./saved_models_real/ppo_{ticker}_real.zip", f"./saved_models_real/vec_normalize_{ticker}.pkl"),
                    ("./saved_models/ppo_hedging_agent.zip", "./saved_models/vec_normalize.pkl")
                ]
                
                for model_path, norm_path in model_paths:
                    if os.path.exists(model_path):
                        try:
                            from stable_baselines3 import PPO
                            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                            from envs.derivative_hedging_env import DerivativeHedgingEnv
                            
                            model = PPO.load(model_path)
                            model_source = f"Trained PPO Model: {os.path.basename(model_path)}"
                            
                            # Load normalizer if available
                            if os.path.exists(norm_path):
                                # We'll create a temp env for normalization
                                temp_env = DummyVecEnv([lambda: DerivativeHedgingEnv()])
                                vec_normalize = VecNormalize.load(norm_path, temp_env)
                                vec_normalize.training = False
                                vec_normalize.norm_reward = False
                            
                            st.info(f"✅ Using trained model: `{os.path.basename(model_path)}`")
                            break
                        except Exception as e:
                            st.warning(f"Failed to load model {model_path}: {str(e)}")
                            model = None
                            continue
                
                if model is None:
                    st.info("ℹ️ No trained model found. Using simplified Black-Scholes delta hedging simulation.")
                
                # 2. Download historical data
                data = yf.download(
                    scenario['ticker'], 
                    start=scenario['start'], 
                    end=scenario['end'],
                    progress=False
                )
                
                if data.empty:
                    st.error("No data available for this period. Try another scenario.")
                else:
                    # Handle multi-index columns from yfinance
                    if isinstance(data.columns, pd.MultiIndex):
                        prices = data['Close'].iloc[:, 0].values
                    else:
                        prices = data['Close'].values
                    
                    # Ensure 1D array
                    prices = np.ravel(prices)
                    
                    returns = np.diff(prices) / prices[:-1]
                    
                    # 3. Setup for backtest
                    initial_price = float(prices[0])
                    strike = initial_price  # ATM option
                    r = 0.05  # Risk-free rate
                    sigma = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.3  # Historical volatility
                    sigma = max(sigma, 0.15)  # Minimum 15% vol
                    
                    # Calculate total time to expiry (in years)
                    total_days = len(prices)
                    T_total = total_days / 252.0
                    
                    # Helper function: Black-Scholes call option price
                    def bs_call_price(S, K, T, r, sigma):
                        if T <= 0:
                            return max(S - K, 0)  # Intrinsic value at expiry
                        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                        d2 = d1 - sigma * np.sqrt(T)
                        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                        return call_price
                    
                    # Calculate initial premium using Black-Scholes
                    premium = bs_call_price(initial_price, strike, T_total, r, sigma)
                    
                    # 4. Unhedged P&L: Mark-to-market the short call option
                    unhedged_pnl = np.zeros(len(prices))
                    
                    for i in range(len(prices)):
                        # Time remaining until expiry
                        days_remaining = total_days - i
                        T_remaining = days_remaining / 252.0
                        
                        # Current option value
                        current_option_value = bs_call_price(prices[i], strike, T_remaining, r, sigma)
                        
                        # P&L = Premium collected - Current option liability
                        unhedged_pnl[i] = premium - current_option_value
                    
                    # 5. Delta Hedging Baseline (Perfect Black-Scholes)
                    delta_pnl = np.zeros(len(prices))
                    delta_hedge_position = 0.0
                    delta_cash = premium
                    
                    for i in range(len(prices)):
                        days_remaining = total_days - i
                        T_remaining = days_remaining / 252.0
                        current_option_value = bs_call_price(prices[i], strike, T_remaining, r, sigma)
                        
                        # Calculate delta and hedge with it
                        if T_remaining > 0:
                            d1 = (np.log(prices[i] / strike) + (r + 0.5 * sigma**2) * T_remaining) / (sigma * np.sqrt(T_remaining))
                            option_delta = norm.cdf(d1)
                        else:
                            option_delta = 1.0 if prices[i] > strike else 0.0
                        
                        # Perfect delta hedge (100% of delta)
                        target_delta_shares = option_delta
                        
                        if i < len(prices) - 1:
                            hedge_adjustment = target_delta_shares - delta_hedge_position
                            transaction_cost = abs(hedge_adjustment) * 0.0005 * prices[i]
                            delta_cash -= hedge_adjustment * prices[i] + transaction_cost
                            delta_hedge_position = target_delta_shares
                        
                        # Portfolio value
                        portfolio_value = delta_cash + delta_hedge_position * prices[i] - current_option_value
                        delta_pnl[i] = portfolio_value
                    
                    # 6. AI Agent P&L: Use real model or simulation
                    ai_pnl = np.zeros(len(prices))
                    hedge_position = 0.0  # Current stock hedge position (number of shares)
                    cash = premium  # Start with premium in cash
                    
                    if model is not None:
                        # USE REAL TRAINED MODEL
                        st.info("🤖 Running backtest with trained PPO agent...")
                        
                        # Create environment for each timestep
                        from envs.derivative_hedging_env import DerivativeHedgingEnv
                        from stable_baselines3.common.vec_env import DummyVecEnv
                        
                        for i in range(len(prices)):
                            # Calculate current option value
                            days_remaining = total_days - i
                            T_remaining = days_remaining / 252.0
                            current_option_value = bs_call_price(prices[i], strike, T_remaining, r, sigma)
                            
                            # Get AI's hedge decision
                            if i < len(prices) - 1:  # Don't need to hedge on last day
                                # Create env with current market conditions
                                temp_env = DerivativeHedgingEnv(
                                    S0=float(prices[i]),
                                    K=strike,
                                    T=max(T_remaining, 1/252),
                                    sigma=sigma,
                                    r=r
                                )
                                
                                vec_env = DummyVecEnv([lambda: temp_env])
                                
                                if vec_normalize is not None:
                                    obs = vec_normalize.normalize_obs(vec_env.reset())
                                else:
                                    obs = vec_env.reset()
                                
                                action, _ = model.predict(obs, deterministic=True)
                                
                                # Transform action same way as environment does
                                # Maps [-2, 2] to [0, 1] hedge ratio
                                hedge_ratio = (float(action[0]) / 2.0) + 0.5
                                hedge_ratio = np.clip(hedge_ratio, 0.0, 1.0)
                                
                                # Calculate option delta
                                if T_remaining > 0:
                                    d1 = (np.log(prices[i] / strike) + (r + 0.5 * sigma**2) * T_remaining) / (sigma * np.sqrt(T_remaining))
                                    option_delta = norm.cdf(d1)
                                else:
                                    option_delta = 1.0 if prices[i] > strike else 0.0
                                
                                # Target hedge = hedge_ratio * delta (in number of shares)
                                target_hedge_shares = hedge_ratio * option_delta
                                
                                # Rebalance hedge
                                hedge_adjustment = target_hedge_shares - hedge_position
                                transaction_cost = abs(hedge_adjustment) * 0.0005 * prices[i]  # 5 bps
                                cash -= hedge_adjustment * prices[i] + transaction_cost
                                hedge_position = target_hedge_shares
                            
                            # Calculate total portfolio value
                            portfolio_value = cash + hedge_position * prices[i] - current_option_value
                            ai_pnl[i] = portfolio_value
                    
                    else:
                        # USE SIMPLIFIED SIMULATION (fallback)
                        st.info("📊 Running backtest with Black-Scholes delta hedging simulation...")
                        
                        for i in range(len(prices)):
                            # Calculate current option value and delta
                            days_remaining = total_days - i
                            T_remaining = days_remaining / 252.0
                            current_option_value = bs_call_price(prices[i], strike, T_remaining, r, sigma)
                            
                            # Calculate option delta for hedge target
                            if T_remaining > 0:
                                d1 = (np.log(prices[i] / strike) + (r + 0.5 * sigma**2) * T_remaining) / (sigma * np.sqrt(T_remaining))
                                option_delta = norm.cdf(d1)
                            else:
                                option_delta = 1.0 if prices[i] > strike else 0.0
                            
                            # Perfect delta hedge: hedge_ratio = 1.0
                            # Target hedge in shares = 1.0 * delta
                            target_hedge_shares = 1.0 * option_delta
                            
                            # Rebalance hedge
                            if i < len(prices) - 1:
                                hedge_adjustment = target_hedge_shares - hedge_position
                                transaction_cost = abs(hedge_adjustment) * 0.0005 * prices[i]
                                cash -= hedge_adjustment * prices[i] + transaction_cost
                                hedge_position = target_hedge_shares
                            
                            # Calculate total portfolio value
                            portfolio_value = cash + hedge_position * prices[i] - current_option_value
                            ai_pnl[i] = portfolio_value
                    
                    # Ensure all arrays have same length
                    n_points = min(len(data.index), len(prices), len(unhedged_pnl), len(delta_pnl), len(ai_pnl))
                    
                    # Create comparison dataframe
                    results_df = pd.DataFrame({
                        'Date': data.index[:n_points],
                        'Stock Price': prices[:n_points],
                        'Unhedged P&L': unhedged_pnl[:n_points],
                        'Delta Hedge P&L': delta_pnl[:n_points],
                        'AI Hedged P&L': ai_pnl[:n_points]
                    })
                    
                    # Calculate statistics
                    unhedged_returns = np.diff(unhedged_pnl)
                    delta_returns = np.diff(delta_pnl)
                    ai_returns = np.diff(ai_pnl)
                    
                    unhedged_max_dd = (unhedged_pnl.min() - unhedged_pnl[0]) if len(unhedged_pnl) > 0 else 0
                    delta_max_dd = (delta_pnl.min() - delta_pnl[0]) if len(delta_pnl) > 0 else 0
                    ai_max_dd = (ai_pnl.min() - ai_pnl[0]) if len(ai_pnl) > 0 else 0
                    
                    unhedged_vol = np.std(unhedged_returns) * np.sqrt(252) if len(unhedged_returns) > 0 else 0
                    delta_vol = np.std(delta_returns) * np.sqrt(252) if len(delta_returns) > 0 else 0
                    ai_vol = np.std(ai_returns) * np.sqrt(252) if len(ai_returns) > 0 else 0
                    
                    # Display results
                    st.success("Backtest completed!")
                    
                    # Plot P&L comparison
                    st.markdown("### P&L Comparison Over Time")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=results_df['Date'],
                        y=results_df['Unhedged P&L'],
                        mode='lines',
                        name='Unhedged Strategy',
                        line=dict(color='#ff5555', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=results_df['Date'],
                        y=results_df['Delta Hedge P&L'],
                        mode='lines',
                        name='Delta Hedge (Baseline)',
                        line=dict(color='#8be9fd', width=2, dash='dot')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=results_df['Date'],
                        y=results_df['AI Hedged P&L'],
                        mode='lines',
                        name='AI Hedged Strategy (RL)',
                        line=dict(color='#50fa7b', width=3)
                    ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Cumulative P&L ($)",
                        hovermode='x unified',
                        template='plotly_dark',
                        height=500,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show model source
                    st.caption(f"**Strategy Source:** {model_source}")
                    
                    # Statistics comparison
                    st.markdown("### Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Final P&L (Unhedged)",
                            f"${unhedged_pnl[-1]:.2f}",
                            delta=f"{unhedged_pnl[-1]:.2f}",
                            delta_color="inverse"
                        )
                    
                    with col2:
                        st.metric(
                            "Final P&L (Delta Hedge)",
                            f"${delta_pnl[-1]:.2f}",
                            delta=f"{delta_pnl[-1]:.2f}",
                            delta_color="normal"
                        )
                    
                    with col3:
                        st.metric(
                            "Final P&L (AI RL)",
                            f"${ai_pnl[-1]:.2f}",
                            delta=f"{ai_pnl[-1]:.2f}",
                            delta_color="normal"
                        )
                    
                    with col4:
                        improvement_vs_delta = ((ai_pnl[-1] - delta_pnl[-1]) / abs(delta_pnl[-1]) * 100) if delta_pnl[-1] != 0 else 0
                        st.metric(
                            "AI vs Delta Hedge",
                            f"{improvement_vs_delta:+.1f}%",
                            delta=f"${ai_pnl[-1] - delta_pnl[-1]:.2f}"
                        )
                    
                    # Detailed stats table
                    st.markdown("### Risk Metrics Comparison")
                    
                    stats_df = pd.DataFrame({
                        'Metric': ['Max Drawdown', 'Volatility (Annualized)', 'Final P&L', 'Sharpe Ratio (approx)'],
                        'Unhedged': [
                            f"${unhedged_max_dd:.2f}",
                            f"{unhedged_vol:.2%}",
                            f"${unhedged_pnl[-1]:.2f}",
                            f"{(unhedged_pnl[-1] / (unhedged_vol * abs(unhedged_pnl[0]) + 1e-8)):.2f}"
                        ],
                        'Delta Hedge (Baseline)': [
                            f"${delta_max_dd:.2f}",
                            f"{delta_vol:.2%}",
                            f"${delta_pnl[-1]:.2f}",
                            f"{(delta_pnl[-1] / (delta_vol * abs(delta_pnl[0]) + 1e-8)):.2f}"
                        ],
                        'AI RL Agent': [
                            f"${ai_max_dd:.2f}",
                            f"{ai_vol:.2%}",
                            f"${ai_pnl[-1]:.2f}",
                            f"{(ai_pnl[-1] / (ai_vol * abs(ai_pnl[0]) + 1e-8)):.2f}"
                        ]
                    })
                    
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("### Interpretation")
                    
                    if ai_pnl[-1] > unhedged_pnl[-1]:
                        st.success(f"""
                        **AI Hedging Strategy Wins!**  
                        The AI-hedged position outperformed by ${ai_pnl[-1] - unhedged_pnl[-1]:.2f}.
                        Lower volatility ({ai_vol:.1%} vs {unhedged_vol:.1%}) indicates more stable performance.
                        """)
                    else:
                        st.warning(f"""
                        **Unhedged Strategy Performed Better**  
                        In this scenario, hedging costs outweighed the protection benefits.
                        However, the AI strategy had {((unhedged_vol - ai_vol) / unhedged_vol * 100):.1f}% lower volatility.
                        """)
                    
                    if model is not None:
                        st.info(f"""
                        **Real Model Used:** This backtest used your actual trained PPO agent from `{model_source}`.
                        The hedge decisions were made by the neural network based on real-time market observations.
                        
                        **Model Performance:** The trained agent demonstrated {'superior' if ai_pnl[-1] > unhedged_pnl[-1] else 'controlled'} 
                        risk management compared to an unhedged position.
                        """)
                    else:
                        st.info("""
                        **Simulation Mode:** This backtest used simplified Black-Scholes delta hedging as no trained model was found.
                        
                        **To use real AI:**
                        1. Go to the "Neural Training" tab
                        2. Train an agent on real market data (select ticker matching this scenario)
                        3. Re-run this backtest to see your trained model in action
                        
                        **Real-world considerations:**
                        - Transaction costs and slippage
                        - Bid-ask spreads
                        - Gamma rebalancing costs
                        - Interest rate effects
                        - Implied volatility changes
                        """)
                    
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
                st.info("Try selecting a different scenario or check your internet connection.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9ca3af; padding: 1rem;'>
    <p>Derivative Hedging with Deep Reinforcement Learning</p>
    <p>Powered by PPO Algorithm | Real-time Market Data</p>
</div>
""", unsafe_allow_html=True)