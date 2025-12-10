"""
FastAPI Backend for Derivative Hedging RL Dashboard
Mirrors the functionality of the Streamlit app.py
"""

import os
import sys
import glob
import subprocess
import numpy as np
import pandas as pd
import scipy.stats as si
import yfinance as yf
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Derivative Hedging RL API",
    description="Backend for AI-powered derivative hedging dashboard",
    version="2.0.0"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== DATA MODELS =====================

class GreeksRequest(BaseModel):
    spot: float
    strike: float = 100.0
    time: float  # Time to maturity in days
    rate: float  # Risk-free rate (percentage, e.g., 5 for 5%)
    vol: float   # Volatility (percentage, e.g., 20 for 20%)
    option_type: str = "call"

class TrainRequest(BaseModel):
    timesteps: int = 50000
    learning_rate: float = 0.0003
    ticker: str = "TSLA"
    data_type: str = "real"  # "real" or "synthetic"
    algorithm: str = "PPO"  # "PPO" or "SAC"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    strike: Optional[float] = None
    model_path: Optional[str] = None  # Path to trained RL model
    algorithm: Optional[str] = "PPO"  # PPO or SAC
    rebalance_freq: str = "daily"  # daily, weekly, threshold

class MarketDataRequest(BaseModel):
    ticker: str
    period: str = "1y"  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

# ===================== HELPER FUNCTIONS =====================

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict[str, float]:
    """
    Calculate Black-Scholes option price and Greeks.
    T is in years, r and sigma are in decimal form (e.g., 0.05 for 5%)
    """
    # Avoid division by zero
    if T <= 0 or sigma <= 0:
        intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
        return {
            "price": intrinsic,
            "delta": 1.0 if S > K and option_type == "call" else 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        delta = si.norm.cdf(d1)
        theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        delta = si.norm.cdf(d1) - 1
        theta = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2)

    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * si.norm.pdf(d1)
    
    return {
        "price": round(float(price), 4),
        "delta": round(float(delta), 4),
        "gamma": round(float(gamma), 6),
        "theta": round(float(theta / 365.0), 4),  # Daily theta
        "vega": round(float(vega / 100.0), 4),    # Per 1% vol change
        "rho": round(float(rho / 100.0), 4)       # Per 1% rate change
    }

def generate_delta_surface(K: float, r: float, sigma: float, spot_range: tuple = (70, 130), time_range: tuple = (1, 365)) -> Dict:
    """Generate 3D surface data for Delta visualization."""
    price_points = np.linspace(spot_range[0], spot_range[1], 30)
    time_points = np.linspace(time_range[0], time_range[1], 30)
    
    X, Y = np.meshgrid(price_points, time_points)
    Z = np.zeros_like(X)
    
    for i in range(len(time_points)):
        for j in range(len(price_points)):
            T_years = time_points[i] / 365.0
            if T_years > 0:
                d1 = (np.log(price_points[j] / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
                Z[i, j] = si.norm.cdf(d1)
            else:
                Z[i, j] = 1.0 if price_points[j] > K else 0.0
    
    return {
        "x": X.tolist(),  # Spot prices
        "y": Y.tolist(),  # Time to expiry (days)
        "z": Z.tolist()   # Delta values
    }

# ===================== API ENDPOINTS =====================

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Derivative Hedging RL Backend",
        "version": "2.0.0",
        "endpoints": ["/market-data", "/calculate-greeks", "/models", "/train", "/launch-sim", "/backtest"]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ==================== MARKET DATA ====================

@app.get("/market-data/{ticker}")
def get_market_data(ticker: str, period: str = "1y", start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Fetches historical OHLCV data for charting."""
    try:
        stock = yf.Ticker(ticker)
        
        # Use date range if provided, otherwise use period
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date)
        else:
            df = stock.history(period=period)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")
        
        # Calculate metrics
        current_price = float(df['Close'].iloc[-1])
        daily_returns = df['Close'].pct_change().dropna()
        daily_vol = float(daily_returns.std())
        annual_vol = daily_vol * np.sqrt(252)
        min_price = float(df['Close'].min())
        max_price = float(df['Close'].max())
        
        # Format for frontend
        data = []
        for date, row in df.iterrows():
            data.append({
                "time": date.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"])
            })
        
        return {
            "ticker": ticker,
            "data": data,
            "metrics": {
                "currentPrice": round(current_price, 2),
                "dailyVol": round(daily_vol * 100, 2),
                "annualVol": round(annual_vol * 100, 2),
                "minPrice": round(min_price, 2),
                "maxPrice": round(max_price, 2),
                "priceRange": f"${min_price:.2f} - ${max_price:.2f}"
            },
            "returns": (daily_returns * 100).tolist()[-100:]  # Last 100 returns for histogram
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== GREEKS CALCULATOR ====================

@app.post("/calculate-greeks")
def calculate_greeks(req: GreeksRequest):
    """Calculates Black-Scholes option price and Greeks."""
    try:
        # Convert percentage inputs to decimals
        T_years = req.time / 365.0  # Days to years
        r_decimal = req.rate / 100.0
        sigma_decimal = req.vol / 100.0
        
        greeks = black_scholes(
            S=req.spot,
            K=req.strike,
            T=T_years,
            r=r_decimal,
            sigma=sigma_decimal,
            option_type=req.option_type
        )
        
        # Add interpretation
        interpretation = {
            "deltaRisk": "Low" if greeks["delta"] < 0.3 else "High" if greeks["delta"] > 0.7 else "Medium",
            "gammaRisk": "High" if greeks["gamma"] > 0.05 else "Low",
            "hedgeRatio": f"{greeks['delta']:.1%}",
            "dailyDecay": f"${abs(greeks['theta']):.2f}"
        }
        
        return {**greeks, "interpretation": interpretation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/delta-surface")
def get_delta_surface(strike: float = 100, rate: float = 5, vol: float = 20):
    """Generate 3D Delta surface data for visualization."""
    try:
        surface = generate_delta_surface(
            K=strike,
            r=rate / 100.0,
            sigma=vol / 100.0
        )
        return surface
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MODEL MANAGEMENT ====================

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/models")
def list_models():
    """Scans saved_models directories and returns model metadata."""
    models = []
    
    # Define model folders to scan (use absolute paths)
    folders = [
        (os.path.join(PROJECT_ROOT, "saved_models"), "synthetic"),
        (os.path.join(PROJECT_ROOT, "saved_models_real"), "real")
    ]
    
    for folder, model_type in folders:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "*.zip"))
            for f in files:
                try:
                    file_size_mb = os.path.getsize(f) / (1024 * 1024)
                    # Use modification time (more reliable than creation time)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(f))
                    
                    # Extract ticker from filename (e.g., ppo_TSLA_real.zip -> TSLA)
                    basename = os.path.basename(f)
                    parts = basename.replace(".zip", "").split("_")
                    ticker = parts[1] if len(parts) > 1 else "Unknown"
                    
                    models.append({
                        "name": basename,
                        "type": model_type,
                        "path": f.replace("\\", "/"),  # Normalize path for frontend
                        "sizeMB": round(file_size_mb, 2),
                        "created": mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                        "ticker": ticker
                    })
                except Exception as e:
                    print(f"Error reading model {f}: {e}")
                    continue
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: x['created'], reverse=True)
    
    return {"models": models, "count": len(models)}

@app.delete("/models/{model_name:path}")
def delete_model(model_name: str):
    """Delete a model and its associated normalizer file."""
    from urllib.parse import unquote
    model_name = unquote(model_name)  # Decode URL-encoded characters
    
    try:
        # Search in both folders using absolute paths
        folders = [
            os.path.join(PROJECT_ROOT, "saved_models"),
            os.path.join(PROJECT_ROOT, "saved_models_real")
        ]
        
        for folder in folders:
            model_path = os.path.join(folder, model_name)
            print(f"[DEBUG] Checking model path: {model_path}")
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"[DEBUG] Deleted model: {model_path}")
                
                # Try to delete normalizer - handle both ppo_ and sac_ prefixes
                base_name = model_name.replace(".zip", "")
                # Extract ticker from model name (e.g., "sac_NVDA_real" -> "NVDA")
                # Or checkpoint names like "sac_NVDA_checkpoint_50000_steps" -> "NVDA"
                parts = base_name.split("_")
                if len(parts) >= 2:
                    algo = parts[0]  # ppo or sac
                    ticker = parts[1]  # NVDA, TSLA, etc.
                    # Try different normalizer naming conventions
                    norm_files = [
                        os.path.join(folder, f"vec_normalize_{algo}_{ticker}.pkl"),
                        os.path.join(folder, f"vec_normalize_{ticker}.pkl"),
                    ]
                    for norm_file in norm_files:
                        if os.path.exists(norm_file):
                            try:
                                os.remove(norm_file)
                                print(f"[DEBUG] Deleted normalizer: {norm_file}")
                            except:
                                pass  # Ignore errors deleting normalizer
                
                return {"status": "deleted", "model": model_name}
        
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TRAINING ====================

# Import TrainingMonitor for reading status file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_monitor import TrainingMonitor

# Store for training status
training_status = {
    "active": False,
    "current_step": 0,
    "total_steps": 0,
    "mean_reward": 0,
    "started_at": None,
    "ticker": None,
    "process": None,
    "rewards_history": [],
    "completed": False,
    "initial_reward": None,  # Store the first reward for accurate improvement calculation
    "baseline_reward": None,  # Average of first 10 stable episodes (professional baseline)
    "worst_reward": None,     # Worst (most negative) reward seen
    "best_reward": None       # Best (least negative) reward seen
}

# Initialize the monitor with ABSOLUTE path to project root's logs folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_FILE_PATH = os.path.join(PROJECT_ROOT, "logs", "training_status.json")
training_monitor = TrainingMonitor(status_file=STATUS_FILE_PATH)

@app.post("/train")
def start_training(req: TrainRequest):
    """Launches training script in background."""
    global training_status
    
    if training_status["active"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Determine which script to run
        if req.data_type == "real":
            script = "train_real_market.py"
            cmd = [
                sys.executable, script,
                "--timesteps", str(req.timesteps),
                "--lr", str(req.learning_rate),
                "--ticker", req.ticker,
                "--algorithm", req.algorithm
            ]
        else:
            script = "train_agent.py"
            cmd = [
                sys.executable, script,
                "--timesteps", str(req.timesteps),
                "--lr", str(req.learning_rate)
            ]
        
        # Reset training status - MUST use global keyword to modify the global dict
        training_status["active"] = True
        training_status["current_step"] = 0
        training_status["total_steps"] = req.timesteps
        training_status["mean_reward"] = 0
        training_status["started_at"] = datetime.now().isoformat()
        training_status["ticker"] = req.ticker
        training_status["process"] = None
        training_status["rewards_history"] = []  # Clear rewards history for new training!
        training_status["completed"] = False
        training_status["initial_reward"] = None  # Reset for new training
        training_status["baseline_reward"] = None  # Reset baseline for new training
        training_status["worst_reward"] = None     # Reset worst for new training
        training_status["best_reward"] = None      # Reset best for new training
        
        # Reset the monitor status file with new total_timesteps
        training_monitor.start_training(req.timesteps)
        
        # Small delay to ensure file is written before process starts
        import time
        time.sleep(0.1)
        
        # Start process in background
        # IMPORTANT: Don't capture stdout/stderr to avoid buffer blocking!
        # The training script writes progress to the status file, so we don't need stdout
        process = subprocess.Popen(
            cmd, 
            cwd=project_root,
            stdout=subprocess.DEVNULL,  # Don't capture stdout to avoid buffer blocking
            stderr=subprocess.DEVNULL,  # Don't capture stderr either
        )
        training_status["process"] = process
        
        # No need for background task - training script writes to status file directly
        # The /training-status endpoint reads from the file
        
        return {
            "status": "Training started",
            "command": " ".join(cmd),
            "config": {
                "timesteps": req.timesteps,
                "learning_rate": req.learning_rate,
                "ticker": req.ticker,
                "data_type": req.data_type
            }
        }
    except Exception as e:
        training_status["active"] = False
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training-status")
def get_training_status():
    """Get current training status by reading the training_status.json file."""
    global training_status
    
    # Check if process is still running (like app.py's process.poll() check)
    process = training_status.get("process")
    process_alive = False
    if process:
        poll_result = process.poll()
        process_alive = poll_result is None
        if not process_alive and training_status["active"]:
            print(f"[DEBUG] Training process exited with code: {poll_result}")
            # Process ended - check if it completed or crashed
            training_status["active"] = False
    
    # Read from the status file (same as app.py)
    file_status = training_monitor.get_status()
    
    # Get the configured total steps for THIS training run
    configured_total_steps = training_status.get("total_steps", 0)
    
    print(f"[DEBUG] File status: {file_status}, process_alive: {process_alive}, configured_steps: {configured_total_steps}")
    
    if file_status:
        current_step = file_status.get('current_timestep', 0)
        file_total_steps = file_status.get('total_timesteps', 0)
        mean_reward = file_status.get('mean_reward')
        status_str = file_status.get('status', 'idle')
        last_update = file_status.get('last_update', '')
        
        # Use configured total_steps (from /train request), not file's total_steps
        # This prevents issues when old file has different total_steps
        total_steps = configured_total_steps if configured_total_steps > 0 else file_total_steps
        
        # Validate: if file shows steps > our configured total, it's stale data from previous run
        # In this case, if training just started (process alive, step 0 in our state), ignore stale file data
        if current_step > total_steps and total_steps > 0 and training_status["current_step"] == 0:
            print(f"[DEBUG] Ignoring stale file data: file_step={current_step} > configured_total={total_steps}")
            current_step = 0
            mean_reward = None
            status_str = 'training'
        
        # Check if training is stuck (no update for 30+ seconds)
        is_stuck = False
        if last_update and training_status["active"]:
            try:
                last_update_time = datetime.fromisoformat(last_update)
                seconds_since_update = (datetime.now() - last_update_time).total_seconds()
                if seconds_since_update > 30 and not process_alive:
                    is_stuck = True
                    print(f"[DEBUG] Training appears stuck - no update for {seconds_since_update:.0f}s and process not alive")
            except:
                pass
        
        print(f"[DEBUG] current_step={current_step}, mean_reward={mean_reward}, status={status_str}")
        
        # Update rewards history if we have a new reward AND step is valid
        if mean_reward is not None and current_step > 0 and current_step <= total_steps:
            # Only add if this is a new step (avoid duplicates) and step is increasing
            should_add = True
            if training_status["rewards_history"]:
                last_entry = training_status["rewards_history"][-1]
                # Only add if step is greater than last recorded step (monotonic increase)
                if last_entry["step"] >= current_step:
                    should_add = False
            
            if should_add:
                reward_value = float(mean_reward)
                training_status["rewards_history"].append({
                    "step": current_step,
                    "reward": reward_value
                })
                
                # Track worst and best rewards (professional metrics)
                if training_status["worst_reward"] is None or reward_value < training_status["worst_reward"]:
                    training_status["worst_reward"] = reward_value
                if training_status["best_reward"] is None or reward_value > training_status["best_reward"]:
                    training_status["best_reward"] = reward_value
                
                # Capture baseline reward (professional approach)
                # Use average of episodes 5-15 (after initial noise settles, before much learning)
                # This gives a stable baseline that represents "untrained" performance
                history_len = len(training_status["rewards_history"])
                if training_status["baseline_reward"] is None and history_len >= 15:
                    # Skip first 5 (warmup noise), use next 10 as baseline
                    baseline_rewards = [r["reward"] for r in training_status["rewards_history"][5:15]]
                    training_status["baseline_reward"] = float(np.mean(baseline_rewards))
                    print(f"[DEBUG] Captured baseline_reward={training_status['baseline_reward']:.4f} (mean of episodes 5-15)")
                
                # Also capture initial_reward for backward compatibility (first 5 episodes avg)
                if training_status["initial_reward"] is None and history_len >= 5:
                    first_rewards = [r["reward"] for r in training_status["rewards_history"][:5]]
                    training_status["initial_reward"] = float(np.mean(first_rewards))
                    print(f"[DEBUG] Captured initial_reward={training_status['initial_reward']:.4f} (mean of first 5 rewards)")
                
                print(f"[DEBUG] Added reward point: step={current_step}, reward={mean_reward}")
        
        # Update internal state
        training_status["current_step"] = current_step
        training_status["mean_reward"] = float(mean_reward) if mean_reward is not None else 0
        if total_steps > 0:
            training_status["total_steps"] = total_steps
        
        # Check if training is complete
        is_completed = (
            status_str == 'completed' or 
            (current_step >= total_steps and current_step > 0 and total_steps > 0) or
            (is_stuck and current_step > 0)  # Consider stuck training as completed
        )
        is_active = process_alive and status_str == 'training' and not is_completed
        
        if is_completed or is_stuck:
            training_status["active"] = False
            training_status["completed"] = True
        
        # Calculate improvement (professional approach)
        # Use baseline_reward (avg of episodes 5-15) as the stable baseline
        # For hedging (negative rewards): improvement = how much closer to 0
        improvement = None
        current_reward = float(mean_reward) if mean_reward is not None else 0
        baseline = training_status.get("baseline_reward")
        
        if baseline is not None and abs(baseline) > 0.05:
            # Both rewards should be negative for hedging
            # Improvement = (|baseline| - |current|) / |baseline| * 100
            # Going from -2.0 to -0.5 = (2.0 - 0.5) / 2.0 * 100 = +75%
            if baseline < 0 and current_reward < 0:
                improvement = ((abs(baseline) - abs(current_reward)) / abs(baseline)) * 100
            elif baseline < 0 and current_reward >= 0:
                # Went from negative to positive (unlikely but handle it)
                improvement = 100.0  # 100% improvement
            else:
                # Standard case
                improvement = ((current_reward - baseline) / abs(baseline)) * 100
        
        response = {
            "active": is_active,
            "current_step": current_step,
            "total_steps": training_status["total_steps"],
            "mean_reward": current_reward,
            "started_at": training_status.get("started_at"),
            "ticker": training_status.get("ticker"),
            "rewards_history": training_status["rewards_history"][-100:],  # Last 100 points
            "initial_reward": training_status.get("initial_reward"),
            "baseline_reward": training_status.get("baseline_reward"),
            "worst_reward": training_status.get("worst_reward"),
            "best_reward": training_status.get("best_reward"),
            "improvement": improvement,  # Server-calculated improvement
            "completed": is_completed,
            "status": "completed" if is_completed else ("stuck" if is_stuck else status_str),
            "process_alive": process_alive
        }
        print(f"[DEBUG] Returning {len(response['rewards_history'])} reward history points, active={is_active}, improvement={improvement}")
        return response
    
    # Fallback to internal state if file not found
    print("[DEBUG] No file status found, returning internal state")
    return {
        "active": training_status["active"],
        "current_step": training_status["current_step"],
        "total_steps": training_status["total_steps"],
        "mean_reward": training_status["mean_reward"],
        "started_at": training_status["started_at"],
        "ticker": training_status["ticker"],
        "rewards_history": training_status["rewards_history"][-100:],
        "initial_reward": training_status.get("initial_reward"),
        "baseline_reward": training_status.get("baseline_reward"),
        "worst_reward": training_status.get("worst_reward"),
        "best_reward": training_status.get("best_reward"),
        "improvement": None,
        "completed": training_status["completed"],
        "status": "idle",
        "process_alive": False
    }

@app.post("/stop-training")
def stop_training():
    """Stop active training."""
    global training_status
    
    try:
        if training_status.get("process"):
            training_status["process"].terminate()
    except:
        pass
    
    training_status["active"] = False
    training_status["completed"] = True
    return {"status": "Training stopped"}

# ==================== SIMULATION / GAME ====================

@app.post("/launch-sim")
def launch_simulation(model_path: Optional[str] = None):
    """Launches the Pygame Man vs Machine game."""
    try:
        script_name = "game_launcher.py"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if not os.path.exists(os.path.join(project_root, script_name)):
            raise HTTPException(status_code=404, detail=f"Game script not found: {script_name}")
        
        # Set environment variable for model path if provided
        env = os.environ.copy()
        if model_path:
            env['GAME_MODEL_PATH'] = model_path
        
        subprocess.Popen(
            [sys.executable, script_name],
            cwd=project_root,
            env=env
        )
        
        return {"status": "Game launched", "script": script_name, "model": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== BACKTEST ====================

import scipy.stats as stats

def calc_max_drawdown(pnl_array: np.ndarray) -> dict:
    """
    Calculate maximum drawdown with professional metrics.
    
    Max Drawdown = max over all t of (peak_before_t - value_at_t)
    where peak_before_t = max(value[0:t])
    """
    if len(pnl_array) < 2:
        return {
            "max_drawdown": 0, "max_drawdown_pct": 0,
            "peak_idx": 0, "trough_idx": 0, "recovery_idx": None,
            "drawdown_duration": 0, "underwater_periods": 0
        }
    
    # Calculate running maximum (peak at each point)
    running_max = np.maximum.accumulate(pnl_array)
    
    # Drawdown at each point = peak - current value
    drawdown = running_max - pnl_array
    
    # Maximum drawdown is the largest peak-to-trough decline
    max_dd = float(np.max(drawdown))
    max_dd_idx = int(np.argmax(drawdown))  # Trough index
    
    # Find the peak that preceded this trough
    # It's the index where running_max equals running_max at trough
    peak_value = running_max[max_dd_idx]
    peak_idx = 0
    for i in range(max_dd_idx + 1):
        if pnl_array[i] == peak_value:
            peak_idx = i
            break
    
    # Find recovery point (when we get back to the peak)
    recovery_idx = None
    for i in range(max_dd_idx, len(pnl_array)):
        if pnl_array[i] >= peak_value:
            recovery_idx = i
            break
    
    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": float(max_dd / (abs(peak_value) + 1e-8) * 100) if peak_value != 0 else 0,
        "peak_idx": peak_idx,
        "trough_idx": max_dd_idx,
        "recovery_idx": recovery_idx,
        "drawdown_duration": max_dd_idx - peak_idx,
        "underwater_periods": int(np.sum(drawdown > 0))
    }


def calc_drawdown_series(pnl_array: np.ndarray) -> list:
    """Calculate drawdown at each point."""
    cumulative = pnl_array - pnl_array[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    return drawdown.tolist()


def calc_hedge_effectiveness(hedged_pnl: np.ndarray, unhedged_pnl: np.ndarray) -> dict:
    """
    Calculate hedge effectiveness ratio (Ederington, 1979).
    
    Formula: HE = 1 - (Var(hedged) / Var(unhedged))
    
    Interpretation:
    - HE = 1.0: Perfect hedge (100% variance reduction)
    - HE = 0.5: 50% variance reduction  
    - HE = 0.0: No hedging benefit
    - HE < 0: Hedge actually increases risk!
    """
    hedged_returns = np.diff(hedged_pnl)
    unhedged_returns = np.diff(unhedged_pnl)
    
    var_hedged = np.var(hedged_returns, ddof=1) if len(hedged_returns) > 1 else 0
    var_unhedged = np.var(unhedged_returns, ddof=1) if len(unhedged_returns) > 1 else 1
    
    # Hedge Effectiveness = 1 - (var_hedged / var_unhedged)
    # Can be negative if hedge increases variance!
    if var_unhedged < 1e-10:
        he_ratio = 0.0  # Avoid division by zero
    else:
        he_ratio = 1 - (var_hedged / var_unhedged)
    
    var_reduction = (var_unhedged - var_hedged) / (var_unhedged + 1e-10) * 100
    
    # Interpretation
    if he_ratio > 0.9:
        interpretation = "Excellent"
    elif he_ratio > 0.7:
        interpretation = "Good"
    elif he_ratio > 0.5:
        interpretation = "Moderate"
    elif he_ratio > 0:
        interpretation = "Poor"
    else:
        interpretation = "Negative (increases risk)"
    
    return {
        "hedge_effectiveness": float(he_ratio),  # Allow negative values!
        "variance_reduction_pct": float(var_reduction),
        "variance_hedged": float(var_hedged),
        "variance_unhedged": float(var_unhedged),
        "interpretation": interpretation
    }


def calc_tracking_error(strategy_pnl: np.ndarray, benchmark_pnl: np.ndarray) -> dict:
    """
    Calculate tracking error vs benchmark.
    
    Tracking Error = annualized std dev of (strategy_returns - benchmark_returns)
    Information Ratio = mean(excess_return) / tracking_error
    """
    strategy_returns = np.diff(strategy_pnl)
    benchmark_returns = np.diff(benchmark_pnl)
    
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return {
            "tracking_error_daily": 0, "tracking_error_annual": 0,
            "information_ratio": 0, "cumulative_excess": 0, "max_deviation": 0
        }
    
    # Make sure arrays are same length
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    tracking_diff = strategy_returns - benchmark_returns
    
    # Tracking error is the std dev of the difference in returns
    te_daily = float(np.std(tracking_diff, ddof=1))
    te_annual = te_daily * np.sqrt(252)
    
    # Information Ratio = annualized excess return / tracking error
    mean_excess = float(np.mean(tracking_diff))
    annualized_excess = mean_excess * 252
    info_ratio = annualized_excess / (te_annual + 1e-10) if te_annual > 0 else 0
    
    return {
        "tracking_error_daily": te_daily,
        "tracking_error_annual": float(te_annual),
        "information_ratio": float(info_ratio),
        "cumulative_excess": float(np.sum(tracking_diff)),
        "max_deviation": float(np.max(np.abs(tracking_diff))) if len(tracking_diff) > 0 else 0
    }


def calc_var_cvar(pnl_array: np.ndarray, confidence_levels: list = [0.95, 0.99]) -> dict:
    """
    Calculate Value at Risk and Conditional VaR (Expected Shortfall).
    
    VaR at confidence α: The loss value such that P(Loss > VaR) = 1-α
    For 95% VaR, we find the 5th percentile of P&L changes.
    
    CVaR (ES): Expected loss given that loss exceeds VaR
    """
    returns = np.diff(pnl_array)
    if len(returns) == 0:
        return {"var_95": 0, "var_99": 0, "cvar_95": 0, "cvar_99": 0}
    
    result = {}
    for conf in confidence_levels:
        alpha = 1 - conf  # For 95% VaR, alpha = 0.05
        conf_str = f"{int(conf*100)}"
        
        # Historical VaR: percentile at alpha level (e.g., 5th percentile for 95% VaR)
        # VaR is the loss threshold - returns below this value occur with probability alpha
        var_value = float(np.percentile(returns, alpha * 100))
        
        # CVaR (Expected Shortfall): mean of returns below VaR threshold
        tail_losses = returns[returns <= var_value]
        cvar_value = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_value
        
        result[f"var_{conf_str}"] = var_value
        result[f"cvar_{conf_str}"] = cvar_value
    
    return result


def calc_all_performance_metrics(pnl_array: np.ndarray) -> dict:
    """Calculate comprehensive performance metrics."""
    returns = np.diff(pnl_array)
    
    if len(returns) == 0:
        return {
            "finalPnL": 0, "totalReturn": 0, "avgDailyPnL": 0,
            "dailyVolatility": 0, "annualVolatility": 0, "variance": 0,
            "sharpeRatio": 0, "sortinoRatio": 0, "calmarRatio": 0,
            "maxDrawdown": 0, "maxDrawdownPct": 0, "avgDrawdown": 0,
            "var95": 0, "var99": 0, "cvar95": 0, "cvar99": 0,
            "skewness": 0, "kurtosis": 0, "winRate": 0,
            "positiveDays": 0, "negativeDays": 0
        }
    
    # Basic P&L
    final_pnl = float(pnl_array[-1])
    total_return = float(pnl_array[-1] - pnl_array[0])
    avg_daily = float(np.mean(returns))
    
    # Volatility
    daily_vol = float(np.std(returns, ddof=1))
    annual_vol = daily_vol * np.sqrt(252)
    variance = float(np.var(returns, ddof=1))
    
    # Sharpe Ratio
    sharpe = float(np.mean(returns) / (daily_vol + 1e-8) * np.sqrt(252))
    
    # Sortino Ratio (downside risk)
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-8
    sortino = float(np.mean(returns) / downside_std * np.sqrt(252))
    
    # Drawdown metrics
    dd_metrics = calc_max_drawdown(pnl_array)
    dd_series = calc_drawdown_series(pnl_array)
    avg_dd = float(np.mean(dd_series))
    
    # Calmar Ratio (return / max drawdown)
    calmar = float(total_return / (dd_metrics["max_drawdown"] + 1e-8))
    
    # VaR/CVaR
    risk_metrics = calc_var_cvar(pnl_array)
    
    # Distribution stats
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))
    
    # Win rate
    positive_days = int(np.sum(returns > 0))
    negative_days = int(np.sum(returns < 0))
    win_rate = float(positive_days / len(returns) * 100) if len(returns) > 0 else 0
    
    return {
        "finalPnL": round(final_pnl, 2),
        "totalReturn": round(total_return, 2),
        "avgDailyPnL": round(avg_daily, 4),
        "dailyVolatility": round(daily_vol, 4),
        "annualVolatility": round(annual_vol * 100, 2),
        "variance": round(variance, 6),
        "sharpeRatio": round(sharpe, 2),
        "sortinoRatio": round(sortino, 2),
        "calmarRatio": round(calmar, 2),
        "maxDrawdown": round(dd_metrics["max_drawdown"], 2),
        "maxDrawdownPct": round(dd_metrics["max_drawdown_pct"], 2),
        "avgDrawdown": round(avg_dd, 2),
        "var95": round(risk_metrics["var_95"], 2),
        "var99": round(risk_metrics["var_99"], 2),
        "cvar95": round(risk_metrics["cvar_95"], 2),
        "cvar99": round(risk_metrics["cvar_99"], 2),
        "skewness": round(skew, 2),
        "kurtosis": round(kurt, 2),
        "winRate": round(win_rate, 1),
        "positiveDays": positive_days,
        "negativeDays": negative_days
    }


def load_rl_model_for_backtest(model_path: str, algorithm: str = "PPO"):
    """Load a trained RL model for backtesting."""
    try:
        if algorithm.upper() == "SAC":
            from stable_baselines3 import SAC
            model = SAC.load(model_path)
        else:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        print(f"Loaded {algorithm} model from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def load_vec_normalize_stats(ticker: str, algorithm: str = "PPO"):
    """Load VecNormalize statistics for proper observation normalization."""
    try:
        import pickle
        stats_path = os.path.join(PROJECT_ROOT, f"saved_models_real/vec_normalize_{algorithm.lower()}_{ticker}.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                vec_normalize = pickle.load(f)
            print(f"Loaded VecNormalize stats from {stats_path}")
            return vec_normalize
        else:
            print(f"VecNormalize stats not found at {stats_path}")
            return None
    except Exception as e:
        print(f"Failed to load VecNormalize: {e}")
        return None


def run_ai_backtest(
    model,
    prices: np.ndarray,
    strike: float,
    T_years: float,
    r: float,
    sigma: float,
    transaction_cost_pct: float = 0.0005,
    vec_normalize = None
) -> np.ndarray:
    """
    Run RL agent through historical data.
    
    IMPORTANT: Observation must match training env format exactly!
    Training env uses 9 observations (normalized):
    [S/S0, sigma, tau/T, delta, gamma*S0, vega/S0, theta/S0, hedge_pos/option_pos, pnl/S0]
    
    Then VecNormalize applies running mean/std normalization on top.
    """
    n_days = len(prices)
    S0 = prices[0]  # Initial price for normalization
    
    # Initialize
    hedge_position = 0.0
    option_position = -1.0  # Short 1 call (matches training env)
    cash = 0.0
    pnl = 0.0
    pnl_history = []
    
    # Initial premium
    bs = black_scholes(S0, strike, T_years, r, sigma)
    premium = bs["price"]
    cash = premium  # Received premium from selling call
    
    for i in range(n_days):
        T_remaining = max(0.001, T_years - (i / 252))
        tau_normalized = T_remaining / T_years  # Normalized time remaining
        
        # Calculate Greeks
        bs = black_scholes(prices[i], strike, T_remaining, r, sigma)
        option_value = bs["price"]
        delta = bs["delta"]
        gamma = bs["gamma"]
        vega = bs["vega"]
        theta = bs["theta"]
        
        # Current portfolio P&L
        pnl = cash + hedge_position * prices[i] - option_value
        
        # Construct observation EXACTLY matching training env format
        # DerivativeHedgingEnv._get_observation() with normalize_state=True:
        obs = np.array([
            prices[i] / S0,                              # S / S0 (normalized price)
            sigma,                                        # Volatility
            tau_normalized,                               # tau / T (normalized time)
            delta,                                        # Delta
            gamma * S0,                                   # Gamma * S0 (scaled gamma)
            vega / S0,                                    # Vega / S0 (scaled vega)
            theta / S0,                                   # Theta / S0 (scaled theta)
            hedge_position / abs(option_position),        # Hedge pos / option pos
            pnl / S0                                      # P&L / S0 (normalized P&L)
        ], dtype=np.float32)
        
        # Apply VecNormalize if available (critical for matching training!)
        if vec_normalize is not None:
            try:
                # VecNormalize expects shape (n_envs, obs_dim), so reshape
                obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))
                obs = obs_normalized.flatten().astype(np.float32)
            except Exception as e:
                print(f"VecNormalize failed: {e}, using raw obs")
        
        # Get action from model
        try:
            action, _ = model.predict(obs, deterministic=True)
            # CRITICAL: Apply same action transformation as training environment!
            # In training, action is in [-1, 1] (clipped from [-2, 2])
            # And transformed as: hedge_ratio = (action / 2.0) + 0.5
            # So: action -1 → hedge 0%, action 0 → hedge 50%, action +1 → hedge 100%
            raw_action = float(action[0])
            hedge_ratio = (raw_action / 2.0) + 0.5
            hedge_ratio = float(np.clip(hedge_ratio, 0.0, 1.0))  # Clip to valid range
            
            # Debug: log first few actions
            if i < 5:
                print(f"Day {i}: raw_action={raw_action:.4f}, hedge_ratio={hedge_ratio:.4f}, delta={delta:.4f}")
        except Exception as e:
            print(f"Model predict error: {e}")
            hedge_ratio = 1.0  # Fallback to delta hedge
        
        # Target hedge position = hedge_ratio * delta * abs(option_position)
        # Since option_position = -1 (short), we hedge with positive delta
        target_hedge = hedge_ratio * delta * abs(option_position)
        
        # Execute trade (except on last day)
        if i < n_days - 1:
            hedge_adj = target_hedge - hedge_position
            if abs(hedge_adj) > 0.001:  # Minimum trade threshold
                trade_cost = abs(hedge_adj) * prices[i] * transaction_cost_pct
                cash -= hedge_adj * prices[i] + trade_cost
            hedge_position = target_hedge
        
        # Record portfolio P&L
        portfolio_value = cash + hedge_position * prices[i] - option_value
        pnl_history.append(portfolio_value)
    
    return np.array(pnl_history)


@app.post("/backtest")
def run_backtest(req: BacktestRequest):
    """Run a comprehensive historical backtest comparing hedging strategies."""
    try:
        # Download data
        data = yf.download(req.ticker, start=req.start_date, end=req.end_date, progress=False)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data for this period")
        
        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close'].iloc[:, 0].values
        else:
            prices = data['Close'].values
        
        prices = np.ravel(prices)
        
        # Setup
        initial_price = float(prices[0])
        strike = req.strike if req.strike else initial_price  # ATM option
        r = 0.05  # Risk-free rate
        returns = np.diff(prices) / prices[:-1]
        sigma = max(np.std(returns) * np.sqrt(252), 0.15)  # Historical volatility
        
        total_days = len(prices)
        T_total = total_days / 252.0
        
        # Calculate initial premium
        bs = black_scholes(initial_price, strike, T_total, r, sigma)
        premium = bs["price"]
        
        # ============ UNHEDGED STRATEGY ============
        unhedged_pnl = []
        for i in range(len(prices)):
            T_remaining = max(0.001, (total_days - i) / 252.0)
            current_bs = black_scholes(prices[i], strike, T_remaining, r, sigma)
            option_value = current_bs["price"]
            # P&L = premium received - current option liability
            unhedged_pnl.append(premium - option_value)
        unhedged_pnl = np.array(unhedged_pnl)
        
        # ============ DELTA HEDGE STRATEGY ============
        delta_pnl = []
        delta_position = 0.0
        delta_cash = premium  # Start with premium received
        delta_trades = 0
        delta_turnover = 0.0
        
        # Rebalancing frequency
        rebalance_interval = 1 if req.rebalance_freq == "daily" else 5  # weekly = 5 days
        
        for i in range(len(prices)):
            T_remaining = max(0.001, (total_days - i) / 252.0)
            current_bs = black_scholes(prices[i], strike, T_remaining, r, sigma)
            option_value = current_bs["price"]
            option_delta = current_bs["delta"]
            
            # Rebalance based on frequency
            should_rebalance = (i % rebalance_interval == 0) or (i == 0)
            
            if should_rebalance and i < len(prices) - 1:
                hedge_adj = option_delta - delta_position
                if abs(hedge_adj) > 0.001:  # Minimum trade threshold
                    trade_value = abs(hedge_adj) * prices[i]
                    cost = trade_value * 0.0005  # 5 bps transaction cost
                    delta_cash -= hedge_adj * prices[i] + cost
                    delta_turnover += abs(hedge_adj)
                    delta_trades += 1
                    delta_position = option_delta
            
            # Portfolio value = cash + stock position - option liability
            portfolio_value = delta_cash + delta_position * prices[i] - option_value
            delta_pnl.append(portfolio_value)
        
        delta_pnl = np.array(delta_pnl)
        
        # ============ AI RL AGENT STRATEGY ============
        ai_pnl = None
        ai_model_loaded = False
        vec_normalize = None
        
        # Try to load AI model if path provided
        if req.model_path and os.path.exists(req.model_path):
            model = load_rl_model_for_backtest(req.model_path, req.algorithm or "PPO")
            if model:
                vec_normalize = load_vec_normalize_stats(req.ticker.upper(), req.algorithm or "PPO")
                ai_pnl = run_ai_backtest(model, prices, strike, T_total, r, sigma, vec_normalize=vec_normalize)
                ai_model_loaded = True
        
        # If no model, try to find the best available model for this ticker
        if ai_pnl is None:
            # Look for models in saved_models_real folder
            model_patterns = [
                (f"./saved_models_real/ppo_{req.ticker}_real.zip", "PPO"),
                (f"./saved_models_real/sac_{req.ticker}_real.zip", "SAC"),
                (f"./saved_models_real/ppo_{req.ticker.upper()}_real.zip", "PPO"),
                (f"./saved_models_real/sac_{req.ticker.upper()}_real.zip", "SAC"),
            ]
            for model_path, algo in model_patterns:
                full_path = os.path.join(PROJECT_ROOT, model_path.replace("./", ""))
                if os.path.exists(full_path):
                    model = load_rl_model_for_backtest(full_path, algo)
                    if model:
                        # Load VecNormalize stats for proper observation normalization
                        vec_normalize = load_vec_normalize_stats(req.ticker.upper(), algo)
                        ai_pnl = run_ai_backtest(model, prices, strike, T_total, r, sigma, vec_normalize=vec_normalize)
                        ai_model_loaded = True
                        print(f"AI model loaded: {full_path}, VecNormalize: {'Yes' if vec_normalize else 'No'}")
                        break
        
        # Fallback: use delta hedge if no model available
        if ai_pnl is None:
            ai_pnl = delta_pnl.copy()
            print("No AI model found, using delta hedge as fallback")
        
        # ============ CALCULATE COMPREHENSIVE METRICS ============
        unhedged_metrics = calc_all_performance_metrics(unhedged_pnl)
        delta_metrics = calc_all_performance_metrics(delta_pnl)
        ai_metrics = calc_all_performance_metrics(ai_pnl)
        
        # Hedge effectiveness comparisons
        delta_effectiveness = calc_hedge_effectiveness(delta_pnl, unhedged_pnl)
        ai_effectiveness = calc_hedge_effectiveness(ai_pnl, unhedged_pnl)
        
        # Tracking error (AI vs Delta baseline)
        tracking = calc_tracking_error(ai_pnl, delta_pnl)
        
        # Drawdown series for charts
        unhedged_dd = calc_drawdown_series(unhedged_pnl)
        delta_dd = calc_drawdown_series(delta_pnl)
        ai_dd = calc_drawdown_series(ai_pnl)
        
        # Determine winner
        if ai_metrics["finalPnL"] > delta_metrics["finalPnL"] and ai_metrics["finalPnL"] > unhedged_metrics["finalPnL"]:
            winner = "ai"
        elif delta_metrics["finalPnL"] > unhedged_metrics["finalPnL"]:
            winner = "delta"
        else:
            winner = "unhedged"
        
        return {
            "summary": {
                "ticker": req.ticker,
                "period": f"{req.start_date} to {req.end_date}",
                "tradingDays": total_days,
                "aiModelLoaded": ai_model_loaded,
                "rebalanceFreq": req.rebalance_freq,
                "optionParams": {
                    "strike": round(strike, 2),
                    "initialPremium": round(premium, 2),
                    "impliedVolatility": round(sigma * 100, 2),
                    "optionType": "European Call",
                    "initialPrice": round(initial_price, 2)
                }
            },
            "performance": {
                "unhedged": unhedged_metrics,
                "deltaHedge": delta_metrics,
                "aiAgent": ai_metrics
            },
            "comparison": {
                "hedgeEffectiveness": {
                    "deltaVsUnhedged": delta_effectiveness,
                    "aiVsUnhedged": ai_effectiveness
                },
                "trackingError": tracking,
                "winner": winner,
                "aiVsDelta": {
                    "pnlDifference": round(ai_metrics["finalPnL"] - delta_metrics["finalPnL"], 2),
                    "pnlDifferencePct": round((ai_metrics["finalPnL"] - delta_metrics["finalPnL"]) / (abs(delta_metrics["finalPnL"]) + 0.01) * 100, 1),
                    "volatilityReduction": round(delta_metrics["annualVolatility"] - ai_metrics["annualVolatility"], 2),
                    "sharpeImprovement": round(ai_metrics["sharpeRatio"] - delta_metrics["sharpeRatio"], 2)
                }
            },
            "timeseries": {
                "dates": [d.strftime("%Y-%m-%d") for d in data.index],
                "prices": prices.tolist(),
                "pnl": {
                    "unhedged": unhedged_pnl.tolist(),
                    "delta": delta_pnl.tolist(),
                    "ai": ai_pnl.tolist()
                },
                "drawdown": {
                    "unhedged": unhedged_dd,
                    "delta": delta_dd,
                    "ai": ai_dd
                }
            },
            "trading": {
                "deltaTrades": delta_trades,
                "deltaTurnover": round(delta_turnover, 2)
            },
            # Legacy format for backward compatibility
            "dates": [d.strftime("%Y-%m-%d") for d in data.index],
            "prices": prices.tolist(),
            "unhedged": {"pnl": unhedged_pnl.tolist(), "stats": unhedged_metrics},
            "deltaHedge": {"pnl": delta_pnl.tolist(), "stats": delta_metrics},
            "aiHedge": {"pnl": ai_pnl.tolist(), "stats": ai_metrics},
            "params": {
                "strike": round(strike, 2),
                "premium": round(premium, 2),
                "volatility": round(sigma * 100, 2)
            }
        }
    except Exception as e:
        import traceback
        print(f"Backtest error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
