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
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    strike: Optional[float] = None

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

@app.delete("/models/{model_name}")
def delete_model(model_name: str):
    """Delete a model and its associated normalizer file."""
    try:
        # Search in both folders
        for folder in ["./saved_models/", "./saved_models_real/"]:
            model_path = os.path.join(folder, model_name)
            if os.path.exists(model_path):
                os.remove(model_path)
                
                # Try to delete normalizer
                norm_file = model_path.replace(".zip", "").replace("ppo_", "vec_normalize_") + ".pkl"
                if os.path.exists(norm_file):
                    os.remove(norm_file)
                
                return {"status": "deleted", "model": model_name}
        
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    except Exception as e:
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
    "initial_reward": None  # Store the first reward for accurate improvement calculation
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
                "--ticker", req.ticker
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
        training_status["initial_reward"] = None  # Reset initial reward for new training
        
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
                training_status["rewards_history"].append({
                    "step": current_step,
                    "reward": float(mean_reward)
                })
                # Capture initial reward for accurate improvement calculation
                # Wait until we have at least 5 data points to get a stable baseline
                # This avoids capturing unstable early rewards during VecNormalize warmup
                if training_status["initial_reward"] is None and len(training_status["rewards_history"]) >= 5:
                    # Use the mean of first 5 rewards as the baseline
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
        
        response = {
            "active": is_active,
            "current_step": current_step,
            "total_steps": training_status["total_steps"],
            "mean_reward": float(mean_reward) if mean_reward is not None else 0,
            "started_at": training_status.get("started_at"),
            "ticker": training_status.get("ticker"),
            "rewards_history": training_status["rewards_history"][-100:],  # Last 100 points
            "initial_reward": training_status.get("initial_reward"),  # First reward for improvement calc
            "completed": is_completed,
            "status": "completed" if is_completed else ("stuck" if is_stuck else status_str),
            "process_alive": process_alive
        }
        print(f"[DEBUG] Returning {len(response['rewards_history'])} reward history points, active={is_active}")
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
        "initial_reward": training_status.get("initial_reward"),  # First reward for improvement calc
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

@app.post("/backtest")
def run_backtest(req: BacktestRequest):
    """Run a historical backtest comparing hedging strategies."""
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
        strike = req.strike if req.strike else initial_price
        r = 0.05
        returns = np.diff(prices) / prices[:-1]
        sigma = max(np.std(returns) * np.sqrt(252), 0.15)
        
        total_days = len(prices)
        T_total = total_days / 252.0
        
        # Calculate initial premium
        bs = black_scholes(initial_price, strike, T_total, r, sigma)
        premium = bs["price"]
        
        # Calculate P&L arrays
        unhedged_pnl = []
        delta_pnl = []
        ai_pnl = []
        
        delta_position = 0.0
        delta_cash = premium
        
        for i in range(len(prices)):
            days_remaining = total_days - i
            T_remaining = days_remaining / 252.0
            
            current_bs = black_scholes(prices[i], strike, T_remaining, r, sigma)
            option_value = current_bs["price"]
            option_delta = current_bs["delta"]
            
            # Unhedged
            unhedged_pnl.append(premium - option_value)
            
            # Delta hedge
            if i < len(prices) - 1:
                hedge_adj = option_delta - delta_position
                cost = abs(hedge_adj) * 0.0005 * prices[i]
                delta_cash -= hedge_adj * prices[i] + cost
                delta_position = option_delta
            
            delta_pnl.append(delta_cash + delta_position * prices[i] - option_value)
            
            # AI (simplified - same as delta for now unless model loaded)
            ai_pnl.append(delta_pnl[-1])
        
        # Calculate stats
        def calc_stats(pnl_array):
            returns = np.diff(pnl_array)
            return {
                "finalPnL": round(pnl_array[-1], 2),
                "maxDrawdown": round(min(pnl_array) - pnl_array[0], 2),
                "volatility": round(np.std(returns) * np.sqrt(252) * 100, 2) if len(returns) > 0 else 0,
                "sharpe": round(pnl_array[-1] / (np.std(returns) * np.sqrt(252) + 1e-8), 2) if len(returns) > 0 else 0
            }
        
        return {
            "ticker": req.ticker,
            "period": f"{req.start_date} to {req.end_date}",
            "dates": [d.strftime("%Y-%m-%d") for d in data.index],
            "prices": prices.tolist(),
            "unhedged": {
                "pnl": unhedged_pnl,
                "stats": calc_stats(np.array(unhedged_pnl))
            },
            "deltaHedge": {
                "pnl": delta_pnl,
                "stats": calc_stats(np.array(delta_pnl))
            },
            "aiHedge": {
                "pnl": ai_pnl,
                "stats": calc_stats(np.array(ai_pnl))
            },
            "params": {
                "strike": strike,
                "premium": round(premium, 2),
                "volatility": round(sigma * 100, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
