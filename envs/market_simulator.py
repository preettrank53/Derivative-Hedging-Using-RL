"""
Real Market Data Simulator Module

This module uses REAL historical stock data from Yahoo Finance instead of 
synthetic Geometric Brownian Motion. Train on 2020-2023, test on 2024.

The Real World Upgrade:
- Elon Musk tweets → Tesla crashes
- Earnings reports → NVIDIA moons
- COVID crash → Real market chaos
- No perfect math equations, just survival
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RealMarketConfig:
    """Configuration for real market data simulation."""
    ticker: str = 'TSLA'                    # Stock ticker (TSLA, NVDA, AAPL, etc.)
    start_date: str = '2020-01-01'          # Training data start
    end_date: str = '2023-12-31'            # Training data end
    test_start_date: str = '2024-01-01'     # Test data start
    test_end_date: str = '2024-11-24'       # Test data end (today)
    cache_dir: str = './data/market_cache'  # Cache downloaded data
    r: float = 0.05                         # Risk-free rate (5%)
    dt: float = 1/252                       # Time step (1 trading day)


class RealMarketSimulator:
    """
    Real Market Data Simulator using Yahoo Finance.
    
    Unlike GBM (perfect math), this uses REAL stock prices:
    - Tesla's volatility swings
    - NVIDIA's AI boom
    - Actual market crashes and rallies
    - News events, earnings, tweets
    
    The AI must learn to hedge in REAL chaos, not theoretical Brownian motion.
    """
    
    def __init__(self, config: RealMarketConfig, mode: str = 'train'):
        """
        Initialize real market simulator.
        
        Parameters:
        -----------
        config : RealMarketConfig
            Market data configuration
        mode : str
            'train' (2020-2023) or 'test' (2024)
        """
        self.config = config
        self.mode = mode
        
        # Load market data
        self._load_market_data()
        
        # Current state
        self.current_index = 0
        self.S = self.prices[0]
        self.time = 0.0
        
        print(f"\n{'='*70}")
        print(f"REAL MARKET SIMULATOR INITIALIZED")
        print(f"{'='*70}")
        print(f"Ticker: {config.ticker}")
        print(f"Mode: {mode.upper()}")
        print(f"Period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Trading days: {len(self.prices)}")
        print(f"Price range: ${self.prices.min():.2f} - ${self.prices.max():.2f}")
        print(f"Avg daily volatility: {self.daily_returns.std():.2%}")
        print(f"{'='*70}\n")
    
    def _load_market_data(self):
        """Load historical market data from Yahoo Finance."""
        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Determine date range based on mode
        if self.mode == 'train':
            start = self.config.start_date
            end = self.config.end_date
            cache_file = f"{self.config.cache_dir}/{self.config.ticker}_train.csv"
        else:  # test
            start = self.config.test_start_date
            end = self.config.test_end_date
            cache_file = f"{self.config.cache_dir}/{self.config.ticker}_test.csv"
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            self.data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            # Download from Yahoo Finance
            print(f"Downloading {self.config.ticker} data from Yahoo Finance...")
            print(f"   Period: {start} to {end}")
            
            try:
                ticker = yf.Ticker(self.config.ticker)
                self.data = ticker.history(start=start, end=end)
                
                if len(self.data) == 0:
                    raise ValueError(f"No data returned for {self.config.ticker}")
                
                # Save to cache
                self.data.to_csv(cache_file)
                print(f"Downloaded {len(self.data)} days of data")
                print(f"Cached to {cache_file}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to download data: {e}")
        
        # Extract close prices
        self.prices = self.data['Close'].values
        self.dates = self.data.index
        
        # Calculate daily returns
        self.daily_returns = np.diff(np.log(self.prices))
        
        # Calculate realized volatility (rolling 30-day)
        returns_series = pd.Series(self.daily_returns, index=self.dates[1:])
        self.realized_vol = returns_series.rolling(window=30).std() * np.sqrt(252)
        self.realized_vol = self.realized_vol.fillna(self.realized_vol.mean())
        
        # Validate data
        if len(self.prices) < 30:
            raise ValueError(f"Insufficient data: only {len(self.prices)} days")
    
    def reset(self, S0: Optional[float] = None, random_start: bool = True) -> float:
        """
        Reset simulator to start of episode.
        
        Parameters:
        -----------
        S0 : float, optional
            Initial price (ignored, uses real data)
        random_start : bool
            If True, start from random point in data
            If False, start from beginning
        
        Returns:
        --------
        float
            Initial stock price
        """
        if random_start and len(self.prices) > 50:
            # Start from random point (leave room for 30-step episode)
            max_start = len(self.prices) - 50
            self.current_index = np.random.randint(0, max_start)
        else:
            self.current_index = 0
        
        self.S = self.prices[self.current_index]
        self.time = 0.0
        
        return self.S
    
    def step(self, dt: Optional[float] = None) -> float:
        """
        Advance one time step using REAL market data.
        
        Parameters:
        -----------
        dt : float, optional
            Time step (ignored, always 1 day)
        
        Returns:
        --------
        float
            Next stock price (REAL historical price)
        """
        self.current_index += 1
        
        # Check if we've reached end of data
        if self.current_index >= len(self.prices):
            # Wrap around (start from beginning)
            self.current_index = 0
        
        self.S = self.prices[self.current_index]
        self.time += self.config.dt
        
        return self.S
    
    def get_current_volatility(self) -> float:
        """
        Get realized volatility at current time.
        
        Returns:
        --------
        float
            Annualized volatility (real market volatility)
        """
        if self.current_index < len(self.realized_vol):
            return self.realized_vol.iloc[self.current_index]
        else:
            return self.realized_vol.mean()
    
    def get_current_price(self) -> float:
        """Get current stock price."""
        return self.S
    
    def get_current_date(self) -> datetime:
        """Get current date in simulation."""
        if self.current_index < len(self.dates):
            return self.dates[self.current_index]
        else:
            return self.dates[-1]
    
    def get_statistics(self) -> dict:
        """Get market statistics."""
        return {
            'ticker': self.config.ticker,
            'mode': self.mode,
            'n_days': len(self.prices),
            'price_min': float(self.prices.min()),
            'price_max': float(self.prices.max()),
            'price_mean': float(self.prices.mean()),
            'daily_vol': float(self.daily_returns.std()),
            'annual_vol': float(self.daily_returns.std() * np.sqrt(252)),
            'total_return': float((self.prices[-1] / self.prices[0]) - 1),
            'sharpe_ratio': float(self.daily_returns.mean() / self.daily_returns.std() * np.sqrt(252))
        }
    
    def plot_price_history(self, save_path: Optional[str] = None):
        """Plot price history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            # Plot price
            ax1.plot(self.dates, self.prices, linewidth=2, color='cyan')
            ax1.set_title(f'{self.config.ticker} Stock Price ({self.mode.upper()} Data)', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#0a0a19')
            
            # Plot volatility
            ax2.plot(self.dates[1:], self.realized_vol, linewidth=2, color='orange')
            ax2.set_title('Realized Volatility (30-day)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Volatility', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#0a0a19')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, facecolor='#0a0a19')
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("WARNING: matplotlib not installed. Cannot plot.")


# Backward compatibility: Keep old GBM class for comparison
@dataclass
class MarketConfig:
    """Configuration for market simulation parameters."""
    S0: float = 100.0           # Initial stock price
    mu: float = 0.05            # Drift (expected return)
    sigma: float = 0.2          # Volatility
    r: float = 0.05             # Risk-free rate
    dt: float = 1/252           # Time step (1 trading day)
    
    def __post_init__(self):
        """Validate parameters."""
        if self.S0 <= 0:
            raise ValueError("Initial stock price must be positive")
        if self.sigma < 0:
            raise ValueError("Volatility must be non-negative")
        if self.dt <= 0:
            raise ValueError("Time step must be positive")


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion (GBM) price simulator.
    
    SYNTHETIC DATA: Perfect mathematical model.
    Use this for theoretical testing, RealMarketSimulator for real-world validation.
    """
    
    def __init__(self, config: MarketConfig):
        self.config = config
        self.S = config.S0
        self.time = 0.0
        
    def reset(self, S0: Optional[float] = None) -> float:
        self.S = S0 if S0 is not None else self.config.S0
        self.time = 0.0
        return self.S
    
    def step(self, dt: Optional[float] = None) -> float:
        dt = dt if dt is not None else self.config.dt
        dW = np.random.normal(0, np.sqrt(dt))
        self.S = self.S * np.exp(
            (self.config.mu - 0.5 * self.config.sigma**2) * dt 
            + self.config.sigma * dW
        )
        self.time += dt
        return self.S
    
    def get_current_price(self) -> float:
        return self.S


# Factory function
def create_market_simulator(model_type: str = 'real', **kwargs):
    """
    Factory function to create market simulators.
    
    Parameters:
    -----------
    model_type : str
        'real' - Real market data (Yahoo Finance)
        'gbm' - Geometric Brownian Motion (synthetic)
    **kwargs
        Model-specific parameters
    
    Returns:
    --------
    Market simulator instance
    """
    if model_type.lower() == 'real':
        config = RealMarketConfig(**kwargs)
        mode = kwargs.get('mode', 'train')
        return RealMarketSimulator(config, mode=mode)
    elif model_type.lower() == 'gbm':
        config = MarketConfig(**kwargs)
        return GeometricBrownianMotion(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'real' or 'gbm'")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REAL MARKET DATA SIMULATOR - TEST")
    print("="*70)
    
    # Test 1: Load Tesla training data
    print("\n1. Loading Tesla Training Data (2020-2023)")
    print("-"*70)
    config = RealMarketConfig(ticker='TSLA', start_date='2020-01-01', end_date='2023-12-31')
    sim_train = RealMarketSimulator(config, mode='train')
    
    stats = sim_train.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Test 2: Load Tesla test data
    print("\n2. Loading Tesla Test Data (2024)")
    print("-"*70)
    config_test = RealMarketConfig(ticker='TSLA', test_start_date='2024-01-01', test_end_date='2024-11-24')
    sim_test = RealMarketSimulator(config_test, mode='test')
    
    # Test 3: Simulate episode
    print("\n3. Simulating 30-Day Episode")
    print("-"*70)
    sim_train.reset(random_start=True)
    print(f"Start date: {sim_train.get_current_date().date()}")
    print(f"Start price: ${sim_train.S:.2f}")
    
    prices = [sim_train.S]
    for i in range(30):
        price = sim_train.step()
        prices.append(price)
    
    print(f"End date: {sim_train.get_current_date().date()}")
    print(f"End price: ${sim_train.S:.2f}")
    print(f"Return: {(prices[-1]/prices[0] - 1):.2%}")
    print(f"Max: ${max(prices):.2f}")
    print(f"Min: ${min(prices):.2f}")
    
    # Test 4: Try NVIDIA
    print("\n4. Loading NVIDIA Data (The AI Boom Stock)")
    print("-"*70)
    config_nvda = RealMarketConfig(ticker='NVDA')
    sim_nvda = RealMarketSimulator(config_nvda, mode='train')
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED - Real Market Simulator Ready!")
    print("="*70 + "\n")
