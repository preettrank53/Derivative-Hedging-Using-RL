"""
Market Simulator Module

This module provides various stochastic models for simulating asset price dynamics,
including Geometric Brownian Motion (GBM), Heston stochastic volatility, and jump-diffusion models.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


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
    
    The stock price follows: dS = μS dt + σS dW
    where dW is a Wiener process (Brownian motion).
    """
    
    def __init__(self, config: MarketConfig):
        """
        Initialize GBM simulator.
        
        Parameters:
        -----------
        config : MarketConfig
            Market simulation configuration
        """
        self.config = config
        self.S = config.S0
        self.time = 0.0
        
    def reset(self, S0: Optional[float] = None) -> float:
        """
        Reset the simulator to initial state.
        
        Parameters:
        -----------
        S0 : float, optional
            Initial stock price (uses config.S0 if not provided)
        
        Returns:
        --------
        float
            Initial stock price
        """
        self.S = S0 if S0 is not None else self.config.S0
        self.time = 0.0
        return self.S
    
    def step(self, dt: Optional[float] = None) -> float:
        """
        Simulate one time step.
        
        Parameters:
        -----------
        dt : float, optional
            Time step size (uses config.dt if not provided)
        
        Returns:
        --------
        float
            New stock price
        """
        dt = dt if dt is not None else self.config.dt
        
        # Generate random shock
        dW = np.random.normal(0, np.sqrt(dt))
        
        # Update stock price using exact solution to GBM SDE
        self.S = self.S * np.exp(
            (self.config.mu - 0.5 * self.config.sigma**2) * dt 
            + self.config.sigma * dW
        )
        
        self.time += dt
        return self.S
    
    def simulate_path(self, n_steps: int, dt: Optional[float] = None, 
                     S0: Optional[float] = None) -> np.ndarray:
        """
        Simulate entire price path.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        dt : float, optional
            Time step size
        S0 : float, optional
            Initial stock price
        
        Returns:
        --------
        np.ndarray
            Array of stock prices (length n_steps + 1)
        """
        dt = dt if dt is not None else self.config.dt
        S0 = S0 if S0 is not None else self.config.S0
        
        # Preallocate price path
        prices = np.zeros(n_steps + 1)
        prices[0] = S0
        
        # Generate all random shocks at once (more efficient)
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        
        # Simulate path
        for i in range(n_steps):
            prices[i + 1] = prices[i] * np.exp(
                (self.config.mu - 0.5 * self.config.sigma**2) * dt 
                + self.config.sigma * dW[i]
            )
        
        return prices
    
    def get_current_price(self) -> float:
        """Get current stock price."""
        return self.S


class HestonModel:
    """
    Heston Stochastic Volatility Model.
    
    The stock price and variance follow:
    dS = μS dt + √v S dW1
    dv = κ(θ - v) dt + σ_v √v dW2
    
    where dW1 and dW2 are correlated Wiener processes with correlation ρ.
    """
    
    def __init__(self, S0: float = 100.0, v0: float = 0.04, mu: float = 0.05,
                 kappa: float = 2.0, theta: float = 0.04, sigma_v: float = 0.3,
                 rho: float = -0.7, dt: float = 1/252):
        """
        Initialize Heston model.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        v0 : float
            Initial variance (volatility squared)
        mu : float
            Drift (expected return)
        kappa : float
            Mean reversion speed for variance
        theta : float
            Long-term mean of variance
        sigma_v : float
            Volatility of variance (vol of vol)
        rho : float
            Correlation between price and variance shocks
        dt : float
            Time step size
        """
        self.S0 = S0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.dt = dt
        
        # Current state
        self.S = S0
        self.v = v0
        self.time = 0.0
        
        # Validate Feller condition for non-negative variance
        if 2 * kappa * theta <= sigma_v**2:
            print("Warning: Feller condition not satisfied. Variance may become negative.")
    
    def reset(self, S0: Optional[float] = None, v0: Optional[float] = None) -> Tuple[float, float]:
        """
        Reset the simulator to initial state.
        
        Parameters:
        -----------
        S0 : float, optional
            Initial stock price
        v0 : float, optional
            Initial variance
        
        Returns:
        --------
        tuple
            (stock price, variance)
        """
        self.S = S0 if S0 is not None else self.S0
        self.v = v0 if v0 is not None else self.v0
        self.time = 0.0
        return self.S, self.v
    
    def step(self, dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Simulate one time step using Euler-Maruyama discretization.
        
        Parameters:
        -----------
        dt : float, optional
            Time step size
        
        Returns:
        --------
        tuple
            (new stock price, new variance)
        """
        dt = dt if dt is not None else self.dt
        
        # Generate correlated Brownian motions
        Z1 = np.random.normal(0, 1)
        Z2 = np.random.normal(0, 1)
        W1 = np.sqrt(dt) * Z1
        W2 = np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
        
        # Ensure variance stays positive (absorption scheme)
        v_pos = max(self.v, 0)
        
        # Update variance (Euler scheme)
        self.v = self.v + self.kappa * (self.theta - v_pos) * dt + self.sigma_v * np.sqrt(v_pos) * W2
        
        # Update stock price
        self.S = self.S * np.exp(
            (self.mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * W1
        )
        
        self.time += dt
        return self.S, self.v
    
    def simulate_path(self, n_steps: int, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate entire price and variance path.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps
        dt : float, optional
            Time step size
        
        Returns:
        --------
        tuple
            (price array, variance array)
        """
        dt = dt if dt is not None else self.dt
        
        prices = np.zeros(n_steps + 1)
        variances = np.zeros(n_steps + 1)
        
        prices[0] = self.S
        variances[0] = self.v
        
        for i in range(n_steps):
            S_new, v_new = self.step(dt)
            prices[i + 1] = S_new
            variances[i + 1] = v_new
        
        return prices, variances
    
    def get_current_state(self) -> Dict[str, float]:
        """Get current state (price, variance, volatility)."""
        return {
            'price': self.S,
            'variance': self.v,
            'volatility': np.sqrt(max(self.v, 0))
        }


class MertonJumpDiffusion:
    """
    Merton Jump-Diffusion Model.
    
    Combines continuous Brownian motion with discontinuous jumps:
    dS = μS dt + σS dW + S(J-1) dN
    
    where dN is a Poisson process and J is the jump size.
    """
    
    def __init__(self, S0: float = 100.0, mu: float = 0.05, sigma: float = 0.2,
                 lambda_jump: float = 0.1, jump_mean: float = 0.0, 
                 jump_std: float = 0.1, dt: float = 1/252):
        """
        Initialize Merton jump-diffusion model.
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        mu : float
            Drift (expected return)
        sigma : float
            Diffusion volatility
        lambda_jump : float
            Jump intensity (expected number of jumps per year)
        jump_mean : float
            Mean of log-jump size
        jump_std : float
            Standard deviation of log-jump size
        dt : float
            Time step size
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.dt = dt
        
        self.S = S0
        self.time = 0.0
    
    def reset(self, S0: Optional[float] = None) -> float:
        """Reset to initial state."""
        self.S = S0 if S0 is not None else self.S0
        self.time = 0.0
        return self.S
    
    def step(self, dt: Optional[float] = None) -> float:
        """
        Simulate one time step with potential jumps.
        
        Parameters:
        -----------
        dt : float, optional
            Time step size
        
        Returns:
        --------
        float
            New stock price
        """
        dt = dt if dt is not None else self.dt
        
        # Continuous diffusion component
        dW = np.random.normal(0, np.sqrt(dt))
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dW
        
        # Jump component (Poisson process)
        n_jumps = np.random.poisson(self.lambda_jump * dt)
        jump_component = 0.0
        
        if n_jumps > 0:
            # Sum of log-normal jumps
            jump_sizes = np.random.normal(self.jump_mean, self.jump_std, n_jumps)
            jump_component = np.sum(jump_sizes)
        
        # Update price
        self.S = self.S * np.exp(drift + diffusion + jump_component)
        self.time += dt
        
        return self.S
    
    def simulate_path(self, n_steps: int, dt: Optional[float] = None) -> np.ndarray:
        """Simulate entire price path with jumps."""
        dt = dt if dt is not None else self.dt
        
        prices = np.zeros(n_steps + 1)
        prices[0] = self.S
        
        for i in range(n_steps):
            prices[i + 1] = self.step(dt)
        
        return prices


# Factory function for creating simulators
def create_market_simulator(model_type: str = 'gbm', **kwargs):
    """
    Factory function to create market simulators.
    
    Parameters:
    -----------
    model_type : str
        Type of model: 'gbm', 'heston', or 'jump_diffusion'
    **kwargs
        Model-specific parameters
    
    Returns:
    --------
    Market simulator instance
    """
    if model_type.lower() == 'gbm':
        config = MarketConfig(**kwargs)
        return GeometricBrownianMotion(config)
    elif model_type.lower() == 'heston':
        return HestonModel(**kwargs)
    elif model_type.lower() in ['jump_diffusion', 'merton']:
        return MertonJumpDiffusion(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'gbm', 'heston', or 'jump_diffusion'")


if __name__ == "__main__":
    # Example usage
    print("Market Simulation Examples")
    print("=" * 70)
    
    # 1. Geometric Brownian Motion
    print("\n1. Geometric Brownian Motion (GBM)")
    print("-" * 70)
    config = MarketConfig(S0=100, mu=0.05, sigma=0.2, dt=1/252)
    gbm = GeometricBrownianMotion(config)
    
    print(f"Initial price: ${gbm.S:.2f}")
    for i in range(5):
        S = gbm.step()
        print(f"Day {i+1}: ${S:.2f}")
    
    # 2. Heston Model
    print("\n2. Heston Stochastic Volatility")
    print("-" * 70)
    heston = HestonModel(S0=100, v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
    heston.reset()
    
    print(f"Initial state: Price=${heston.S:.2f}, Vol={np.sqrt(heston.v):.2%}")
    for i in range(5):
        S, v = heston.step()
        print(f"Day {i+1}: Price=${S:.2f}, Vol={np.sqrt(max(v, 0)):.2%}")
    
    # 3. Jump-Diffusion
    print("\n3. Merton Jump-Diffusion")
    print("-" * 70)
    jump_diff = MertonJumpDiffusion(S0=100, mu=0.05, sigma=0.2, lambda_jump=2.0, 
                                    jump_mean=-0.02, jump_std=0.05)
    jump_diff.reset()
    
    print(f"Initial price: ${jump_diff.S:.2f}")
    for i in range(5):
        S_old = jump_diff.S
        S = jump_diff.step()
        change = (S - S_old) / S_old
        print(f"Day {i+1}: ${S:.2f} (change: {change:+.2%})")
    
    # 4. Simulate full path
    print("\n4. Full Path Simulation (30 days)")
    print("-" * 70)
    gbm.reset()
    path = gbm.simulate_path(n_steps=30)
    print(f"Start: ${path[0]:.2f}")
    print(f"End:   ${path[-1]:.2f}")
    print(f"Return: {(path[-1]/path[0] - 1):.2%}")
    print(f"Max:   ${path.max():.2f}")
    print(f"Min:   ${path.min():.2f}")
