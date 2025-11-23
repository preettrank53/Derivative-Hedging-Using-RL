"""
Option Pricing and Greeks Calculation Module

This module provides functions for pricing European options using the Black-Scholes model
and calculating option Greeks (delta, gamma, vega, theta, rho).
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple


class BlackScholesModel:
    """
    Black-Scholes option pricing model for European options.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call'):
        """
        Initialize Black-Scholes model parameters.
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility (annualized)
        option_type : str
            'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def _d1(self) -> float:
        """Calculate d1 parameter in Black-Scholes formula."""
        if self.T <= 0:
            return 0
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def _d2(self) -> float:
        """Calculate d2 parameter in Black-Scholes formula."""
        if self.T <= 0:
            return 0
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    def price(self) -> float:
        """
        Calculate option price using Black-Scholes formula.
        
        Returns:
        --------
        float
            Option price
        """
        if self.T <= 0:
            # Option at maturity
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)
        
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        return price
    
    def delta(self) -> float:
        """
        Calculate option delta (∂V/∂S).
        
        Delta measures the rate of change of option price with respect to the underlying asset price.
        
        Returns:
        --------
        float
            Delta value
        """
        if self.T <= 0:
            if self.option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        
        d1 = self._d1()
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def gamma(self) -> float:
        """
        Calculate option gamma (∂²V/∂S²).
        
        Gamma measures the rate of change of delta with respect to the underlying asset price.
        
        Returns:
        --------
        float
            Gamma value
        """
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        """
        Calculate option vega (∂V/∂σ).
        
        Vega measures the sensitivity of option price to volatility changes.
        Returns vega per 1% (0.01) change in volatility.
        
        Returns:
        --------
        float
            Vega value (per 1% volatility change)
        """
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100
    
    def theta(self) -> float:
        """
        Calculate option theta (∂V/∂t).
        
        Theta measures the rate of change of option price with respect to time.
        Returns theta per day (1/252 of a year).
        
        Returns:
        --------
        float
            Theta value (per day)
        """
        if self.T <= 0:
            return 0.0
        
        d1 = self._d1()
        d2 = self._d2()
        
        if self.option_type == 'call':
            theta = (-(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            theta = (-(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        return theta / 252  # Convert to daily theta
    
    def rho(self) -> float:
        """
        Calculate option rho (∂V/∂r).
        
        Rho measures the sensitivity of option price to interest rate changes.
        Returns rho per 1% (0.01) change in interest rate.
        
        Returns:
        --------
        float
            Rho value (per 1% interest rate change)
        """
        if self.T <= 0:
            return 0.0
        
        d2 = self._d2()
        
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
    
    def get_all_greeks(self) -> Dict[str, float]:
        """
        Calculate all Greeks at once.
        
        Returns:
        --------
        dict
            Dictionary containing delta, gamma, vega, theta, and rho
        """
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }


# Convenience functions for quick calculations

def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calculate Black-Scholes option price.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    float
        Option price
    """
    bs = BlackScholesModel(S, K, T, r, sigma, option_type)
    return bs.price()


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate all option Greeks.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    dict
        Dictionary containing all Greeks
    """
    bs = BlackScholesModel(S, K, T, r, sigma, option_type)
    return bs.get_all_greeks()


def implied_volatility(option_price: float, S: float, K: float, T: float, r: float, 
                       option_type: str = 'call', max_iterations: int = 100, 
                       tolerance: float = 1e-6) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
    -----------
    option_price : float
        Observed market price of the option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    option_type : str
        'call' or 'put'
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
    
    Returns:
    --------
    float
        Implied volatility
    """
    if T <= 0:
        raise ValueError("Time to maturity must be positive")
    
    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * option_price / S
    
    for i in range(max_iterations):
        bs = BlackScholesModel(S, K, T, r, sigma, option_type)
        price = bs.price()
        vega = bs.vega() * 100  # Convert back to per-unit change
        
        diff = option_price - price
        
        if abs(diff) < tolerance:
            return sigma
        
        if vega < 1e-10:
            raise ValueError("Vega too small, cannot converge")
        
        # Newton-Raphson update
        sigma = sigma + diff / vega
        
        # Keep sigma positive
        sigma = max(sigma, 1e-6)
    
    raise ValueError(f"Implied volatility did not converge after {max_iterations} iterations")


if __name__ == "__main__":
    # Example usage
    print("Black-Scholes Option Pricing Example")
    print("=" * 50)
    
    # Parameters
    S = 100.0      # Current stock price
    K = 100.0      # Strike price
    T = 30/252     # Time to maturity (30 trading days)
    r = 0.05       # Risk-free rate (5%)
    sigma = 0.2    # Volatility (20%)
    
    # Calculate call option
    bs_call = BlackScholesModel(S, K, T, r, sigma, 'call')
    print(f"\nCall Option (ATM):")
    print(f"Price: ${bs_call.price():.4f}")
    print(f"Delta: {bs_call.delta():.4f}")
    print(f"Gamma: {bs_call.gamma():.4f}")
    print(f"Vega: {bs_call.vega():.4f}")
    print(f"Theta: {bs_call.theta():.4f}")
    print(f"Rho: {bs_call.rho():.4f}")
    
    # Calculate put option
    bs_put = BlackScholesModel(S, K, T, r, sigma, 'put')
    print(f"\nPut Option (ATM):")
    print(f"Price: ${bs_put.price():.4f}")
    print(f"Delta: {bs_put.delta():.4f}")
    print(f"Gamma: {bs_put.gamma():.4f}")
    
    # Test implied volatility
    market_price = bs_call.price()
    implied_vol = implied_volatility(market_price, S, K, T, r, 'call')
    print(f"\nImplied Volatility Test:")
    print(f"Input volatility: {sigma:.4f}")
    print(f"Recovered implied volatility: {implied_vol:.4f}")
