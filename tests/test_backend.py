"""
Tests for the FastAPI Backend
Tests all API endpoints and core functionality
"""

import pytest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import black_scholes, generate_delta_surface


class TestBlackScholesCalculations:
    """Test Black-Scholes pricing and Greeks calculations"""
    
    def test_call_option_atm(self):
        """Test ATM call option pricing"""
        result = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        
        assert "price" in result
        assert "delta" in result
        assert "gamma" in result
        assert "theta" in result
        assert "vega" in result
        assert "rho" in result
        
        # ATM call should have delta around 0.5-0.6
        assert 0.4 < result["delta"] < 0.7
        assert result["price"] > 0
    
    def test_put_option_atm(self):
        """Test ATM put option pricing"""
        result = black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="put")
        
        assert -0.6 < result["delta"] < -0.3
        assert result["price"] > 0
    
    def test_deep_itm_call(self):
        """Test deep in-the-money call option"""
        result = black_scholes(S=150, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        
        assert result["delta"] > 0.9
        assert result["price"] >= 50
    
    def test_deep_otm_call(self):
        """Test deep out-of-the-money call option"""
        result = black_scholes(S=50, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
        
        assert result["delta"] < 0.1
    
    def test_zero_time_to_expiry_itm(self):
        """Test ITM option at expiry"""
        result = black_scholes(S=110, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert result["price"] == 10
        assert result["delta"] == 1.0
    
    def test_zero_time_to_expiry_otm(self):
        """Test OTM option at expiry"""
        result = black_scholes(S=90, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
        assert result["price"] == 0
        assert result["delta"] == 0.0
    
    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        
        call = black_scholes(S, K, T, r, sigma, "call")
        put = black_scholes(S, K, T, r, sigma, "put")
        
        lhs = call["price"] - put["price"]
        rhs = S - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 0.01


class TestDeltaSurface:
    """Test Delta surface generation"""
    
    def test_delta_surface_generation(self):
        """Test that delta surface is generated correctly"""
        surface = generate_delta_surface(K=100, r=0.05, sigma=0.2)
        
        assert "x" in surface
        assert "y" in surface
        assert "z" in surface
        
        assert len(surface["x"]) == 30
        assert len(surface["y"]) == 30
        assert len(surface["z"]) == 30
    
    def test_delta_values_range(self):
        """Test that delta values are between 0 and 1 for calls"""
        surface = generate_delta_surface(K=100, r=0.05, sigma=0.2)
        
        for row in surface["z"]:
            for val in row:
                assert 0 <= val <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
