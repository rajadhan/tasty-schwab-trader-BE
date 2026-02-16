import numpy as np
from scipy.stats import norm
from datetime import datetime

class GammaRWREngine:
    """
    Core math engine for calculating Greeks and Threat Metrics (G.A.R.).
    Optimized for 0DTE options.
    """
    
    @staticmethod
    def black_scholes_greeks(S, K, T, v, r, q, option_type='call'):
        """
        Calculates Black-Scholes Greeks for a single option.
        S: Spot Price
        K: Strike Price
        T: Time to Expiration (in years)
        v: Volatility (IV)
        r: Risk-free rate
        q: Dividend yield
        """
        # Avoid division by zero for T=0
        if T <= 0:
            T = 1e-9
            
        d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        
        # Gamma is the same for calls and puts
        gamma = pdf_d1 / (S * v * np.sqrt(T))
        
        if option_type.lower() == 'call':
            delta = np.exp(-q * T) * cdf_d1
            theta = (- (S * v * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2) 
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            delta = np.exp(-q * T) * (cdf_d1 - 1)
            theta = (- (S * v * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))
            
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365.0  # Daily theta
        }

    def calculate_position_greeks(self, spot, legs):
        """
        Calculates net Greeks for a multi-leg position.
        legs: List of dicts {'strike', 'iv', 'expiry_years', 'qty', 'type'}
              qty is positive for long, negative for short.
        """
        net_delta = 0
        net_gamma = 0
        net_theta = 0
        
        # Standard assumptions for 0DTE
        r = 0.05  # 5% risk-free rate
        q = 0.0   # Assume no dividends for 0DTE snapshot
        
        for leg in legs:
            greeks = self.black_scholes_greeks(
                spot, 
                leg['strike'], 
                leg['expiry_years'], 
                leg['iv'], 
                r, 
                q, 
                leg['type']
            )
            
            qty = leg['qty']
            net_delta += greeks['delta'] * qty
            net_gamma += greeks['gamma'] * qty
            net_theta += greeks['theta'] * qty
            
        # Calculate Gamma Acceleration Ratio (G.A.R.)
        # Prevent division by zero if theta is near 0
        theta_abs = abs(net_theta) if abs(net_theta) > 1e-9 else 1e-9
        gar = abs(net_gamma) / theta_abs
        
        return {
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'gar': gar,
            'timestamp': datetime.now().isoformat()
        }

    def classify_threat(self, gar):
        """
        Classifies threat based on G.A.R. levels.
        Level 1 (SEARCH): < 0.5
        Level 2 (LOCK): 0.5 - 1.5
        Level 3 (LAUNCH): >= 1.5
        """
        if gar < 0.5:
            return 'SEARCH', 'Green'
        elif gar < 1.5:
            return 'LOCK', 'Orange'
        else:
            return 'LAUNCH', 'Red'
