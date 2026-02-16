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

    @staticmethod
    def black_scholes_price(S, K, T, v, r, q, option_type='call'):
        """Calculates the Black-Scholes price of an option."""
        if T <= 0: T = 1e-9
        d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)
        if option_type.lower() == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def solve_for_iv(self, market_price, S, K, T, r, q, option_type='call', precision=1e-5, max_iter=100):
        """Finds implied volatility using Newton-Raphson."""
        # Initial guess
        v = 0.5 
        for i in range(max_iter):
            price = self.black_scholes_price(S, K, T, v, r, q, option_type)
            diff = market_price - price
            if abs(diff) < precision:
                return v
            
            # Vega is the derivative of price w.r.t sigma
            # vega = S * exp(-qT) * pdf(d1) * sqrt(T)
            d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
            
            if abs(vega) < 1e-9:
                break
            v += diff / vega # Newton step
            
            if v <= 0: v = 0.0001 # Clamp to avoid domain errors
            if v > 5.0: v = 5.0    # Clamp crazy IVs
            
        return v

    def calculate_position_greeks(self, spot, legs):
        """
        Calculates net Greeks for a multi-leg position.
        legs: List of dicts/events containing market prices or pre-calculated Greeks.
        """
        net_delta = 0
        net_gamma = 0
        net_theta = 0
        
        r = 0.05  
        q = 0.0   
        
        for leg in legs:
            # Normalize leg items from dict if necessary
            l_strike = float(leg['strike'])
            l_expiry = float(leg['expiry_years'])
            l_type = str(leg['type'])
            l_qty = float(leg['qty'])
            l_price = float(leg.get('price', 0))

            # 1. Use existing Greeks if provided (highest priority)
            if all(leg.get(k) is not None for k in ['delta', 'gamma', 'theta']):
                greeks = {'delta': float(leg['delta']), 'gamma': float(leg['gamma']), 'theta': float(leg['theta'])}
            
            # 2. Solve for IV from market price if available
            elif l_price > 0.01:
                iv = self.solve_for_iv(
                    l_price, spot, l_strike, l_expiry, r, q, l_type
                )
                greeks = self.black_scholes_greeks(
                    spot, l_strike, l_expiry, iv, r, q, l_type
                )
            
            # 3. Fallback to constant IV (lowest priority)
            else:
                fallback_iv = float(leg.get('iv') or 0.4)
                greeks = self.black_scholes_greeks(
                    spot, l_strike, l_expiry, fallback_iv, r, q, l_type
                )
            
            net_delta += greeks['delta'] * l_qty
            net_gamma += greeks['gamma'] * l_qty
            net_theta += greeks['theta'] * l_qty
            
        # Unified VAGR (Volatility-Adjusted Gamma Rent) Calculation
        # Derived from BS PDE: Theta = -0.5 * sigma^2 * S^2 * Gamma
        # sigma_be = sqrt( 2 * |Theta| / (S^2 * |Gamma|) )
        
        abs_gamma = abs(net_gamma)
        # Convert daily theta back to annualized for PDE consistency
        abs_theta_annual = abs(net_theta) * 365.0
        
        if abs_gamma > 1e-12:
            sigma_be_sq = (2.0 * abs_theta_annual) / (spot**2 * abs_gamma)
            sigma_be = np.sqrt(max(1e-9, sigma_be_sq))
        else:
            sigma_be = 10.0 # Effectively infinity risk cover for zero gamma
            
        # Calculate Market-Implied Volatility (weighted average of legs or constant)
        # For 0DTE, we use a high-resolution IV. If solving failed, 0.40 is the baseline.
        market_iv = np.mean([float(leg.get('iv') or 0.4) for leg in legs]) if legs else 0.4
        
        # G.A.R. = Market Vol / Break-even Vol
        # If GAR > 1.0, the market expects more movement than your theta rent covers.
        gar = market_iv / sigma_be
        
        return {
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_theta': net_theta,
            'gar': gar,
            'sigma_be': sigma_be,
            'market_iv': market_iv,
            'strikes': [(leg['strike'], leg['qty']) for leg in legs],
            'timestamp': datetime.now().isoformat()
        }

    def calculate_probability_of_touch(self, spot, strike, T, sigma):
        """
        Calculates the probability that the underlying will touch the strike
        before expiration.
        
        Uses the standard deviation distance formula:
        PoT ≈ 2 * N(-|z|) where z = ln(S/K) / (sigma * sqrt(T))
        
        This gives the probability of touching either above or below.
        """
        if T <= 0:
            return 1.0 if abs(spot - strike) < 0.01 else 0.0
            
        # Calculate number of standard deviations away
        z = abs(np.log(spot / strike)) / (sigma * np.sqrt(T))
        
        # Probability of touching = 2 * N(-z) for a one-sided barrier
        # This approaches 0 as the strike gets further OTM
        pot = 2.0 * norm.cdf(-z)
        
        return max(0.0, min(1.0, pot))
    
    def classify_threat(self, gar, pot=1.0):
        """
        Classifies threat based on VAGR (Market Vol / Break-even Vol)
        weighted by Probability of Touch (PoT).
        
        Risk Intensity = GAR * PoT
        
        Level 1 (SEARCH): < 0.5  (Low risk or far OTM)
        Level 2 (LOCK): 0.5 - 1.0 (Moderate risk)
        Level 3 (LAUNCH): >= 1.0 (High risk and near strike)
        """
        risk_intensity = gar * pot
        
        if risk_intensity < 0.5:
            return 'SEARCH', 'Green'
        elif risk_intensity < 1.0:
            return 'LOCK', 'Orange'
        else:
            return 'LAUNCH', 'Red'
