import unittest
import numpy as np
from gamma_rwr_engine import GammaRWREngine

class TestGammaRWREngine(unittest.TestCase):
    def setUp(self):
        self.engine = GammaRWREngine()

    def test_black_scholes_greeks(self):
        # Test basic call Greeks
        S = 100
        K = 100
        T = 1/365  # 1 day
        v = 0.2    # 20% IV
        r = 0.05
        q = 0.0
        
        greeks = self.engine.black_scholes_greeks(S, K, T, v, r, q, 'call')
        
        self.assertGreater(greeks['delta'], 0.4)  # ATM delta ~0.5
        self.assertLess(greeks['delta'], 0.6)
        self.assertGreater(greeks['gamma'], 0)
        self.assertLess(greeks['theta'], 0)  # Theta should be negative for long call

    def test_position_netting(self):
        # 100/105 Call Credit Spread (Short 100C, Long 105C)
        spot = 95
        legs = [
            {'strike': 100, 'iv': 0.2, 'expiry_years': 1/365, 'qty': -1, 'type': 'call'},
            {'strike': 105, 'iv': 0.2, 'expiry_years': 1/365, 'qty': 1, 'type': 'call'}
        ]
        
        net_greeks = self.engine.calculate_position_greeks(spot, legs)
        
        # Net gamma should be positive for a credit spread? 
        # Actually short the 100C (higher gamma) and long the 105C (lower gamma)
        # So net gamma is negative (short gamma).
        self.assertLess(net_greeks['net_gamma'], 0)
        self.assertIn('gar', net_greeks)

    def test_threat_classification(self):
        self.assertEqual(self.engine.classify_threat(0.2)[0], 'SEARCH')
        self.assertEqual(self.engine.classify_threat(0.8)[0], 'LOCK')
        self.assertEqual(self.engine.classify_threat(2.0)[0], 'LAUNCH')

if __name__ == '__main__':
    unittest.main()
