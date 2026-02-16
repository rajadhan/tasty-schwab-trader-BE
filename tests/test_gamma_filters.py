import unittest
import numpy as np
from gamma_rwr_filters import GammaRWRFilters

class TestGammaRWRFilters(unittest.TestCase):
    def setUp(self):
        self.filters = GammaRWRFilters(windows=[1, 5, 10])

    def test_confidence_scoring(self):
        # Case 1: High volume, expanding IV
        self.filters.add_snapshot(gar=1.6, volume_ratio=2.0, iv_delta=0.01)
        conf = self.filters.calculate_confidence()
        self.assertGreaterEqual(conf, 0.7)
        
        # Case 2: Low volume
        self.filters.add_snapshot(gar=1.6, volume_ratio=0.5, iv_delta=0.01)
        conf = self.filters.calculate_confidence()
        self.assertLess(conf, 0.7)

    def test_multi_window_averaging(self):
        for i in range(10):
            self.filters.add_snapshot(gar=float(i), volume_ratio=1.0, iv_delta=0.0)
            
        gar_results = self.filters.get_multi_window_gar()
        self.assertEqual(gar_results['1m'], 9.0)
        self.assertLess(gar_results['10m'], 9.0)

    def test_ejection_logic(self):
        # Trigger persistence
        for _ in range(5):
            self.filters.add_snapshot(gar=2.0, volume_ratio=2.0, iv_delta=0.01)
            gar_results = self.filters.get_multi_window_gar()
            conf = self.filters.calculate_confidence()
            should_eject = self.filters.should_eject(gar_results, conf)
            
        self.assertTrue(should_eject)

if __name__ == '__main__':
    unittest.main()
