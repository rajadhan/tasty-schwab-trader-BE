import numpy as np
from collections import deque
from datetime import datetime

class GammaRWRFilters:
    """
    ECCM (False Positive Mitigation) and Multi-Horizon Radar.
    Handles volume-weighted confidence and rolling window analysis.
    """
    
    def __init__(self, windows=[1, 10, 30, 60]):
        # Rolling buffers for different time windows (in minutes)
        self.windows = sorted(windows)
        self.max_window = max(self.windows)
        # Buffer stores (timestamp, gar, volume_ratio, iv_delta)
        self.buffer = deque(maxlen=self.max_window + 5)
        self.persistence_threshold = 3  # Ticks required in Red level
        self.launch_counter = 0

    def add_snapshot(self, gar, volume_ratio, iv_delta):
        """
        Adds a new 1-minute snapshot to the rolling buffer.
        """
        self.buffer.append({
            'timestamp': datetime.now(),
            'gar': gar,
            'volume_ratio': volume_ratio,
            'iv_delta': iv_delta
        })

    def calculate_confidence(self):
        """
        Bayesian Confidence Model (0.0 - 1.0).
        Based on Volume Persistence and IV Correlation.
        """
        if not self.buffer:
            return 0.0
            
        latest = self.buffer[-1]
        
        # 1. Volume Filter (Current Vol > 1.5x MA)
        # We assume volume_ratio = current_vol / avg_vol
        vol_score = min(1.0, latest['volume_ratio'] / 1.5)
        
        # 2. Persistence (Must be in level for >3 ticks)
        # This is handled by the persistence_counter in the main loop
        
        # 3. IV Correlation (IV expanding moves closer to strikes)
        # Positive iv_delta when IV is expanding
        iv_score = 1.0 if latest['iv_delta'] > 0 else 0.5
        
        # Weighted combination
        confidence = (vol_score * 0.7) + (iv_score * 0.3)
        return round(confidence, 2)

    def get_multi_window_gar(self):
        """
        Returns average G.A.R. across all configured windows.
        """
        if not self.buffer:
            return {}
            
        results = {}
        for win in self.windows:
            slice_len = min(len(self.buffer), win)
            data_slice = list(self.buffer)[-slice_len:]
            avg_gar = np.mean([d['gar'] for d in data_slice])
            results[f'{win}m'] = round(avg_gar, 2)
            
        return results

    def should_eject(self, gar_results, confidence):
        """
        Final signal validation logic.
        """
        # LAUNCH level requires G.A.R >= 1.5 in multiple windows OR 1m spike
        # AND confidence >= 0.70
        
        is_high_gamma = any(v >= 1.5 for v in gar_results.values())
        is_confident = confidence >= 0.70
        
        if is_high_gamma and is_confident:
            self.launch_counter += 1
        else:
            self.launch_counter = 0
            
        return self.launch_counter >= self.persistence_threshold
