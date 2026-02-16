import time
import logging

class RWRAlertManager:
    """
    Manages RWR alerts, HUD rendering, and future SMS/External notifications.
    """
    
    def __init__(self, ticker, logger=None):
        self.ticker = ticker
        self.logger = logger or logging.getLogger(ticker)
        self.last_alert_time = 0
        self.alert_threshold_seconds = 60  # Rate limit alerts
        self.current_threat_level = 'SEARCH'

    def render_hud(self, gar_results, confidence, threat_level, timestamp=None, spot=0, greeks=None):
        """
        Prints a high-visibility ASCII HUD to the terminal.
        """
        color_code = {
            'SEARCH': '\033[92m', # Green
            'LOCK': '\033[93m',   # Orange/Yellow
            'LAUNCH': '\033[91m'  # Red
        }.get(threat_level, '\033[0m')
        
        reset = '\033[0m'
        display_time = timestamp if timestamp else time.strftime('%H:%M:%S')
        
        greeks_str = f"D: {greeks.get('net_delta', 0):.4f} | G: {greeks.get('net_gamma', 0):.4f} | T: {greeks.get('net_theta', 0):.4f}" if greeks else "N/A"
        
        # Calculate distance to strike if possible
        dist_str = ""
        if greeks and 'strikes' in greeks:
            short_strikes = [s for s, q in greeks['strikes'] if q < 0]
            if short_strikes:
                nearest = min(abs(spot - k) for k in short_strikes)
                dist_str = f"\nDIST TO STRIKE: {nearest:.2f}"

        banner = f"""
{color_code}================================================================
>>> RWR THREAT STATUS: {threat_level} [{self.ticker}] <<<
================================================================
SPOT PRICE: {spot:.2f} {dist_str}
NET GREEKS: {greeks_str}
GA.R. WINDOWS: {gar_results}
CONFIDENCE: {confidence * 100}%
TIME: {display_time}
----------------------------------------------------------------{reset}
"""
        print(banner)
        self.current_threat_level = threat_level

    def trigger_alert(self, threat_level, message):
        """
        Triggers a high-priority alert (Terminal for Iteration 1).
        """
        now = time.time()
        
        if threat_level == 'LAUNCH' and (now - self.last_alert_time > self.alert_threshold_seconds):
            self.last_alert_time = now
            
            # High-visibility terminal alert
            alert_banner = f"""
\033[91m
[!!!] MISSILE WARNING: GAMMA RAMPS DETECTED [!!!]
STATION: {self.ticker}
ACTION: EJECT / MARKET CLOSE ALL
MESSAGE: {message}
\033[0m
"""
            print(alert_banner)
            self.logger.critical(f"RWR LAUNCH ALERT: {message}")
            
            # Future SMS integration point
            self._send_to_external_providers(message)

    def _send_to_external_providers(self, message):
        """
        Placeholder for future SMS/Twilio integration.
        """
        # Iteration 2: loop through providers like TwilioProvider, SNSProvider
        pass
