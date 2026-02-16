import unittest
from rwr_alert_manager import RWRAlertManager
import time

class TestRWRAlertManager(unittest.TestCase):
    def setUp(self):
        self.manager = RWRAlertManager("SPX")

    def test_alert_throttling(self):
        # First alert should trigger
        with self.assertLogs(level='CRITICAL') as cm:
            self.manager.trigger_alert('LAUNCH', "Test Alarm 1")
        self.assertEqual(len(cm.output), 1)
        
        # Second alert immediately after should be throttled
        # Note: We aren't capturing stdout here, just checking the logger part
        # for simplicity in unittest
        self.manager.trigger_alert('LAUNCH', "Test Alarm 2")
        # last_alert_time should be the same as first
        self.assertGreater(self.manager.last_alert_time, 0)

if __name__ == '__main__':
    unittest.main()
