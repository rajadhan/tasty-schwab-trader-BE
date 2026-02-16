import unittest
from brokers.schwab import discover_credit_spreads

class TestSchwabSync(unittest.TestCase):
    def test_discover_credit_spreads(self):
        # Mock positions 
        mock_positions = [
            {
                "longQuantity": 0, "shortQuantity": 1,
                "instrument": {"symbol": "SPX   260215C05000000", "assetType": "OPTION", "underlyingSymbol": "SPX"}
            },
            {
                "longQuantity": 1, "shortQuantity": 0,
                "instrument": {"symbol": "SPX   260215C05100000", "assetType": "OPTION", "underlyingSymbol": "SPX"}
            }
        ]
        
        spreads = discover_credit_spreads(mock_positions)
        self.assertIn("SPX", spreads)
        self.assertEqual(len(spreads["SPX"]), 1)
        self.assertEqual(spreads["SPX"][0]['type'], 'CALL')

if __name__ == '__main__':
    unittest.main()
