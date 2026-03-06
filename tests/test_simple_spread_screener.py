import pytest # type: ignore
from simple_spread_screener import StrategyPermutator

@pytest.fixture
def mock_chain():
    """Returns a mock simplified option chain for SPX."""
    return [
        {"strike": 500.0, "call_bid": 25.0, "call_ask": 25.5, "put_bid": 1.0, "put_ask": 1.2},
        {"strike": 505.0, "call_bid": 20.0, "call_ask": 20.5, "put_bid": 2.0, "put_ask": 2.2},
        {"strike": 510.0, "call_bid": 15.0, "call_ask": 15.5, "put_bid": 5.0, "put_ask": 5.2},
        {"strike": 515.0, "call_bid": 10.0, "call_ask": 10.5, "put_bid": 10.0, "put_ask": 10.5}, # ATM
        {"strike": 520.0, "call_bid": 5.0, "call_ask": 5.2, "put_bid": 15.0, "put_ask": 15.5},
        {"strike": 525.0, "call_bid": 2.0, "call_ask": 2.2, "put_bid": 20.0, "put_ask": 20.5},
        {"strike": 530.0, "call_bid": 1.0, "call_ask": 1.2, "put_bid": 25.0, "put_ask": 25.5},
    ]

def test_permutator_initialization(mock_chain):
    """Test that the permutator correctly maps the chain."""
    permutator = StrategyPermutator(mock_chain)
    assert len(permutator.strikes) == 7
    assert permutator.strikes[0] == 500.0
    assert permutator.strikes[-1] == 530.0

def test_get_leg_price(mock_chain):
    """Test the bid/ask natural pricing mechanics."""
    permutator = StrategyPermutator(mock_chain)
    
    # Sell Call -> Hits Bid
    assert permutator.get_leg_price(520.0, "call", "sell") == 5.0
    
    # Buy Call -> Hits Ask
    assert permutator.get_leg_price(520.0, "call", "buy") == -5.2
    
    # Sell Put -> Hits Bid
    assert permutator.get_leg_price(510.0, "put", "sell") == 5.0
    
    # Buy Put -> Hits Ask
    assert permutator.get_leg_price(510.0, "put", "buy") == -5.2

def test_find_vertical_spreads_credit(mock_chain):
    """Test correctly identifying Credit Spreads."""
    permutator = StrategyPermutator(mock_chain)
    
    # Bear Call Spread (Short Call lower strike, Long Call higher strike)
    # e.g., Short 520, Long 525 -> Width 5
    call_spreads = permutator.find_vertical_spreads("call", "credit", allowed_widths=[5.0])
    
    assert len(call_spreads) > 0
    # Let's find the 520/525 spread
    spread_520 = next((s for s in call_spreads if s["strikes"] == [520.0, 525.0]), None)
    assert spread_520 is not None
    assert spread_520["premiums"] == [5.0, -2.2] # Sell 520 (Bid 5.0), Buy 525 (Ask 2.2)
    assert sum(spread_520["premiums"]) == 2.8 # Net Credit

    # Bull Put Spread (Short Put higher strike, Long Put lower strike)
    put_spreads = permutator.find_vertical_spreads("put", "credit", allowed_widths=[5.0])
    spread_510 = next((s for s in put_spreads if s["strikes"] == [510.0, 505.0]), None)
    assert spread_510 is not None
    assert spread_510["premiums"] == [5.0, -2.2] # Sell 510 (Bid 5.0), Buy 505 (Ask 2.2)
    assert sum(spread_510["premiums"]) == 2.8 # Net Credit

def test_find_vertical_spreads_debit(mock_chain):
    """Test correctly identifying Debit Spreads."""
    permutator = StrategyPermutator(mock_chain)
    
    # Bull Call Spread (Buy lower strike, Sell higher strike) -> Built from permutator's "short strike" definition
    # Our permutator logic: option_type="call", spread_type="debit"
    # long_strike = short_strike - width
    # Example: Short 525, Long 520
    call_spreads = permutator.find_vertical_spreads("call", "debit", allowed_widths=[5.0])
    spread_525 = next((s for s in call_spreads if s["strikes"] == [525.0, 520.0]), None)
    assert spread_525 is not None
    assert spread_525["premiums"] == [2.0, -5.2] # Sell 525 (Bid 2.0), Buy 520 (Ask 5.2)
    assert sum(spread_525["premiums"]) == -3.2 # Net Debit

def test_find_iron_condors(mock_chain):
    """Test combining verticals into Iron Condors."""
    permutator = StrategyPermutator(mock_chain)
    
    condors = permutator.find_iron_condors(allowed_widths=[5.0], min_wing_distance=10.0)
    assert len(condors) > 0
    
    # Find a specific condor e.g., Short Call 520, Short Put 510
    target_condor = next((c for c in condors 
                          if c["strikes"][1] == 510.0 and c["strikes"][2] == 520.0), None)
    
    assert target_condor is not None
    # PreTrade Evaluator expects strikes: [Long Put, Short Put, Short Call, Long Call]
    assert target_condor["strikes"] == [505.0, 510.0, 520.0, 525.0]
    
    # Premiums: [Long Put, Short Put, Short Call, Long Call]
    # Long Put 505: Buy = -2.2
    # Short Put 510: Sell = 5.0
    # Short Call 520: Sell = 5.0
    # Long Call 525: Buy = -2.2
    assert target_condor["premiums"] == [-2.2, 5.0, 5.0, -2.2]
    assert round(sum(target_condor["premiums"]), 2) == 5.6 # Net Credit

def test_find_long_butterflies(mock_chain):
    permutator = StrategyPermutator(mock_chain)
    butterflies = permutator.find_butterflies("call", allowed_widths=[5.0])
    
    # E.g. Body at 520: Long 515, Short 2x 520, Long 525
    fly_520 = next((b for b in butterflies if b["strikes"][1] == 520.0), None)
    assert fly_520 is not None
    assert fly_520["strikes"] == [515.0, 520.0, 520.0, 525.0]
    # Premiums: 
    # Long 515 Call (Ask 10.5) -> -10.5
    # Short 520 Call x2 (Bid 5.0) -> +10.0
    # Long 525 Call (Ask 2.2) -> -2.2
    assert fly_520["premiums"] == [-10.5, 5.0, 5.0, -2.2]
    assert round(sum(fly_520["premiums"]), 2) == -2.7 # Net Debit

def test_find_iron_butterflies(mock_chain):
    permutator = StrategyPermutator(mock_chain)
    iron_flies = permutator.find_iron_butterflies(allowed_widths=[5.0])
    
    # Look for ATM Iron Butterfly at 515 body
    # Strikes: [Long Put (510), Short Put (515), Short Call (515), Long Call (520)]
    fly_515 = next((b for b in iron_flies if b["strikes"][1] == 515.0), None)
    assert fly_515 is not None
    assert fly_515["strikes"] == [510.0, 515.0, 515.0, 520.0]
    assert fly_515["premiums"] == [-5.2, 10.0, 10.0, -5.2]
    assert round(sum(fly_515["premiums"]), 2) == 9.6 # Net Credit

def test_find_broken_wing_butterflies(mock_chain):
    permutator = StrategyPermutator(mock_chain)
    
    # Test a Call BWB: e.g. Long ITM, Short 2 ATM, Long far OTM
    bwbs = permutator.find_broken_wing_butterflies("call", allowed_narrow_widths=[5.0], allowed_wide_widths=[10.0])
    
    # Body at 515. Narrow wing (5.0) down -> 510. Wide wing (10.0) up -> 525.
    bwb_515_call = next((b for b in bwbs if b["strikes"][1] == 515.0), None)
    
    assert bwb_515_call is not None
    assert bwb_515_call["strikes"] == [510.0, 515.0, 515.0, 525.0]
    # Long 510 Call (Ask 15.5) -> -15.5
    # Short 2x 515 Call (Bid 10.0) -> +20.0
    # Long 525 Call (Ask 2.2) -> -2.2
    assert bwb_515_call["premiums"] == [-15.5, 10.0, 10.0, -2.2]
    assert round(sum(bwb_515_call["premiums"]), 2) == 2.3 # Net Credit
