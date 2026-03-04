import pytest
import numpy as np
from pre_trade_evaluator import PreTradeEvaluator

@pytest.fixture
def evaluator():
    # Use a lambda penalty of 1.0 for simpler mental math in tests
    return PreTradeEvaluator(lambda_penalty=1.0)

def test_format_price(evaluator):
    """Test format price explicitly enforces strict '[CREDIT] X.XX' and '[DEBIT] X.XX' constraints."""
    assert evaluator.format_price(1.50) == "[CREDIT] 1.50"
    assert evaluator.format_price(0.50) == "[CREDIT] 0.50"
    assert evaluator.format_price(-1.25) == "[DEBIT] 1.25"
    assert evaluator.format_price(-0.05) == "[DEBIT] 0.05"
    assert evaluator.format_price(0.0) == "[EVEN] 0.00"

def test_manageability_score_no_drawdown(evaluator):
    """Test that a P&L profile fully above zero only returns the exact EV (no penalty applied)."""
    ev = 2.0
    prices = np.array([100, 105, 110, 115])
    # P&L is strictly positive, no drawdown
    pnl = np.array([0.5, 1.0, 1.5, 2.0]) 
    
    score = evaluator.calculate_manageability_score(ev, prices, pnl)
    assert np.isclose(score, ev), "Score should equal EV if there's no drawdown penalty."

def test_manageability_score_cliff_penalty(evaluator):
    """Test that steep negative P&L gradients in drawdown zones properly reduce the score."""
    ev = 1.0
    prices = np.array([100, 105, 110, 115])
    # The gradient from 105 -> 110 drops from -1 to -6.
    # dp = 5, dpnl = -5. G = -1.0
    pnl = np.array([2.0, -1.0, -6.0, -8.0])
    
    # Calculate expected penalty manually:
    # G1 (100->105) PNL drops 2 to -1. G = -3/5 = -0.6. Initial node PNL=2 (no drawdown penalty here)
    # G2 (105->110) PNL drops -1 to -6. dpnl = -5. dp=5. G = -1.0. Start node (-1.0) is in drawdown. Penalty = max(0, 1.0) = 1.0
    # G3 (110->115) PNL drops -6 to -8. dpnl = -2. dp=5. G = -0.4. Start node (-6.0) is in drawdown. Penalty = max(0, 0.4) = 0.4
    # Total penalty sum = 1.4
    # Expected M = EV - lambda * sum_penalty = 1.0 - 1.0 * 1.4 = -0.4
    
    score = evaluator.calculate_manageability_score(ev, prices, pnl)
    assert np.isclose(score, -0.4), f"Expected -0.4, got {score}"

def test_monte_carlo_debit_spread(evaluator):
    """Test that the engine successfully evaluates a debit vertical spread math correctly."""
    # Premium sum is -1.50 (Net Debit)
    # Long 95 Call (cost 2.50), Short 105 Call (credit 1.00) => Call Debit Spread
    result = evaluator.evaluate_monte_carlo_jump_diffusion(
        S0=100.0,
        strikes=[95.0, 105.0],
        premiums=[-2.50, 1.00],
        vol=0.15,
        jump_intensity=1.0,
        jump_mean=0.0,
        jump_std=0.01,
        num_paths=1000,
        steps_per_day=50
    )
    assert result is not None, "Should no longer reject net debit configurations."
    assert result["net_premium"] == "[DEBIT] 1.50"
    assert "expected_value" in result
    assert "manageability_score" in result
def test_monte_carlo_valid_credit_spread(evaluator):
    """Test valid Monte Carlo evaluation output dictionary structure for a credit spread."""
    result = evaluator.evaluate_monte_carlo_jump_diffusion(
        S0=100.0,
        strikes=[95.0, 105.0], # E.g. Call Credit Spread (short 95, long 105)
        premiums=[2.50, -1.00], # Net credit = 1.50
        vol=0.15,
        jump_intensity=1.0,
        jump_mean=0.0,
        jump_std=0.01,
        num_paths=1000,
        steps_per_day=50
    )
    assert result is not None
    assert "manageability_score" in result
    assert "expected_value" in result
    assert result["net_premium"] == "[CREDIT] 1.50"

def test_mdp_valid_iron_condor(evaluator):
    """Test valid MDP evaluation output dictionary structure for an Iron Condor."""
    result = evaluator.evaluate_mdp_state_space(
        S0=100.0,
        strikes=[90.0, 95.0, 105.0, 110.0],
        premiums=[-1.00, 3.00, 2.50, -0.50], # Net credit = 4.00
        time_steps=50,
        vol=0.15,
        price_nodes=100
    )
    assert result is not None
    assert "manageability_score" in result
    assert "expected_value" in result
    assert result["net_premium"] == "[CREDIT] 4.00"
