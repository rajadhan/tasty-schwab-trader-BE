import numpy as np
import QuantLib as ql
import logging
from typing import Dict, Tuple, List, Optional

# Configure module-level logger
logger = logging.getLogger("PreTradeEvaluator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

class PreTradeEvaluator:
    """
    Standalone pre-trade options strategy evaluation engine utilizing QuantLib and numpy.
    Designed exclusively for pre-trade 0DTE spread evaluation and structural edge scoring.
    """
    
    def __init__(self, lambda_penalty: float = 1.0):
        """
        Initializes the evaluation engine.
        
        :param lambda_penalty: High weight to strictly penalize cliff edges (steep negative P&L gradients).
        """
        self.lambda_penalty = lambda_penalty

    @staticmethod
    def format_price(value: float) -> str:
        """
        Formats price outputs using absolute float values with explicit tags.
        Strictly forbids negative signs to represent spread prices or credits/debits.
        
        :param value: The net price value (positive for credit received, negative for debit paid).
        :return: Formatted string like "[CREDIT] 1.50" or "[DEBIT] 0.50".
        """
        if value > 0:
            return f"[CREDIT] {abs(value):.2f}"
        elif value < 0:
            return f"[DEBIT] {abs(value):.2f}"
        else:
            return f"[EVEN] 0.00"

    def calculate_manageability_score(self, ev: float, prices: np.ndarray, pnl: np.ndarray) -> float:
        """
        The Manageability Score: M = EV - lambda * sum(max(0, -G_local))
        Penalizes steep negative P&L gradients (escalating losses) in active drawdown zones.
        
        :param ev: Expected Value of the spread.
        :param prices: 1D numpy array of sequential price nodes in the state space.
        :param pnl: 1D numpy array of the spread's Profit & Loss at the corresponding price nodes.
        :return: The Manageability Score M.
        """
        if len(prices) < 2 or len(pnl) < 2:
            return ev

        # Calculate localized gradient G = d(P&L) / d(price)
        dp = np.diff(prices)
        dpnl = np.diff(pnl)
        
        # Avoid division by zero in flat price intervals
        with np.errstate(divide='ignore', invalid='ignore'):
            G = np.where(dp != 0, dpnl / dp, 0.0)
        
        # Identify drawdown zones: regions where the localized P&L curve dips below 0
        # We index [:-1] to align with the gradient array shape
        drawdown_mask = pnl[:-1] < 0 
        
        # Calculate the penalty: Sum of max(0, -G_local) for states in a drawdown.
        # This explicitly punishes gradients where losses accelerate rapidly (-G is positive).
        penalty_sum = np.sum(np.maximum(0, -G[drawdown_mask]))
        
        manageability_score = ev - (self.lambda_penalty * penalty_sum)
        return float(manageability_score)

    def evaluate_monte_carlo_jump_diffusion(self, S0: float, strikes: List[float], 
                                            vol: float, jump_intensity: float, 
                                            jump_mean: float, jump_std: float, 
                                            time_to_expiry_days: float = 1/365.25, 
                                            num_paths: int = 10000, 
                                            steps_per_day: int = 390) -> Optional[Dict]:
        """
        Mode 1: Monte Carlo (Jump-Diffusion) Evaluation
        Simulates 10,000 intraday price paths using a Jump-Diffusion model to account for sudden
        violent intraday moves followed by range-bound drift.
        
        :return: Dictionary containing the manageability score and expected value, or None if net debit.
        """
        logger.info(f"Starting Mode 1 (Monte Carlo Jump-Diffusion) for S0={S0:.2f} | Paths={num_paths}")
        
        # Define simulation parameters
        dt = time_to_expiry_days / steps_per_day
        paths = np.zeros((num_paths, steps_per_day + 1))
        paths[:, 0] = S0
        
        # Vectorized path generation
        for t in range(1, steps_per_day + 1):
            # Standard normally distributed random numbers
            z = np.random.standard_normal(num_paths)
            
            # Poisson jump occurrence
            poisson_jumps = np.random.poisson(jump_intensity * dt, num_paths)
            
            # Normally distributed jump sizes
            jump_sizes = np.random.normal(jump_mean, jump_std, num_paths) * poisson_jumps
            
            # Merton Jump-Diffusion geometric step (simplified risk-neutral drift r=0 for intraday zero-interest)
            drift = -0.5 * (vol**2) * dt
            diffusion = vol * np.sqrt(dt) * z
            
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion + jump_sizes)
            
        # -- Placeholder for spread generation logic --
        # We must strictly construct a spread that yields a net credit.
        mock_net_credit = 0.85 # Assumed Net Credit collected
        if mock_net_credit <= 0:
            logger.warning(f"Spread construction yields {self.format_price(mock_net_credit)}. Rejecting spread (Net Debit).")
            return None # Strict Business Logic Constraint: Reject Net Debits
            
        logger.info(f"Spread successfully constructed at {self.format_price(mock_net_credit)}.")

        # -- Mock EV and manageability mapping -- 
        # In a real run, we would map the final paths against the payoff profile, calculate mean P&L for EV,
        # and then map the localized price nodes to the exact calculate_manageability_score() function.
        
        expected_value = 1.25 # Mock positive EV (Net Credit realized + decay)
        
        # Mocking an arbitrary localized price array and P&L array for the manageability calculation
        simulated_prices = np.linspace(S0 * 0.95, S0 * 1.05, 100)
        simulated_pnl = expected_value - np.maximum(0, simulated_prices - strikes[0]) # Arbitrary short call payoff shape
        
        manageability_score = self.calculate_manageability_score(expected_value, simulated_prices, simulated_pnl)
        
        return {
            "mode": "Monte Carlo (Jump-Diffusion)",
            "paths_simulated": num_paths,
            "net_premium": self.format_price(mock_net_credit),
            "expected_value": self.format_price(expected_value),
            "manageability_score": round(manageability_score, 4)
        }

    def evaluate_mdp_state_space(self, S0: float, time_steps: int, 
                                 price_nodes: int = 50, iv_nodes: int = 10) -> Optional[Dict]:
        """
        Mode 2: MDP State-Space Evaluation
        Constructs a static, multi-dimensional grid (Time Remaining, Price, IV).
        Uses backward induction to calculate the Expected Value (EV) of the spread at T=0
        and measures the fragility to immediate price shocks via EV gradients.
        """
        logger.info(f"Starting Mode 2 (MDP State-Space) Grid: {time_steps}T x {price_nodes}P x {iv_nodes}V")
        
        # Form grid dimensions: S = (Time, Price, IV)
        # Using QuantLib finite difference / trinomial trees or explicit ND grid solvers goes here.
        # ...
        
        # Strict business logic: Ensure the transition matrix starts from a state of net credit.
        mock_net_credit = 1.15
        if mock_net_credit <= 0:
            logger.warning(f"Initial State Space boundary evaluates to {self.format_price(mock_net_credit)}. Rejecting spread.")
            return None
            
        logger.info(f"Initial Grid Boundary constructed at {self.format_price(mock_net_credit)}.")

        # Backward Induction Step (Mock logic)
        # T=N (Expiration) payoff is deterministic.
        # Step backward from T=N to T=0 applying transition probabilities & discounting.
        
        expected_value_t0 = 1.05 # Mocked resulting EV at T=0
        
        # Map EV gradient across adjacent immediate price nodes to calculate manageability.
        # fragility = d(EV) / d(Price)
        mock_prices = np.linspace(S0 - 10, S0 + 10, price_nodes)
        mock_ev_landscape = expected_value_t0 * np.sin(np.linspace(0, np.pi, price_nodes)) # Arbitrary landscape shape
        
        m_score = self.calculate_manageability_score(expected_value_t0, mock_prices, mock_ev_landscape)
        
        return {
            "mode": "MDP State-Space Evaluation",
            "grid_dimensions": f"{time_steps}x{price_nodes}x{iv_nodes}",
            "net_premium": self.format_price(mock_net_credit),
            "expected_value": self.format_price(expected_value_t0),
            "manageability_score": round(m_score, 4)
        }

if __name__ == "__main__":
    # Example Usage demonstrating the framework constraints and output formats.
    evaluator = PreTradeEvaluator(lambda_penalty=1.5)
    
    # Run Mode 1
    mc_results = evaluator.evaluate_monte_carlo_jump_diffusion(
        S0=520.0, 
        strikes=[525.0, 530.0], 
        vol=0.15, 
        jump_intensity=2.0, 
        jump_mean=0.0, 
        jump_std=0.02
    )
    if mc_results:
        logger.info(f"MC Results: {mc_results}")
        
    # Run Mode 2
    mdp_results = evaluator.evaluate_mdp_state_space(S0=520.0, time_steps=390)
    if mdp_results:
        logger.info(f"MDP Results: {mdp_results}")
