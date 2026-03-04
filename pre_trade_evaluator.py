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

    def evaluate_monte_carlo_jump_diffusion(self, S0: float, 
                                            strikes: List[float], 
                                            premiums: List[float],
                                            vol: float, jump_intensity: float, 
                                            jump_mean: float, jump_std: float, 
                                            time_to_expiry_days: float = 1/365.25, 
                                            num_paths: int = 10000, 
                                            steps_per_day: int = 390) -> Optional[Dict]:
        """
        Mode 1: Monte Carlo (Jump-Diffusion) Evaluation
        Simulates intraday price paths using a Jump-Diffusion model to account for sudden
        violent intraday moves. Evaluates an Iron Condor (or single spread) against these paths.
        
        :param strikes: List of 4 strikes for an Iron Condor/Butterfly [Put, Put, Call, Call]
                        or 2 strikes for a vertical spread.
        :param premiums: List of prices paid/received for the legs in the same order as strikes. 
                         (Positive for credit/selling, Negative for debit/buying).
        :return: Dictionary containing the manageability score and expected value.
        """
        logger.info(f"Starting Mode 1 (Monte Carlo Jump-Diffusion) for S0={S0:.2f} | Paths={num_paths}")
        
        net_premium = sum(premiums)
        logger.info(f"Spread successfully constructed at {self.format_price(net_premium)}.")

        # Define simulation parameters
        dt = time_to_expiry_days / steps_per_day
        paths = np.zeros((num_paths, steps_per_day + 1))
        paths[:, 0] = S0
        
        # Vectorized path generation (Jump-Diffusion)
        for t in range(1, steps_per_day + 1):
            z = np.random.standard_normal(num_paths)
            poisson_jumps = np.random.poisson(jump_intensity * dt, num_paths)
            jump_sizes = np.random.normal(jump_mean, jump_std, num_paths) * poisson_jumps
            
            # Merton Jump-Diffusion geometric step (simplified risk-neutral drift r=0 for intraday)
            drift = -0.5 * (vol**2) * dt
            diffusion = vol * np.sqrt(dt) * z
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion + jump_sizes)
            
        final_prices = paths[:, -1]
        
        # Calculate Payoff at Expiration
        directions = [-1.0 if p > 0 else 1.0 for p in premiums]
        
        if len(strikes) == 4: # Iron Condor/Butterfly: Put, Put, Call, Call
            p1, p2, c1, c2 = strikes
            payoffs = net_premium \
                      + directions[0] * np.maximum(0, p1 - final_prices) \
                      + directions[1] * np.maximum(0, p2 - final_prices) \
                      + directions[2] * np.maximum(0, final_prices - c1) \
                      + directions[3] * np.maximum(0, final_prices - c2)
            
        elif len(strikes) == 2: # Vertical Spread
            s1, s2 = strikes
            is_call = (s1 + s2) / 2.0 >= S0
            if is_call:
                payoffs = net_premium \
                          + directions[0] * np.maximum(0, final_prices - s1) \
                          + directions[1] * np.maximum(0, final_prices - s2)
            else:
                payoffs = net_premium \
                          + directions[0] * np.maximum(0, s1 - final_prices) \
                          + directions[1] * np.maximum(0, s2 - final_prices)
        else:
            payoffs = np.zeros(num_paths)
        
        # Expected Value (Mean of all simulated path payoffs)
        expected_value = float(np.mean(payoffs))
        
        # To calculate Manageability Score, we need a deterministic P&L curve across a price range
        sorted_indices = np.argsort(final_prices)
        sorted_prices = final_prices[sorted_indices]
        
        # Smooth the P&L curve slightly for gradient calculation
        percentiles = np.linspace(1, 99, 200)
        p_prices = np.asarray(np.percentile(sorted_prices, percentiles))
        
        # Calculate deterministic payoff for these percentiled prices to get a clean gradient
        if len(strikes) == 4:
            p1, p2, c1, c2 = strikes
            p_payoffs = net_premium \
                        + directions[0] * np.maximum(0, p1 - p_prices) \
                        + directions[1] * np.maximum(0, p2 - p_prices) \
                        + directions[2] * np.maximum(0, p_prices - c1) \
                        + directions[3] * np.maximum(0, p_prices - c2)
        elif len(strikes) == 2:
            s1, s2 = strikes
            is_call = (s1 + s2) / 2.0 >= S0
            if is_call:
                p_payoffs = net_premium \
                            + directions[0] * np.maximum(0, p_prices - s1) \
                            + directions[1] * np.maximum(0, p_prices - s2)
            else:
                p_payoffs = net_premium \
                            + directions[0] * np.maximum(0, s1 - p_prices) \
                            + directions[1] * np.maximum(0, s2 - p_prices)
        else:
            p_payoffs = np.zeros_like(p_prices)
            
        manageability_score = self.calculate_manageability_score(expected_value, p_prices, p_payoffs)
        
        return {
            "mode": "Monte Carlo (Jump-Diffusion)",
            "paths_simulated": num_paths,
            "net_premium": self.format_price(net_premium),
            "expected_value": self.format_price(expected_value),
            "manageability_score": round(manageability_score, 4),
            "win_rate": f"{(np.sum(payoffs > 0) / num_paths) * 100:.1f}%"
        }

    def evaluate_mdp_state_space(self, S0: float, 
                                 strikes: List[float], 
                                 premiums: List[float],
                                 time_steps: int, 
                                 vol: float,
                                 price_nodes: int = 50, 
                                 time_to_expiry_days: float = 1/365.25) -> Optional[Dict]:
        """
        Mode 2: MDP State-Space Evaluation
        Constructs a static, multi-dimensional grid (Time Remaining, Price).
        Uses geometric random walk probabilities to calculate the Expected Value (EV) of the spread at T=0
        and measures the fragility to immediate price shocks via EV gradients.
        
        :return: Dictionary containing the manageability score and expected value, or None if net debit.
        """
        logger.info(f"Starting Mode 2 (MDP State-Space) Grid: {time_steps}T x {price_nodes}P")
        
        net_premium = sum(premiums)
        logger.info(f"Initial Grid Boundary constructed at {self.format_price(net_premium)}.")

        # Simplified 1D State Space Grid for Intraday (Price x Time)
        dt = time_to_expiry_days / time_steps
        
        # Define price bounds (e.g. +/- 3 standard deviations)
        std_dev = vol * np.sqrt(time_to_expiry_days)
        S_min = S0 * np.exp(-3 * std_dev)
        S_max = S0 * np.exp(3 * std_dev)
        prices = np.linspace(S_min, S_max, price_nodes)
        
        # Transition Probabilities (Simplified Trinomial approximations for dt drift/diffusion)
        # In a full QuantLib implementation, we would extract transition probabilities from the tree.
        # Here we use an explicit finite difference approximation equivalent.
        dx = np.log(prices[1] / prices[0]) if price_nodes > 1 else 1.0
        
        # Probabilities for Up, Middle, Down movements
        pu = 0.5 * (vol**2 * dt / dx**2 + (0.0 - 0.5 * vol**2) * dt / dx)
        pd = 0.5 * (vol**2 * dt / dx**2 - (0.0 - 0.5 * vol**2) * dt / dx)
        pm = 1.0 - pu - pd
        
        # Ensure numerical stability
        pu = max(0.0, min(1.0, pu))
        pd = max(0.0, min(1.0, pd))
        pm = max(0.0, 1.0 - pu - pd)

        # Initialize Terminal Payoff State at Expiration T=N
        V = np.zeros(price_nodes)
        directions = [-1.0 if p > 0 else 1.0 for p in premiums]
        
        if len(strikes) == 4:
            p1, p2, c1, c2 = strikes
            V = net_premium \
                + directions[0] * np.maximum(0, p1 - prices) \
                + directions[1] * np.maximum(0, p2 - prices) \
                + directions[2] * np.maximum(0, prices - c1) \
                + directions[3] * np.maximum(0, prices - c2)
        elif len(strikes) == 2:
            s1, s2 = strikes
            is_call = (s1 + s2) / 2.0 >= S0
            if is_call:
                V = net_premium \
                    + directions[0] * np.maximum(0, prices - s1) \
                    + directions[1] * np.maximum(0, prices - s2)
            else:
                V = net_premium \
                    + directions[0] * np.maximum(0, s1 - prices) \
                    + directions[1] * np.maximum(0, s2 - prices)

        # Backward Induction Step
        # Roll back from T=N to T=0
        for _ in range(time_steps - 1, -1, -1):
            V_new = np.zeros(price_nodes)
            # Boundary conditions (Dirichlet)
            V_new[0] = V[0]    # Absorption at lower bound
            V_new[-1] = V[-1]  # Absorption at upper bound
            
            # Interior nodes transition
            V_new[1:-1] = pu * V[2:] + pm * V[1:-1] + pd * V[:-2]
            V = V_new
        
        # V now represents the Expected Value vector at T=0 across the price state space
        
        # Interpolate the EV at exactly S0 for the main output
        expected_value_t0 = float(np.interp(S0, prices, V))
        
        # Map EV gradient across adjacent immediate price nodes to calculate manageability.
        # Fragility = d(EV) / d(Price)
        # Using the full state space prices and V vectors is perfect for our custom manageability function
        m_score = self.calculate_manageability_score(expected_value_t0, prices, V)
        
        return {
            "mode": "MDP State-Space Evaluation",
            "grid_dimensions": f"{time_steps}x{price_nodes}",
            "net_premium": self.format_price(net_premium),
            "expected_value": self.format_price(expected_value_t0),
            "manageability_score": round(m_score, 4)
        }

if __name__ == "__main__":
    # Example Usage demonstrating the framework constraints and output formats.
    evaluator = PreTradeEvaluator(lambda_penalty=1.5)
    
    # Run Mode 1 (Iron Condor example: Long Put 510, Short Put 515, Short Call 525, Long Call 530)
    logger.info("--- Testing Mode 1: Monte Carlo ---")
    mc_results = evaluator.evaluate_monte_carlo_jump_diffusion(
        S0=520.0, 
        strikes=[510.0, 515.0, 525.0, 530.0], 
        premiums=[-1.20, 3.50, 2.80, -0.90], # Net Credit = 4.20
        vol=0.15, 
        jump_intensity=2.0, 
        jump_mean=0.0, 
        jump_std=0.02
    )
    if mc_results:
        logger.info(f"MC Results: {mc_results}")
        
    # Run Mode 2
    logger.info("--- Testing Mode 2: MDP State-Space ---")
    mdp_results = evaluator.evaluate_mdp_state_space(
        S0=520.0, 
        strikes=[510.0, 515.0, 525.0, 530.0], 
        premiums=[-1.20, 3.50, 2.80, -0.90],
        time_steps=390,
        vol=0.15
    )
    if mdp_results:
        logger.info(f"MDP Results: {mdp_results}")
