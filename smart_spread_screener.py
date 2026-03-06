import os
import json
import logging
from typing import List, Dict
from public_chain_service import PublicChainService
from simple_spread_screener import StrategyPermutator
from pre_trade_evaluator import PreTradeEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartScreener")

class SmartSpreadScreener:
    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        self.public_svc = PublicChainService()
        self.evaluator = PreTradeEvaluator(lambda_penalty=1.5)
        self.permutator = None
        self.s0 = 0.0
        self.atm_vol = 0.15  # Default estimate, can be refined

    def run(self, target_expiration_index: int = 0):
        print(f"\n🚀 --- Starting Smart Spread Screener for {self.symbol} ---\n")
        
        # 1. Authenticate and Get Market Context
        if not self.public_svc.authenticate():
            logger.error("Authentication failed.")
            return

        print(f"🔍 Fetching market context for {self.symbol}...")
        inst_data = self.public_svc.get_instrument(self.symbol)
        if inst_data and "last" in inst_data:
            self.s0 = float(inst_data["last"])
        else:
            # Fallback for SPY or similar if last not in instrument data directly
            # We'll try to get it from the chain later if needed
            pass

        # 2. Get Expirations
        exps = self.public_svc.get_expirations(self.symbol)
        if not exps:
            logger.error("No expirations found.")
            return
        
        target_exp = exps[target_expiration_index]
        print(f"📅 Target Expiration: {target_exp}")

        # 3. Fetch and Format Option Chain
        formatted_chain = self.public_svc.get_formatted_chain(self.symbol, target_exp)
        if not formatted_chain:
            logger.error("Failed to retrieve or format the option chain.")
            return

        # If s0 still not set, use the strike closest to mid-point
        if self.s0 == 0.0:
            self.s0 = sum(s['strike'] for s in formatted_chain) / len(formatted_chain)

        print(f"📈 Spot Price (S0): ${self.s0:.2f}")
        print(f"⛓️  Retrieved {len(formatted_chain)} strikes.")

        # 4. Generate Candidates
        self.permutator = StrategyPermutator(formatted_chain)
        print("\n🛠️  Generating candidate strategies...")
        
        pools = {
            "Bull Put Spreads": self.permutator.find_vertical_spreads("put", "credit", [5.0, 10.0]),
            "Bear Call Spreads": self.permutator.find_vertical_spreads("call", "credit", [5.0, 10.0]),
            "Iron Condors": self.permutator.find_iron_condors([5.0], 10.0),
            "Iron Butterflies": self.permutator.find_iron_butterflies([5.0])
        }

        # 5. Rank Candidates using PreTradeEvaluator
        print("🧠 Evaluating and ranking by manageability score...")
        
        results = {}
        for category, candidates in pools.items():
            scored = []
            for c in candidates:
                score_data = self._score_candidate(c)
                if score_data:
                    c.update(score_data)
                    scored.append(c)
            
            # Sort by Manageability Score (Descending)
            scored.sort(key=lambda x: x.get("manageability_score", -999), reverse=True)
            results[category] = scored[:3]  # Keep top 3

        # 6. Structured Reporting
        self._print_report(results)

    def _score_candidate(self, candidate: Dict) -> Dict:
        """Helper to call PreTradeEvaluator based on candidate type."""
        strikes = candidate["strikes"]
        premiums = candidate["premiums"]
        
        # Use MDA State-Space for fast ranking
        try:
            # evaluator expects [Put, Put, Call, Call] for 4-legs, or 2 for verticals.
            # StrategyPermutator already provides strikes/premiums in consistent orders.
            eval_res = self.evaluator.evaluate_mdp_state_space(
                S0=self.s0,
                strikes=strikes,
                premiums=premiums,
                time_steps=390,
                vol=self.atm_vol
            )
            return eval_res
        except Exception as e:
            # logger.debug(f"Evaluation failed for {candidate['name']}: {e}")
            return {}

    def _print_report(self, results: Dict[str, List[Dict]]):
        print("\n" + "="*80)
        print(f"{'SMART SPREER REPORT':^80}")
        print("="*80)
        
        for category, candidates in results.items():
            print(f"\n💎 {category.upper()}")
            print("-" * 40)
            if not candidates:
                print("   No valid candidates found.")
                continue
                
            for i, c in enumerate(candidates):
                print(f" [{i+1}] {c['name']}")
                print(f"     Strikes: {c['strikes']}")
                print(f"     EV: {c.get('expected_value', 'N/A'):<15} | M-Score: {c.get('manageability_score', 'N/A')}")
                print(f"     Premium: {c.get('net_premium', 'N/A')}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    screener = SmartSpreadScreener("SPY")
    screener.run()
