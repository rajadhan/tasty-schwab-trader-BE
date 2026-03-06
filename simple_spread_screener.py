import pandas as pd
from typing import List, Dict, Optional
from public_chain_service import PublicChainService

class StrategyPermutator:
    """
    Takes a snapshot of an option chain and permutes valid Level-2 options strategies.
    Strategies include:
    - Bull Put Spread (Credit)
    - Bear Call Spread (Credit)
    - Bull Call Spread (Debit)
    - Bear Put Spread (Debit)
    - Iron Condor (Credit)
    - Long Call/Put Butterfly (Debit)
    - Iron Butterfly (Credit)
    - Broken Wing Butterfly (Credit)
    """

    def __init__(self, chain: List[Dict]):
        """
        :param chain: List of dictionaries representing the option chain.
                      Expected format for each dict:
                      {
                          "strike": float,
                          "call_bid": float,
                          "call_ask": float,
                          "put_bid": float,
                          "put_ask": float
                      }
                      Requires strikes to be sorted in ascending order.
        """
        # Ensure strikes are sorted
        self.chain = sorted(chain, key=lambda x: x['strike'])
        self.strikes = [leg['strike'] for leg in self.chain]
        self.chain_map = {leg['strike']: leg for leg in self.chain}

    def get_leg_price(self, strike: float, option_type: str, action: str) -> float:
        """
        Get the price for a specific leg using natural pricing to account for slippage.
        Selling (short) -> hits the Bid.
        Buying (long) -> hits the Ask.
        
        :param strike: The strike price.
        :param option_type: "call" or "put".
        :param action: "buy" or "sell".
        :return: Premium value (positive for sell/credit, negative for buy/debit).
        """
        leg = self.chain_map.get(strike)
        if not leg:
            return 0.0
            
        if option_type == "call":
            if action == "sell":
                return leg["call_bid"]
            elif action == "buy":
                return -leg["call_ask"]
        elif option_type == "put":
            if action == "sell":
                return leg["put_bid"]
            elif action == "buy":
                return -leg["put_ask"]
        return 0.0

    def find_vertical_spreads(self, 
                              option_type: str, 
                              spread_type: str, 
                              allowed_widths: List[float] = [5.0, 10.0]) -> List[Dict]:
        """
        Permute vertical spreads.
        
        :param option_type: "call" or "put".
        :param spread_type: "credit" or "debit".
        :param allowed_widths: List of allowed strike widths.
        :return: List of candidate spread dictionaries.
        """
        candidates = []
        for i, short_strike in enumerate(self.strikes):
            for width in allowed_widths:
                # Determine long strike based on option type and spread type
                if option_type == "call":
                    if spread_type == "credit": # Bear Call Spread
                        long_strike = short_strike + width
                    else: # Debit: Bull Call Spread
                        long_strike = short_strike - width
                else: # put
                    if spread_type == "credit": # Bull Put Spread
                        long_strike = short_strike - width
                    else: # Debit: Bear Put Spread
                        long_strike = short_strike + width
                        
                if long_strike in self.chain_map:
                    # Construct
                    short_premium = self.get_leg_price(short_strike, option_type, "sell")
                    long_premium = self.get_leg_price(long_strike, option_type, "buy")
                    
                    # Both strikes must yield valid non-zero prices theoretically (or actual)
                    if short_premium == 0 or long_premium == 0:
                        continue
                        
                    candidate = {
                        "name": f"{'Bear' if (option_type=='call' and spread_type=='credit') or (option_type=='put' and spread_type=='debit') else 'Bull'} {option_type.capitalize()} Spread",
                        "type": "vertical",
                        "option_type": option_type,
                        "spread_type": spread_type,
                        "strikes": [short_strike, long_strike],
                        "premiums": [short_premium, long_premium],
                        "width": width
                    }
                    candidates.append(candidate)
                    
        return candidates

    def find_iron_condors(self, allowed_widths: List[float] = [5.0, 10.0], min_wing_distance: float = 10.0) -> List[Dict]:
        """
        Permute Iron Condors (combining a Bull Put Spread and Bear Call Spread).
        
        :param allowed_widths: Allowed widths for the individual wings.
        :param min_wing_distance: Minimum distance between the short put and short call.
        :return: List of candidate Iron Condor dictionaries.
        """
        candidates = []
        
        put_spreads = self.find_vertical_spreads("put", "credit", allowed_widths)
        call_spreads = self.find_vertical_spreads("call", "credit", allowed_widths)
        
        for p_spread in put_spreads:
            for c_spread in call_spreads:
                short_put = p_spread["strikes"][0]
                short_call = c_spread["strikes"][0]
                
                # Ensure the calls are above the puts and separated by minimum distance
                if short_call - short_put >= min_wing_distance:
                    # Valid Iron Condor
                    # Pre-trade evaluator expects [Long Put, Short Put, Short Call, Long Call]
                    long_put = p_spread["strikes"][1]
                    long_call = c_spread["strikes"][1]
                    
                    p_premium_short = p_spread["premiums"][0]
                    p_premium_long = p_spread["premiums"][1]
                    c_premium_short = c_spread["premiums"][0]
                    c_premium_long = c_spread["premiums"][1]
                    
                    candidate = {
                        "name": "Iron Condor",
                        "type": "iron_condor",
                        "strikes": [long_put, short_put, short_call, long_call],
                        "premiums": [p_premium_long, p_premium_short, c_premium_short, c_premium_long],
                        "width_put": p_spread["width"],
                        "width_call": c_spread["width"]
                    }
                    candidates.append(candidate)
                    
        return candidates

    def find_butterflies(self, option_type: str, allowed_widths: List[float] = [5.0, 10.0]) -> List[Dict]:
        """
        Permute Long Butterflies (All Calls or All Puts).
        A Long Call Butterfly is: Long 1 lower strike, Short 2 middle strikes, Long 1 higher strike.
        The wings are equidistant from the body.
        
        :param option_type: "call" or "put".
        :param allowed_widths: Width between the body and each wing.
        :return: List of candidate Butterfly dictionaries.
        """
        candidates = []
        for strike in self.strikes:
            for width in allowed_widths:
                lower_strike = strike - width
                upper_strike = strike + width
                
                if lower_strike in self.chain_map and upper_strike in self.chain_map:
                    # Construct Long Butterfly (Debit)
                    long_lower_prem = self.get_leg_price(lower_strike, option_type, "buy")
                    short_body_prem = self.get_leg_price(strike, option_type, "sell") * 2
                    long_upper_prem = self.get_leg_price(upper_strike, option_type, "buy")
                    
                    if long_lower_prem == 0 or short_body_prem == 0 or long_upper_prem == 0:
                        continue
                        
                    candidate = {
                        "name": f"Long {option_type.capitalize()} Butterfly",
                        "type": "butterfly",
                        "option_type": option_type,
                        "strikes": [lower_strike, strike, strike, upper_strike],
                        "premiums": [long_lower_prem, short_body_prem / 2, short_body_prem / 2, long_upper_prem],
                        "width": width
                    }
                    candidates.append(candidate)
        return candidates

    def find_iron_butterflies(self, allowed_widths: List[float] = [5.0, 10.0]) -> List[Dict]:
        """
        Permute Iron Butterflies.
        An Iron Butterfly is an Iron Condor where the short call and short put share the exact same strike (ATM).
        
        :param allowed_widths: Width of the wings from the short straddle body.
        :return: List of candidate Iron Butterfly dictionaries.
        """
        candidates = []
        for body_strike in self.strikes:
            for width in allowed_widths:
                long_put_strike = body_strike - width
                long_call_strike = body_strike + width
                
                if long_put_strike in self.chain_map and long_call_strike in self.chain_map:
                    long_put_prem = self.get_leg_price(long_put_strike, "put", "buy")
                    short_put_prem = self.get_leg_price(body_strike, "put", "sell")
                    short_call_prem = self.get_leg_price(body_strike, "call", "sell")
                    long_call_prem = self.get_leg_price(long_call_strike, "call", "buy")
                    
                    if 0 in (long_put_prem, short_put_prem, short_call_prem, long_call_prem):
                        continue
                        
                    candidate = {
                        "name": "Iron Butterfly",
                        "type": "iron_butterfly",
                        "strikes": [long_put_strike, body_strike, body_strike, long_call_strike],
                        "premiums": [long_put_prem, short_put_prem, short_call_prem, long_call_prem],
                        "width": width
                    }
                    candidates.append(candidate)
        return candidates

    def find_broken_wing_butterflies(self, option_type: str, allowed_narrow_widths: List[float] = [5.0], allowed_wide_widths: List[float] = [10.0]) -> List[Dict]:
        """
        Permute Broken Wing Butterflies (BWB).
        Usually established for a net credit, where one wing is further away than the other.
        Example Call BWB: Long 1 ITM Call, Short 2 ATM Calls, Long 1 OTM Call (where OTM width > ITM width).
        
        :param option_type: "call" or "put".
        :return: List of candidate Broken Wing Butterfly dictionaries.
        """
        candidates = []
        for body_strike in self.strikes:
            for n_width in allowed_narrow_widths:
                for w_width in allowed_wide_widths:
                    if n_width >= w_width:
                        continue # BWB explicitly requires asymmetric wings
                        
                    # Standard BWB: The skipped strike is on the OTM side (directional risk)
                    if option_type == "call":
                        lower_strike = body_strike - n_width
                        upper_strike = body_strike + w_width
                    else: # put
                        upper_strike = body_strike + n_width
                        lower_strike = body_strike - w_width
                        
                    if lower_strike in self.chain_map and upper_strike in self.chain_map:
                        long_lower_prem = self.get_leg_price(lower_strike, option_type, "buy")
                        short_body_prem = self.get_leg_price(body_strike, option_type, "sell") * 2
                        long_upper_prem = self.get_leg_price(upper_strike, option_type, "buy")
                        
                        if long_lower_prem == 0 or short_body_prem == 0 or long_upper_prem == 0:
                            continue
                            
                        candidate = {
                            "name": f"Broken Wing {option_type.capitalize()} Butterfly",
                            "type": "broken_wing_butterfly",
                            "option_type": option_type,
                            "strikes": [lower_strike, body_strike, body_strike, upper_strike],
                            "premiums": [long_lower_prem, short_body_prem / 2, short_body_prem / 2, long_upper_prem],
                            "narrow_width": n_width,
                            "wide_width": w_width
                        }
                        candidates.append(candidate)
        return candidates

def run_screener_on_public(symbol="SPY"):
    """Orchestrates the screener using Public.com live data."""
    print(f"--- Starting Public.com Screener for {symbol} ---")
    
    # 1. Initialize Service
    svc = PublicChainService()
    if not svc.authenticate():
        print("Authentication failed.")
        return

    # 2. Get Expirations
    print(f"Fetching expirations for {symbol}...")
    exps = svc.get_expirations(symbol)
    if not exps:
        print("No expirations found.")
        return
    
    # Target the nearest monthly (usually 3rd Friday, but we'll take the first available for now)
    target_exp = exps[0]
    print(f"Targeting expiration: {target_exp}")

    # 3. Fetch and Format Chain
    formatted_chain = svc.get_formatted_chain(symbol, target_exp)
    if not formatted_chain:
        print("Failed to retrieve or format the option chain.")
        return

    print(f"Retrieved chain for {len(formatted_chain)} strikes.")

    # 4. Permute Strategies
    permutator = StrategyPermutator(formatted_chain)
    
    print("\nPermuting Strategies...")
    verticals = permutator.find_vertical_spreads("put", "credit", [5.0, 10.0])
    condors = permutator.find_iron_condors([5.0], 10.0)
    iron_butterflies = permutator.find_iron_butterflies([5.0])
    
    all_candidates = verticals + condors + iron_butterflies
    print(f"Found {len(all_candidates)} candidate strategies.")

    # 5. Simple Display (Top 10 Interesting)
    # Filter for non-zero net premium and sort by credit (if credit spread)
    interesting = [c for c in all_candidates if abs(sum(c['premiums'])) > 0.01]
    interesting.sort(key=lambda x: sum(x['premiums']), reverse=True)

    print(f"\nTop {min(10, len(interesting))} High-Credit Strategies:")
    for i, candidate in enumerate(interesting[:10]):
        net_prem = sum(candidate['premiums'])
        print(f"[{i+1}] {candidate['name']:<25} | Strikes: {str(candidate['strikes']):<25} | Net Credit: ${net_prem:.2f}")

if __name__ == "__main__":
    # You can change the symbol here
    run_screener_on_public("SPY")
