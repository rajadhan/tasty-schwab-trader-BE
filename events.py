from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class OptionLegEvent:
    symbol: str
    price: float
    volume: int
    strike: float
    type: str  # 'call' or 'put'
    expiry_years: float
    qty: float
    # Optional Greeks if available from provider
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    iv: Optional[float] = None

@dataclass
class BarEvent:
    ticker: str
    timestamp: str  # ISO8601
    underlying_price: float
    legs: List[OptionLegEvent] = field(default_factory=list)

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "underlying_price": self.underlying_price,
            "legs": [
                {
                    "symbol": l.symbol,
                    "price": l.price,
                    "volume": l.volume,
                    "strike": l.strike,
                    "type": l.type,
                    "expiry_years": l.expiry_years,
                    "qty": l.qty,
                    "delta": l.delta,
                    "gamma": l.gamma,
                    "theta": l.theta,
                    "iv": l.iv
                } for l in self.legs
            ]
        }
