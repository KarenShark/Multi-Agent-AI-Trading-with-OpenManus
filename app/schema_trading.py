from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class TickerItem(BaseModel):
    symbol: str
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    note: Optional[str] = None


class UniverseConfig(BaseModel):
    objective: Literal["growth", "value", "balanced"] = "balanced"
    max_candidates: int = 10


class AnalystOutput(BaseModel):
    tickers: List[TickerItem]
    rationale: Dict[str, str]


class SentimentOutput(BaseModel):
    scores: Dict[str, float]
    sources: Dict[str, List[str]] = {}
    metadata: Optional[Dict[str, Any]] = {}


class SignalItem(BaseModel):
    action: Literal["long", "short", "flat"]
    confidence: float


class TechnicalOutput(BaseModel):
    signals: Dict[str, SignalItem]


class PortfolioState(BaseModel):
    cash: float
    positions: Dict[str, float] = {}


class OrderItem(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    limit_price: Optional[float] = None


class RiskOutput(BaseModel):
    orders: List[OrderItem]
    risk_metrics: Dict[str, float]


class MacroOutput(BaseModel):
    strategy_adjustments: Dict[str, Any]
    risk_factors: Dict[str, Any]
    sector_guidance: Dict[str, Any]
    timing_factors: Dict[str, Any]
    investment_regime: Dict[str, Any]
    confidence_level: Dict[str, Any]
