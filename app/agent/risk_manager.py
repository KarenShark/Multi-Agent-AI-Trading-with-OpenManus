from app.agent.base import BaseAgent
from app.schema_trading import OrderItem, PortfolioState, RiskOutput, TechnicalOutput


class RiskManagerAgent(BaseAgent):
    def __init__(self, name="risk_manager"):
        super().__init__(name=name, system_prompt="")

    def step(self, inputs):
        tech = TechnicalOutput(**inputs["technical"])
        port = PortfolioState(**inputs.get("portfolio", {"cash": 100000}))
        orders = []
        for sym, sig in tech.signals.items():
            if sig.action == "long":
                qty = int(port.cash / 100)
                orders.append(OrderItem(symbol=sym, side="buy", qty=qty))
        out = RiskOutput(orders=orders, risk_metrics={"exposure": len(orders)})
        return {"risk": out.model_dump()}
