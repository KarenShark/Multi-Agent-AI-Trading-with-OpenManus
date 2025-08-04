from app.agent.market_analyst import MarketAnalystAgent
from app.agent.risk_manager import RiskManagerAgent
from app.agent.sentiment_analyzer import SentimentAnalyzerAgent
from app.agent.technical_trader import TechnicalTraderAgent
from app.task.trading_task import TradingTask

if __name__ == "__main__":
    agents = {
        "analyst": MarketAnalystAgent(),
        "sentiment": SentimentAnalyzerAgent(),
        "technical": TechnicalTraderAgent(),
        "risk": RiskManagerAgent(),
    }
    task = TradingTask(agents)
    result = task.run_once(
        {
            "universe": {"objective": "balanced", "max_candidates": 5},
            "portfolio": {"cash": 100000},
        }
    )
    print("Final orders:", result["risk"]["orders"])
