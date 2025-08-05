"""
单元测试：StockSynergy Trading Pipeline
"""

import pytest

from app.agent.market_analyst import MarketAnalystAgent
from app.agent.risk_manager import RiskManagerAgent
from app.agent.sentiment_analyzer import SentimentAnalyzerAgent
from app.agent.technical_trader import TechnicalTraderAgent
from app.schema_trading import AnalystOutput, OrderItem, UniverseConfig
from app.task.trading_task import TradingTask


@pytest.fixture
def agents():
    """创建测试用的 agents"""
    return {
        "analyst": MarketAnalystAgent(),
        "sentiment": SentimentAnalyzerAgent(),
        "technical": TechnicalTraderAgent(),
        "risk": RiskManagerAgent(),
    }


def test_schema_models():
    """测试数据模型"""
    # 测试 UniverseConfig
    config = UniverseConfig(objective="growth", max_candidates=5)
    assert config.objective == "growth"
    assert config.max_candidates == 5

    # 测试默认值
    default_config = UniverseConfig()
    assert default_config.objective == "balanced"
    assert default_config.max_candidates == 10


def test_market_analyst_agent():
    """测试市场分析 Agent"""
    agent = MarketAnalystAgent()
    result = agent.step({"universe": {"objective": "growth", "max_candidates": 3}})

    assert "analyst" in result
    analyst_data = result["analyst"]
    assert "tickers" in analyst_data
    assert "rationale" in analyst_data
    assert len(analyst_data["tickers"]) <= 3


def test_sentiment_analyzer_agent():
    """测试情感分析 Agent"""
    # 先获取分析结果
    analyst = MarketAnalystAgent()
    analyst_result = analyst.step(
        {"universe": {"objective": "balanced", "max_candidates": 2}}
    )

    # 测试情感分析
    sentiment_agent = SentimentAnalyzerAgent()
    result = sentiment_agent.step(analyst_result)

    assert "sentiment" in result
    sentiment_data = result["sentiment"]
    assert "scores" in sentiment_data
    assert "sources" in sentiment_data


def test_technical_trader_agent():
    """测试技术分析 Agent"""
    # 准备输入数据
    analyst = MarketAnalystAgent()
    analyst_result = analyst.step(
        {"universe": {"objective": "balanced", "max_candidates": 2}}
    )

    sentiment = SentimentAnalyzerAgent()
    sentiment_result = sentiment.step(analyst_result)

    # 测试技术分析
    technical_agent = TechnicalTraderAgent()
    combined_input = {**analyst_result, **sentiment_result}
    result = technical_agent.step(combined_input)

    assert "technical" in result
    technical_data = result["technical"]
    assert "signals" in technical_data


def test_risk_manager_agent():
    """测试风险管理 Agent"""
    # 准备完整流程的输入
    analyst = MarketAnalystAgent()
    analyst_result = analyst.step(
        {"universe": {"objective": "balanced", "max_candidates": 2}}
    )

    sentiment = SentimentAnalyzerAgent()
    sentiment_result = sentiment.step(analyst_result)

    technical = TechnicalTraderAgent()
    technical_result = technical.step({**analyst_result, **sentiment_result})

    # 测试风险管理
    risk_agent = RiskManagerAgent()
    full_input = {
        **analyst_result,
        **sentiment_result,
        **technical_result,
        "portfolio": {"cash": 100000},
    }
    result = risk_agent.step(full_input)

    assert "risk" in result
    risk_data = result["risk"]
    assert "orders" in risk_data
    assert "risk_metrics" in risk_data
    assert isinstance(risk_data["orders"], list)


def test_trading_flow(agents):
    """测试完整交易流程"""
    task = TradingTask(agents)
    ctx = task.run_once(
        {
            "universe": {"objective": "balanced", "max_candidates": 3},
            "portfolio": {"cash": 50000, "positions": {}},
        }
    )

    # 检查关键节点输出
    assert "analyst" in ctx
    assert "sentiment" in ctx
    assert "technical" in ctx
    assert "risk" in ctx

    # 检查 orders 结构
    orders = ctx["risk"]["orders"]
    assert isinstance(orders, list)

    # 如果有订单，检查订单格式
    for order in orders:
        assert "symbol" in order
        assert "side" in order
        assert "qty" in order
        assert order["side"] in ["buy", "sell"]
        assert isinstance(order["qty"], (int, float))


def test_different_objectives(agents):
    """测试不同投资目标"""
    task = TradingTask(agents)
    objectives = ["growth", "value", "balanced"]

    for objective in objectives:
        result = task.run_once(
            {
                "universe": {"objective": objective, "max_candidates": 2},
                "portfolio": {"cash": 100000},
            }
        )

        # 验证基本结构
        assert "analyst" in result
        assert "sentiment" in result
        assert "technical" in result
        assert "risk" in result

        # 验证股票池符合预期
        tickers = [t["symbol"] for t in result["analyst"]["tickers"]]
        assert len(tickers) <= 2


def test_data_consistency(agents):
    """测试数据一致性"""
    task = TradingTask(agents)
    result = task.run_once(
        {
            "universe": {"objective": "balanced", "max_candidates": 3},
            "portfolio": {"cash": 100000},
        }
    )

    # 获取各阶段的股票符号
    analyst_symbols = {t["symbol"] for t in result["analyst"]["tickers"]}
    sentiment_symbols = set(result["sentiment"]["scores"].keys())
    technical_symbols = set(result["technical"]["signals"].keys())

    # 检查符号一致性
    assert analyst_symbols == sentiment_symbols
    assert analyst_symbols == technical_symbols

    # 检查订单符号
    order_symbols = {order["symbol"] for order in result["risk"]["orders"]}
    assert order_symbols.issubset(analyst_symbols)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
