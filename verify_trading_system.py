#!/usr/bin/env python3
"""
验证 StockSynergy Trading System 的完整性和正确性
"""

from app.agent.market_analyst import MarketAnalystAgent
from app.agent.risk_manager import RiskManagerAgent
from app.agent.sentiment_analyzer import SentimentAnalyzerAgent
from app.agent.technical_trader import TechnicalTraderAgent
from app.schema_trading import *
from app.task.trading_task import TradingTask


def test_schema_validation():
    """测试数据模型验证"""
    print("🔍 Testing Schema Validation...")

    # 测试 UniverseConfig
    cfg = UniverseConfig(objective="growth", max_candidates=3)
    assert cfg.objective == "growth"
    assert cfg.max_candidates == 3

    # 测试 TickerItem
    ticker = TickerItem(symbol="AAPL", industry="Technology", market_cap=3000000000000)
    assert ticker.symbol == "AAPL"

    # 测试 OrderItem
    order = OrderItem(symbol="AAPL", side="buy", qty=100)
    assert order.symbol == "AAPL"
    assert order.side == "buy"
    assert order.qty == 100

    print("✅ Schema validation passed!")
    return True


def test_individual_agents():
    """测试单个 Agent 功能"""
    print("\n🤖 Testing Individual Agents...")

    # 测试 MarketAnalystAgent
    analyst = MarketAnalystAgent()
    analyst_result = analyst.step(
        {"universe": {"objective": "balanced", "max_candidates": 2}}
    )
    assert "analyst" in analyst_result
    assert "tickers" in analyst_result["analyst"]
    assert "rationale" in analyst_result["analyst"]
    print("✅ MarketAnalystAgent working")

    # 测试 SentimentAnalyzerAgent
    sentiment = SentimentAnalyzerAgent()
    sentiment_result = sentiment.step(analyst_result)
    assert "sentiment" in sentiment_result
    assert "scores" in sentiment_result["sentiment"]
    print("✅ SentimentAnalyzerAgent working")

    # 测试 TechnicalTraderAgent
    technical = TechnicalTraderAgent()
    combined_input = {**analyst_result, **sentiment_result}
    technical_result = technical.step(combined_input)
    assert "technical" in technical_result
    assert "signals" in technical_result["technical"]
    print("✅ TechnicalTraderAgent working")

    # 测试 RiskManagerAgent
    risk = RiskManagerAgent()
    full_input = {**combined_input, **technical_result, "portfolio": {"cash": 100000}}
    risk_result = risk.step(full_input)
    assert "risk" in risk_result
    assert "orders" in risk_result["risk"]
    assert "risk_metrics" in risk_result["risk"]
    print("✅ RiskManagerAgent working")

    return True


def test_complete_pipeline():
    """测试完整流水线"""
    print("\n🔄 Testing Complete Pipeline...")

    # 初始化 agents
    agents = {
        "analyst": MarketAnalystAgent(),
        "sentiment": SentimentAnalyzerAgent(),
        "technical": TechnicalTraderAgent(),
        "risk": RiskManagerAgent(),
    }

    # 创建 task
    task = TradingTask(agents)

    # 测试不同策略
    objectives = ["growth", "value", "balanced"]

    for objective in objectives:
        print(f"  Testing {objective} strategy...")
        result = task.run_once(
            {
                "universe": {"objective": objective, "max_candidates": 3},
                "portfolio": {"cash": 100000},
            }
        )

        # 验证输出结构
        assert "analyst" in result
        assert "sentiment" in result
        assert "technical" in result
        assert "risk" in result

        # 验证 analyst 输出
        analyst_data = result["analyst"]
        assert "tickers" in analyst_data
        assert "rationale" in analyst_data
        assert len(analyst_data["tickers"]) <= 3  # max_candidates

        # 验证 sentiment 输出
        sentiment_data = result["sentiment"]
        assert "scores" in sentiment_data
        assert "sources" in sentiment_data

        # 验证 technical 输出
        technical_data = result["technical"]
        assert "signals" in technical_data

        # 验证 risk 输出
        risk_data = result["risk"]
        assert "orders" in risk_data
        assert "risk_metrics" in risk_data
        assert isinstance(risk_data["orders"], list)

        # 验证订单格式（如果有订单的话）
        for order in risk_data["orders"]:
            assert "symbol" in order
            assert "side" in order
            assert "qty" in order
            assert order["side"] in ["buy", "sell"]
            assert isinstance(order["qty"], (int, float))
            assert order["qty"] > 0

        print(f"    ✅ {objective} strategy validated")

    print("✅ Complete pipeline test passed!")
    return True


def test_data_flow_consistency():
    """测试数据流一致性"""
    print("\n🔗 Testing Data Flow Consistency...")

    agents = {
        "analyst": MarketAnalystAgent(),
        "sentiment": SentimentAnalyzerAgent(),
        "technical": TechnicalTraderAgent(),
        "risk": RiskManagerAgent(),
    }

    task = TradingTask(agents)
    result = task.run_once(
        {
            "universe": {"objective": "balanced", "max_candidates": 3},
            "portfolio": {"cash": 100000},
        }
    )

    # 检查符号一致性：每个阶段都应该处理相同的股票符号
    analyst_symbols = {t["symbol"] for t in result["analyst"]["tickers"]}
    sentiment_symbols = set(result["sentiment"]["scores"].keys())
    technical_symbols = set(result["technical"]["signals"].keys())

    assert (
        analyst_symbols == sentiment_symbols
    ), f"Symbol mismatch: analyst={analyst_symbols}, sentiment={sentiment_symbols}"
    assert (
        analyst_symbols == technical_symbols
    ), f"Symbol mismatch: analyst={analyst_symbols}, technical={technical_symbols}"

    # 检查订单中的符号是否来自分析的股票
    order_symbols = {order["symbol"] for order in result["risk"]["orders"]}
    assert order_symbols.issubset(
        analyst_symbols
    ), f"Order symbols {order_symbols} not in analyzed symbols {analyst_symbols}"

    print("✅ Data flow consistency verified!")
    return True


def main():
    """主验证函数"""
    print("🚀 StockSynergy Trading System Verification")
    print("=" * 60)

    try:
        # 运行所有验证
        test_schema_validation()
        test_individual_agents()
        test_complete_pipeline()
        test_data_flow_consistency()

        print("\n" + "=" * 60)
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("✅ StockSynergy Trading System is working correctly!")
        print("✅ Ready for production deployment or further enhancement!")

        return True

    except Exception as e:
        print(f"\n❌ Verification failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
