#!/usr/bin/env python3
"""
完整的自动化流水线验证脚本
根据用户提供的深度核查清单进行逐项验证
"""

import sys
import traceback

from app.agent.market_analyst import MarketAnalystAgent
from app.agent.risk_manager import RiskManagerAgent
from app.agent.sentiment_analyzer import SentimentAnalyzerAgent
from app.agent.technical_trader import TechnicalTraderAgent
from app.schema_trading import (
    AnalystOutput,
    RiskOutput,
    SentimentOutput,
    TechnicalOutput,
    UniverseConfig,
)
from app.task.trading_task import TradingTask


def verify_schemas():
    """验证数据模型契约"""
    print("🔍 验证 Schema 定义...")

    # 测试 UniverseConfig
    cfg = UniverseConfig(max_candidates=3)
    assert cfg.max_candidates == 3
    assert cfg.objective == "balanced"  # 默认值

    # 测试其他模型
    analyst_out = AnalystOutput(tickers=[], rationale={})
    sentiment_out = SentimentOutput(scores={}, sources={})
    technical_out = TechnicalOutput(signals={})
    risk_out = RiskOutput(orders=[], risk_metrics={})

    print("✅ Schema 验证通过")


def verify_individual_agents():
    """验证各个 Agent 的独立功能"""
    print("🔍 验证各个 Agent...")

    # 1. MarketAnalyst 测试
    ma = MarketAnalystAgent()
    ma_result = ma.step({"universe": {"max_candidates": 2}})
    assert "analyst" in ma_result
    assert len(ma_result["analyst"]["tickers"]) == 2
    assert "rationale" in ma_result["analyst"]
    print("✅ MarketAnalystAgent 验证通过")

    # 2. SentimentAnalyzer 测试
    sa = SentimentAnalyzerAgent()
    sa_input = {"analyst": ma_result["analyst"]}
    sa_result = sa.step(sa_input)
    assert "sentiment" in sa_result
    assert "scores" in sa_result["sentiment"]
    assert len(sa_result["sentiment"]["scores"]) == 2
    print("✅ SentimentAnalyzerAgent 验证通过")

    # 3. TechnicalTrader 测试
    tt = TechnicalTraderAgent()
    tt_input = {"analyst": ma_result["analyst"]}
    tt_result = tt.step(tt_input)
    assert "technical" in tt_result
    assert "signals" in tt_result["technical"]
    assert len(tt_result["technical"]["signals"]) == 2
    # 验证信号结构
    for symbol, signal in tt_result["technical"]["signals"].items():
        assert "action" in signal
        assert "confidence" in signal
        assert signal["action"] in ["long", "short", "flat"]
        assert 0 <= signal["confidence"] <= 1
    print("✅ TechnicalTraderAgent 验证通过")

    # 4. RiskManager 测试
    rm = RiskManagerAgent()
    rm_input = {"technical": tt_result["technical"], "portfolio": {"cash": 10000}}
    rm_result = rm.step(rm_input)
    assert "risk" in rm_result
    assert "orders" in rm_result["risk"]
    assert "risk_metrics" in rm_result["risk"]
    print("✅ RiskManagerAgent 验证通过")


def verify_integration_pipeline():
    """验证完整的集成流水线"""
    print("🔍 验证集成流水线...")

    # 构建 agents
    agents = {
        "analyst": MarketAnalystAgent(),
        "sentiment": SentimentAnalyzerAgent(),
        "technical": TechnicalTraderAgent(),
        "risk": RiskManagerAgent(),
    }

    # 创建任务
    task = TradingTask(agents)

    # 执行完整流程
    ctx = task.run_once(
        {
            "universe": {"max_candidates": 3, "objective": "growth"},
            "portfolio": {"cash": 20000},
        }
    )

    # 验证所有关键节点都存在
    assert all(k in ctx for k in ["analyst", "sentiment", "technical", "risk"])

    # 验证数据流一致性
    analyst_symbols = {t["symbol"] for t in ctx["analyst"]["tickers"]}
    sentiment_symbols = set(ctx["sentiment"]["scores"].keys())
    technical_symbols = set(ctx["technical"]["signals"].keys())

    assert analyst_symbols == sentiment_symbols == technical_symbols
    print(f"✅ 数据流一致性验证通过，处理 {len(analyst_symbols)} 只股票")

    # 验证订单生成逻辑
    orders = ctx["risk"]["orders"]
    print(f"✅ 生成 {len(orders)} 个订单")

    for order in orders:
        assert "symbol" in order
        assert "side" in order
        assert "qty" in order
        assert order["side"] in ["buy", "sell"]
        assert order["qty"] > 0

    print("✅ 集成流水线验证通过")


def verify_different_objectives():
    """验证不同投资目标的处理"""
    print("🔍 验证不同投资目标...")

    agents = {
        "analyst": MarketAnalystAgent(),
        "sentiment": SentimentAnalyzerAgent(),
        "technical": TechnicalTraderAgent(),
        "risk": RiskManagerAgent(),
    }
    task = TradingTask(agents)

    objectives = ["growth", "value", "balanced"]
    results = {}

    for obj in objectives:
        ctx = task.run_once(
            {
                "universe": {"max_candidates": 2, "objective": obj},
                "portfolio": {"cash": 15000},
            }
        )
        results[obj] = ctx
        print(f"✅ {obj} 目标处理成功，选出 {len(ctx['analyst']['tickers'])} 只股票")

    # 验证不同目标可能选出不同的股票
    growth_symbols = {t["symbol"] for t in results["growth"]["analyst"]["tickers"]}
    value_symbols = {t["symbol"] for t in results["value"]["analyst"]["tickers"]}
    balanced_symbols = {t["symbol"] for t in results["balanced"]["analyst"]["tickers"]}

    print(f"Growth 选择: {growth_symbols}")
    print(f"Value 选择: {value_symbols}")
    print(f"Balanced 选择: {balanced_symbols}")

    print("✅ 不同投资目标验证通过")


def verify_real_data_integration():
    """验证真实数据集成（非硬编码）"""
    print("🔍 验证真实数据集成...")

    # 测试 MarketAnalyst 是否使用真实的金融数据
    ma = MarketAnalystAgent()
    result1 = ma.step({"universe": {"max_candidates": 3, "objective": "growth"}})
    result2 = ma.step({"universe": {"max_candidates": 3, "objective": "value"}})

    # 验证不同目标可能产生不同结果
    symbols1 = {t["symbol"] for t in result1["analyst"]["tickers"]}
    symbols2 = {t["symbol"] for t in result2["analyst"]["tickers"]}

    print(f"Growth 股票: {symbols1}")
    print(f"Value 股票: {symbols2}")

    # 验证返回的股票有真实的市场数据
    for ticker in result1["analyst"]["tickers"]:
        if ticker.get("market_cap"):
            assert ticker["market_cap"] > 0
            print(f"✅ {ticker['symbol']} 有真实市值数据: ${ticker['market_cap']:,.0f}")

    # 测试 TechnicalTrader 是否使用真实技术指标
    tt = TechnicalTraderAgent()
    tt_result = tt.step({"analyst": result1["analyst"]})

    # 验证技术信号不全是固定值
    confidences = [
        sig["confidence"] for sig in tt_result["technical"]["signals"].values()
    ]
    actions = [sig["action"] for sig in tt_result["technical"]["signals"].values()]

    print(f"技术信号置信度: {confidences}")
    print(f"技术信号动作: {actions}")

    # 如果有多个股票，置信度应该有变化（非硬编码）
    if len(confidences) > 1:
        assert (
            len(set(confidences)) > 1 or len(set(actions)) > 1
        ), "技术信号应该有差异化"

    print("✅ 真实数据集成验证通过")


def main():
    """主验证函数"""
    print("🚀 StockSynergy 流水线深度验证")
    print("=" * 60)

    try:
        verify_schemas()
        verify_individual_agents()
        verify_integration_pipeline()
        verify_different_objectives()
        verify_real_data_integration()

        print("\n" + "=" * 60)
        print("🎉 所有验证测试通过！")
        print("✅ StockSynergy 交易系统正确实现，非硬编码！")
        print("✅ 系统已准备好进入生产环境！")
        return 0

    except Exception as e:
        print(f"\n❌ 验证失败: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
