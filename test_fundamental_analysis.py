#!/usr/bin/env python3
"""
测试增强的基本面分析功能
"""

from app.agent.market_analyst import MarketAnalystAgent
from app.tool.fundamental_fetcher import FundamentalFetcher


def test_fundamental_fetcher():
    """测试基本面数据获取工具"""
    print("🔍 测试基本面数据获取工具...")

    fundamental_tool = FundamentalFetcher()

    # 测试获取少量股票的基本面数据
    symbols = ["AAPL", "MSFT"]

    result = fundamental_tool.execute(
        symbols=symbols,
        metrics=["valuation", "profitability", "liquidity", "leverage", "growth"],
    )

    if result.get("success"):
        print("✅ 基本面数据获取成功")
        data = result["data"]

        for symbol in symbols:
            symbol_data = data.get(symbol, {})
            print(f"\n📊 {symbol} 基本面分析:")

            # 公司信息
            company_info = symbol_data.get("raw_data", {}).get("company_info", {})
            print(f"  公司: {company_info.get('name', 'N/A')}")
            print(f"  行业: {company_info.get('industry', 'N/A')}")
            print(f"  市值: ${company_info.get('market_cap', 0)/1e9:.1f}B")

            # 估值指标
            valuation = symbol_data.get("raw_data", {}).get("valuation", {})
            print(f"  P/E比率: {valuation.get('trailing_pe', 'N/A')}")
            print(f"  P/B比率: {valuation.get('price_to_book', 'N/A')}")
            print(f"  P/S比率: {valuation.get('price_to_sales', 'N/A')}")

            # 盈利能力
            profitability = symbol_data.get("raw_data", {}).get("profitability", {})
            roe = profitability.get("return_on_equity")
            profit_margin = profitability.get("profit_margin")
            if roe:
                print(f"  ROE: {roe*100:.1f}%")
            if profit_margin:
                print(f"  净利润率: {profit_margin*100:.1f}%")

            # 财务健康分数
            financial_health = symbol_data.get("calculated_metrics", {}).get(
                "financial_health", {}
            )
            overall_score = financial_health.get("overall_score", 0)
            print(f"  财务健康分数: {overall_score:.0f}/100")

            # 投资建议
            analysis = symbol_data.get("analysis", {})
            recommendation = analysis.get("recommendation", "N/A")
            print(f"  投资建议: {recommendation}")

            # 数据质量
            data_quality = symbol_data.get("data_quality", {})
            quality_assessment = data_quality.get("assessment", "N/A")
            completeness = data_quality.get("completeness", 0)
            print(f"  数据质量: {quality_assessment} ({completeness:.0f}%完整)")

    else:
        print(f"❌ 基本面数据获取失败: {result.get('error')}")
        return False

    return True


def test_enhanced_market_analyst():
    """测试增强的市场分析Agent"""
    print("\n🤖 测试增强的MarketAnalyst Agent...")

    market_agent = MarketAnalystAgent()

    # 测试不同的投资策略
    strategies = ["growth", "value", "balanced"]

    for strategy in strategies:
        print(f"\n📈 测试 {strategy} 策略:")

        result = market_agent.step(
            {"universe": {"max_candidates": 3, "objective": strategy}}
        )

        if "analyst" in result:
            analyst_data = result["analyst"]

            print(f"✅ {strategy} 策略分析完成")

            # 显示选择的股票
            tickers = analyst_data.get("tickers", [])
            rationale = analyst_data.get("rationale", {})

            print(f"  选择股票: {[t['symbol'] for t in tickers]}")

            # 显示详细理由
            for ticker_info in tickers:
                symbol = ticker_info["symbol"]
                reason = rationale.get(symbol, "N/A")
                market_cap = ticker_info.get("market_cap", 0)
                industry = ticker_info.get("industry", "N/A")

                print(f"  {symbol}:")
                print(f"    行业: {industry}")
                if market_cap:
                    print(f"    市值: ${market_cap/1e9:.1f}B")
                print(f"    选择理由: {reason}")
        else:
            print(f"❌ {strategy} 策略分析失败")
            return False

    return True


def test_fundamental_scoring_logic():
    """测试基本面评分逻辑"""
    print("\n🎯 测试基本面评分逻辑...")

    market_agent = MarketAnalystAgent()

    # 获取基本面数据
    fundamental_tool = FundamentalFetcher()
    test_symbols = ["AAPL", "MSFT", "GOOGL", "JPM"]

    result = fundamental_tool.execute(
        symbols=test_symbols,
        metrics=["valuation", "profitability", "liquidity", "leverage", "growth"],
    )

    if result.get("success"):
        fundamental_data = result["data"]

        # 测试不同策略的评分
        strategies = ["growth", "value", "balanced"]

        for strategy in strategies:
            print(f"\n📊 {strategy} 策略评分:")

            scored_candidates = market_agent._score_candidates(
                fundamental_data, strategy
            )

            for symbol, score, data in scored_candidates:
                company_info = data.get("raw_data", {}).get("company_info", {})
                company_name = company_info.get("name", symbol)

                print(f"  {symbol} ({company_name}): {score:.1f}/100")

                # 显示关键指标
                raw_data = data.get("raw_data", {})
                valuation = raw_data.get("valuation", {})
                profitability = raw_data.get("profitability", {})

                pe_ratio = valuation.get("trailing_pe")
                roe = profitability.get("return_on_equity")

                indicators = []
                if pe_ratio:
                    indicators.append(f"P/E: {pe_ratio:.1f}")
                if roe:
                    indicators.append(f"ROE: {roe*100:.1f}%")

                if indicators:
                    print(f"    关键指标: {', '.join(indicators)}")

        return True
    else:
        print("❌ 无法获取基本面数据进行评分测试")
        return False


def test_strategy_differentiation():
    """测试不同策略的差异化效果"""
    print("\n🔄 测试策略差异化效果...")

    market_agent = MarketAnalystAgent()

    results = {}

    # 运行不同策略
    for strategy in ["growth", "value", "balanced"]:
        result = market_agent.step(
            {"universe": {"max_candidates": 4, "objective": strategy}}
        )

        if "analyst" in result:
            tickers = result["analyst"]["tickers"]
            symbols = {t["symbol"] for t in tickers}
            results[strategy] = symbols
        else:
            print(f"❌ {strategy} 策略执行失败")
            return False

    # 分析差异化程度
    all_symbols = set()
    for symbols in results.values():
        all_symbols.update(symbols)

    print("✅ 策略差异化分析:")
    print(f"  Growth 选择: {results['growth']}")
    print(f"  Value 选择: {results['value']}")
    print(f"  Balanced 选择: {results['balanced']}")

    # 计算重叠度
    growth_value_overlap = len(results["growth"] & results["value"])
    growth_balanced_overlap = len(results["growth"] & results["balanced"])
    value_balanced_overlap = len(results["value"] & results["balanced"])

    print(f"\n  策略重叠分析:")
    print(f"    Growth-Value 重叠: {growth_value_overlap} 只股票")
    print(f"    Growth-Balanced 重叠: {growth_balanced_overlap} 只股票")
    print(f"    Value-Balanced 重叠: {value_balanced_overlap} 只股票")

    total_unique = len(all_symbols)
    print(f"    总共涉及: {total_unique} 只不同股票")

    # 验证有足够的差异化
    if total_unique >= 6:  # 期望至少有6只不同的股票
        print("✅ 策略差异化程度良好")
        return True
    else:
        print("⚠️ 策略差异化程度较低")
        return False


def test_integration_with_pipeline():
    """测试与完整流水线的集成"""
    print("\n🔗 测试与完整流水线集成...")

    from app.agent.risk_manager import RiskManagerAgent
    from app.agent.sentiment_analyzer import SentimentAnalyzerAgent
    from app.agent.technical_trader import TechnicalTraderAgent

    # 1. 增强的MarketAnalyst
    market_agent = MarketAnalystAgent()
    analyst_result = market_agent.step(
        {"universe": {"max_candidates": 2, "objective": "growth"}}
    )

    print("✅ 增强市场分析完成")
    print(f"  选择股票: {[t['symbol'] for t in analyst_result['analyst']['tickers']]}")

    # 2. 增强的SentimentAnalyzer
    sentiment_agent = SentimentAnalyzerAgent()
    sentiment_result = sentiment_agent.step(analyst_result)

    print("✅ 增强情感分析完成")
    sentiment_scores = sentiment_result["sentiment"]["scores"]
    print(f"  情感分数: {sentiment_scores}")

    # 3. TechnicalTrader
    tech_agent = TechnicalTraderAgent()
    combined_input = {**analyst_result, **sentiment_result}
    tech_result = tech_agent.step(combined_input)

    print("✅ 技术分析完成")
    tech_signals = tech_result["technical"]["signals"]
    print(f"  技术信号: {tech_signals}")

    # 4. RiskManager
    risk_agent = RiskManagerAgent()
    full_input = {**combined_input, **tech_result}
    risk_result = risk_agent.step({**full_input, "portfolio": {"cash": 100000}})

    print("✅ 风险管理完成")
    orders = risk_result["risk"]["orders"]
    print(f"  生成订单: {len(orders)} 个")

    # 5. 综合决策分析
    print(f"\n📊 综合投资决策:")
    for symbol in sentiment_scores.keys():
        # 基本面信息
        analyst_rationale = analyst_result["analyst"]["rationale"].get(symbol, "")

        # 情感分析
        sentiment_score = sentiment_scores[symbol]

        # 技术分析
        tech_signal = tech_signals.get(symbol, {})
        tech_action = tech_signal.get("action", "N/A")
        tech_confidence = tech_signal.get("confidence", 0)

        print(f"  {symbol}:")
        print(f"    基本面: {analyst_rationale[:100]}...")
        print(f"    情感分数: {sentiment_score:.3f}")
        print(f"    技术信号: {tech_action} (置信度: {tech_confidence:.3f})")

        # 综合评分 (基本面40%, 情感20%, 技术40%)
        # 简化的综合评分算法
        fundamental_score = (
            0.7
            if "Strong" in analyst_rationale or "score: 8" in analyst_rationale
            else 0.5
        )
        sentiment_weight = sentiment_score
        tech_weight = {"long": 1, "flat": 0, "short": -1}.get(
            tech_action, 0
        ) * tech_confidence

        combined_score = (
            fundamental_score * 0.4 + sentiment_weight * 0.2 + tech_weight * 0.4
        )

        print(f"    综合评分: {combined_score:.3f}")

        if combined_score > 0.6:
            final_recommendation = "强烈推荐"
        elif combined_score > 0.3:
            final_recommendation = "推荐"
        elif combined_score > -0.3:
            final_recommendation = "持有"
        else:
            final_recommendation = "回避"

        print(f"    最终建议: {final_recommendation}")

    return True


if __name__ == "__main__":
    print("🚀 基本面分析增强功能测试")
    print("=" * 60)

    tests = [
        ("基本面数据获取工具", test_fundamental_fetcher),
        ("增强的市场分析Agent", test_enhanced_market_analyst),
        ("基本面评分逻辑", test_fundamental_scoring_logic),
        ("策略差异化效果", test_strategy_differentiation),
        ("完整流水线集成", test_integration_with_pipeline),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n📋 执行测试: {test_name}")
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {str(e)}")

    print("\n" + "=" * 60)
    print(f"🎯 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有基本面分析功能测试通过！")
        print("✅ 系统基本面分析能力已大幅增强！")
    else:
        print("⚠️ 部分测试未通过，需要进一步优化")

    print("\n📈 新增基本面分析能力:")
    print("  ✅ 全面的财务指标计算")
    print("  ✅ 多维度财务健康评分")
    print("  ✅ 策略导向的股票评分")
    print("  ✅ 投资建议和风险评估")
    print("  ✅ 详细的基本面选股理由")
    print("  ✅ 与技术分析和情感分析的完美集成")
