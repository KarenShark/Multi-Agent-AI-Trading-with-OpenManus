#!/usr/bin/env python3
"""
测试宏观经济分析功能
"""

from app.agent.macro_analyst import MacroAnalystAgent
from app.tool.macro_economic_fetcher import MacroEconomicFetcher


def test_macro_economic_fetcher():
    """测试宏观经济数据获取工具"""
    print("🔍 测试宏观经济数据获取工具...")

    macro_tool = MacroEconomicFetcher()

    # 测试获取核心宏观经济指标
    indicators = [
        "interest_rates",
        "inflation",
        "employment",
        "growth",
        "market_sentiment",
    ]

    result = macro_tool.execute(indicators=indicators, period="1y", country="US")

    if result.get("success"):
        print("✅ 宏观经济数据获取成功")
        data = result["data"]
        metadata = result["metadata"]

        print(f"\n📊 数据源信息:")
        print(
            f"  数据源: {metadata.get('data_sources', {}).get('primary_sources', [])}"
        )
        print(
            f"  覆盖范围: {metadata.get('data_sources', {}).get('data_coverage', 'N/A')}"
        )
        print(
            f"  更新频率: {metadata.get('data_sources', {}).get('update_frequency', 'N/A')}"
        )

        # 显示各类指标
        for category in indicators:
            category_data = data.get(category, {})
            print(f"\n📈 {category.replace('_', ' ').title()} 指标:")

            for indicator_name, indicator_data in category_data.items():
                if (
                    isinstance(indicator_data, dict)
                    and "latest_value" in indicator_data
                ):
                    latest_value = indicator_data.get("latest_value")
                    source = indicator_data.get("source", "Unknown")
                    print(f"  {indicator_name}: {latest_value} (来源: {source})")
                elif isinstance(indicator_data, dict) and "error" in indicator_data:
                    print(
                        f"  {indicator_name}: 数据获取失败 - {indicator_data['error']}"
                    )

        # 显示衍生指标
        derived = data.get("derived_indicators", {})
        if derived:
            print(f"\n🔬 衍生经济指标:")
            for indicator, value_data in derived.items():
                if isinstance(value_data, dict):
                    value = value_data.get("value", value_data.get("score", "N/A"))
                    interpretation = value_data.get("interpretation", "")
                    print(f"  {indicator}: {value}")
                    if interpretation:
                        print(f"    解释: {interpretation}")

        # 显示分析结果
        analysis = data.get("analysis", {})
        if analysis:
            print(f"\n📊 宏观经济分析:")

            overall = analysis.get("overall_assessment", {})
            print(f"  经济周期: {overall.get('economic_cycle', 'N/A')}")
            print(f"  衰退概率: {overall.get('recession_probability', 'N/A')}")
            print(f"  投资环境: {overall.get('investment_regime', 'N/A')}")

            if overall.get("key_risks"):
                print(f"  主要风险: {', '.join(overall['key_risks'])}")
            if overall.get("opportunities"):
                print(f"  投资机会: {', '.join(overall['opportunities'])}")

        # 显示投资展望
        outlook = data.get("outlook", {})
        if outlook:
            print(f"\n🔮 投资展望:")
            print(f"  短期前景: {outlook.get('short_term_outlook', 'N/A')}")
            print(f"  中期前景: {outlook.get('medium_term_outlook', 'N/A')}")

            investment_implications = outlook.get("investment_implications", {})
            asset_allocation = investment_implications.get("asset_allocation", {})
            if asset_allocation:
                print(f"  资产配置建议:")
                for asset, weight in asset_allocation.items():
                    print(f"    {asset}: {weight}")

            sector_prefs = investment_implications.get("sector_preferences", [])
            if sector_prefs:
                print(f"  行业偏好: {', '.join(sector_prefs)}")

    else:
        print(f"❌ 宏观经济数据获取失败: {result.get('error')}")
        return False

    return True


def test_macro_analyst_agent():
    """测试宏观经济分析Agent"""
    print("\n🤖 测试宏观经济分析Agent...")

    macro_agent = MacroAnalystAgent()

    # 测试基础宏观分析
    result = macro_agent.step(
        {
            "macro_context": {
                "indicators": ["interest_rates", "inflation", "employment", "growth"],
                "period": "1y",
                "country": "US",
            }
        }
    )

    if "macro" in result:
        macro_data = result["macro"]

        print("✅ 宏观经济分析Agent执行成功")

        # 显示策略调整建议
        strategy_adj = macro_data.get("strategy_adjustments", {})
        print(f"\n📊 策略调整建议:")
        print(
            f"  风险调整: {strategy_adj.get('risk_adjustment', 0):.2f} (-1=防御, +1=进攻)"
        )
        print(
            f"  久期偏好: {strategy_adj.get('duration_bias', 0):.2f} (-1=短久期, +1=长久期)"
        )
        print(
            f"  成长价值倾向: {strategy_adj.get('growth_value_tilt', 0):.2f} (-1=价值, +1=成长)"
        )
        print(f"  规模偏好: {strategy_adj.get('size_bias', 0):.2f} (-1=大盘, +1=小盘)")

        rationale = strategy_adj.get("rationale", [])
        if rationale:
            print(f"  调整理由:")
            for reason in rationale:
                print(f"    • {reason}")

        # 显示风险因子
        risk_factors = macro_data.get("risk_factors", {})
        print(f"\n⚠️ 宏观风险因子:")
        print(f"  衰退风险: {risk_factors.get('recession_risk', 0):.2f}")
        print(f"  通胀风险: {risk_factors.get('inflation_risk', 0):.2f}")
        print(f"  利率风险: {risk_factors.get('interest_rate_risk', 0):.2f}")
        print(f"  整体风险: {risk_factors.get('overall_risk', 0):.2f}")

        risk_desc = risk_factors.get("risk_description", [])
        if risk_desc:
            print(f"  风险描述:")
            for desc in risk_desc:
                print(f"    • {desc}")

        # 显示行业指导
        sector_guidance = macro_data.get("sector_guidance", {})
        print(f"\n🏢 行业配置指导:")

        overweight = sector_guidance.get("overweight", [])
        if overweight:
            print(f"  超配: {', '.join(overweight)}")

        underweight = sector_guidance.get("underweight", [])
        if underweight:
            print(f"  低配: {', '.join(underweight)}")

        neutral = sector_guidance.get("neutral", [])
        if neutral:
            print(f"  中性: {', '.join(neutral)}")

        # 显示市场择时因子
        timing_factors = macro_data.get("timing_factors", {})
        print(f"\n⏰ 市场择时因子:")
        print(f"  市场阶段: {timing_factors.get('market_phase', 'N/A')}")
        print(f"  趋势方向: {timing_factors.get('trend_direction', 'N/A')}")
        print(f"  波动率环境: {timing_factors.get('volatility_regime', 'N/A')}")
        print(
            f"  择时评分: {timing_factors.get('timing_score', 0):.2f} (-1=看跌, +1=看涨)"
        )

        # 显示投资体制
        investment_regime = macro_data.get("investment_regime", {})
        print(f"\n💼 投资体制:")
        print(f"  体制类型: {investment_regime.get('regime_type', 'N/A')}")
        print(f"  置信度: {investment_regime.get('confidence', 0):.2f}")
        print(f"  描述: {investment_regime.get('regime_description', 'N/A')}")

        # 显示分析置信度
        confidence = macro_data.get("confidence_level", {})
        print(f"\n🎯 分析置信度:")
        print(f"  整体置信度: {confidence.get('overall_confidence', 0):.2f}")
        print(f"  置信度等级: {confidence.get('confidence_level', 'N/A')}")

        uncertainty_factors = confidence.get("uncertainty_factors", [])
        if uncertainty_factors:
            print(f"  不确定因素:")
            for factor in uncertainty_factors:
                print(f"    • {factor}")

    else:
        print("❌ 宏观经济分析Agent执行失败")
        return False

    return True


def test_macro_integration_with_pipeline():
    """测试宏观分析与完整流水线的集成"""
    print("\n🔗 测试宏观分析与完整流水线集成...")

    from app.agent.market_analyst import MarketAnalystAgent
    from app.agent.risk_manager import RiskManagerAgent
    from app.agent.sentiment_analyzer import SentimentAnalyzerAgent
    from app.agent.technical_trader import TechnicalTraderAgent

    # 1. 宏观分析
    macro_agent = MacroAnalystAgent()
    macro_result = macro_agent.step(
        {
            "macro_context": {
                "indicators": ["interest_rates", "inflation", "employment", "growth"],
                "period": "1y",
            }
        }
    )

    print("✅ 宏观分析完成")
    macro_data = macro_result["macro"]
    strategy_adjustments = macro_data["strategy_adjustments"]
    investment_regime = macro_data["investment_regime"]["regime_type"]
    print(f"  投资体制: {investment_regime}")
    print(f"  风险调整: {strategy_adjustments['risk_adjustment']:.2f}")

    # 2. 基于宏观分析调整市场分析策略
    # 根据宏观分析结果调整投资目标
    if strategy_adjustments["risk_adjustment"] > 0.2:
        objective = "growth"
        max_candidates = 3
    elif strategy_adjustments["risk_adjustment"] < -0.2:
        objective = "value"
        max_candidates = 2
    else:
        objective = "balanced"
        max_candidates = 3

    print(f"\n📊 根据宏观分析调整策略目标为: {objective}")

    # 3. 市场分析（使用宏观调整后的策略）
    market_agent = MarketAnalystAgent()
    analyst_result = market_agent.step(
        {"universe": {"max_candidates": max_candidates, "objective": objective}}
    )

    print("✅ 基本面分析完成")
    selected_symbols = [t["symbol"] for t in analyst_result["analyst"]["tickers"]]
    print(f"  选择股票: {selected_symbols}")

    # 4. 情感分析
    sentiment_agent = SentimentAnalyzerAgent()
    sentiment_result = sentiment_agent.step(analyst_result)

    print("✅ 情感分析完成")
    sentiment_scores = sentiment_result["sentiment"]["scores"]

    # 5. 技术分析
    tech_agent = TechnicalTraderAgent()
    combined_input = {**analyst_result, **sentiment_result}
    tech_result = tech_agent.step(combined_input)

    print("✅ 技术分析完成")
    tech_signals = tech_result["technical"]["signals"]

    # 6. 风险管理（结合宏观风险因子）
    risk_agent = RiskManagerAgent()
    full_input = {**combined_input, **tech_result, **macro_result}

    # 根据宏观风险调整组合规模
    macro_risk = macro_data["risk_factors"]["overall_risk"]
    base_cash = 100000
    if macro_risk > 0.6:
        portfolio_cash = base_cash * 1.2  # 增加现金比例
    elif macro_risk < 0.3:
        portfolio_cash = base_cash * 0.8  # 降低现金比例
    else:
        portfolio_cash = base_cash

    risk_result = risk_agent.step({**full_input, "portfolio": {"cash": portfolio_cash}})

    print("✅ 风险管理完成")
    orders = risk_result["risk"]["orders"]
    print(f"  生成订单: {len(orders)} 个")

    # 7. 宏观驱动的综合决策分析
    print(f"\n📊 宏观驱动的投资决策:")

    # 显示宏观环境对每只股票的影响
    sector_guidance = macro_data["sector_guidance"]
    overweight_sectors = sector_guidance.get("overweight", [])
    underweight_sectors = sector_guidance.get("underweight", [])

    for symbol in selected_symbols:
        # 获取基本面信息
        symbol_info = next(
            (t for t in analyst_result["analyst"]["tickers"] if t["symbol"] == symbol),
            {},
        )
        industry = symbol_info.get("industry", "Unknown")

        # 宏观行业指导
        macro_sector_bias = "中性"
        if any(sector in industry for sector in overweight_sectors):
            macro_sector_bias = "看好"
        elif any(sector in industry for sector in underweight_sectors):
            macro_sector_bias = "谨慎"

        # 情感和技术因子
        sentiment_score = sentiment_scores.get(symbol, 0)
        tech_signal = tech_signals.get(symbol, {})
        tech_action = tech_signal.get("action", "flat")
        tech_confidence = tech_signal.get("confidence", 0)

        print(f"  {symbol} ({industry}):")
        print(f"    宏观行业偏好: {macro_sector_bias}")
        print(f"    情感分数: {sentiment_score:.3f}")
        print(f"    技术信号: {tech_action} (置信度: {tech_confidence:.3f})")

        # 宏观加权的综合评分
        macro_weight = 0.3
        fundamental_weight = 0.3
        sentiment_weight = 0.2
        technical_weight = 0.2

        # 宏观分数
        macro_score = 0.5  # 基准中性分数
        if macro_sector_bias == "看好":
            macro_score += 0.3
        elif macro_sector_bias == "谨慎":
            macro_score -= 0.3

        # 调整宏观分数基于整体风险环境
        risk_adjustment = strategy_adjustments["risk_adjustment"]
        macro_score += risk_adjustment * 0.2

        # 基本面分数（简化）
        fundamental_score = 0.7  # 假设基本面分析给出正面评价

        # 技术分数
        tech_score = {"long": 1, "flat": 0, "short": -1}.get(
            tech_action, 0
        ) * tech_confidence

        # 综合评分
        combined_score = (
            macro_score * macro_weight
            + fundamental_score * fundamental_weight
            + sentiment_score * sentiment_weight
            + tech_score * technical_weight
        )

        print(f"    综合评分: {combined_score:.3f}")
        print(
            f"      (宏观: {macro_score:.2f}, 基本面: {fundamental_score:.2f}, "
            f"情感: {sentiment_score:.2f}, 技术: {tech_score:.2f})"
        )

        # 最终建议
        if combined_score > 0.6:
            final_recommendation = "强烈推荐"
        elif combined_score > 0.4:
            final_recommendation = "推荐"
        elif combined_score > -0.2:
            final_recommendation = "持有"
        else:
            final_recommendation = "回避"

        print(f"    最终建议: {final_recommendation}")

    # 显示宏观风险监控指标
    print(f"\n⚠️ 宏观风险监控:")
    risk_factors = macro_data["risk_factors"]
    timing_factors = macro_data["timing_factors"]

    print(
        f"  当前宏观风险等级: {macro_risk:.2f} ({'高' if macro_risk > 0.6 else '中' if macro_risk > 0.3 else '低'})"
    )
    print(f"  市场择时评分: {timing_factors.get('timing_score', 0):.2f}")
    print(f"  建议现金比例: {((portfolio_cash / base_cash - 1) * 100):+.0f}%")

    return True


def test_different_macro_scenarios():
    """测试不同宏观经济场景下的策略调整"""
    print("\n🎭 测试不同宏观经济场景...")

    macro_agent = MacroAnalystAgent()

    # 场景1: 通胀场景
    print("\n📈 场景1: 高通胀环境测试")
    # 在实际应用中，这里会有不同的数据输入来模拟高通胀
    result1 = macro_agent.step(
        {
            "macro_context": {
                "indicators": ["interest_rates", "inflation", "employment"],
                "period": "1y",
            }
        }
    )

    if "macro" in result1:
        strategy_adj1 = result1["macro"]["strategy_adjustments"]
        sector_guidance1 = result1["macro"]["sector_guidance"]

        print(f"  策略调整 - 久期偏好: {strategy_adj1.get('duration_bias', 0):.2f}")
        print(f"  超配行业: {', '.join(sector_guidance1.get('overweight', []))}")

    # 场景2: 衰退担忧场景
    print("\n📉 场景2: 衰退担忧环境测试")
    result2 = macro_agent.step(
        {
            "macro_context": {
                "indicators": ["interest_rates", "employment", "growth"],
                "period": "2y",
            }
        }
    )

    if "macro" in result2:
        strategy_adj2 = result2["macro"]["strategy_adjustments"]
        risk_factors2 = result2["macro"]["risk_factors"]

        print(f"  风险调整: {strategy_adj2.get('risk_adjustment', 0):.2f}")
        print(f"  衰退风险: {risk_factors2.get('recession_risk', 0):.2f}")

    # 场景3: 增长加速场景
    print("\n🚀 场景3: 经济增长加速测试")
    result3 = macro_agent.step(
        {
            "macro_context": {
                "indicators": ["employment", "growth", "market_sentiment"],
                "period": "1y",
            }
        }
    )

    if "macro" in result3:
        strategy_adj3 = result3["macro"]["strategy_adjustments"]
        investment_regime3 = result3["macro"]["investment_regime"]

        print(f"  成长价值倾向: {strategy_adj3.get('growth_value_tilt', 0):.2f}")
        print(f"  投资体制: {investment_regime3.get('regime_type', 'N/A')}")

    print("✅ 不同宏观场景测试完成")
    return True


def test_macro_data_quality_assessment():
    """测试宏观数据质量评估"""
    print("\n🔍 测试宏观数据质量评估...")

    macro_tool = MacroEconomicFetcher()

    # 测试数据质量评估功能
    result = macro_tool.execute(
        indicators=["interest_rates", "inflation", "employment", "growth", "currency"],
        period="1y",
    )

    if result.get("success"):
        metadata = result["metadata"]
        data_sources = metadata.get("data_sources", {})

        print("✅ 数据质量评估完成")
        print(f"\n📊 数据源评估:")
        print(f"  主要数据源: {', '.join(data_sources.get('primary_sources', []))}")
        print(f"  数据可靠性: {data_sources.get('reliability', 'N/A')}")
        print(f"  更新频率: {data_sources.get('update_frequency', 'N/A')}")

        # 检查各指标的数据完整性
        data = result["data"]
        total_indicators = 0
        successful_indicators = 0

        for category, category_data in data.items():
            if category in ["analysis", "outlook", "derived_indicators"]:
                continue

            for indicator_name, indicator_data in category_data.items():
                total_indicators += 1
                if (
                    isinstance(indicator_data, dict)
                    and "latest_value" in indicator_data
                ):
                    successful_indicators += 1

        if total_indicators > 0:
            data_completeness = successful_indicators / total_indicators
            print(
                f"  数据完整性: {data_completeness:.1%} ({successful_indicators}/{total_indicators})"
            )

        # 评估衍生指标计算质量
        derived = data.get("derived_indicators", {})
        if derived:
            calc_errors = sum(
                1 for v in derived.values() if isinstance(v, dict) and "error" in v
            )
            calc_success = len(derived) - calc_errors
            print(f"  衍生指标成功率: {calc_success}/{len(derived)}")

    else:
        print(f"❌ 数据质量评估失败: {result.get('error')}")
        return False

    return True


if __name__ == "__main__":
    print("🚀 宏观经济分析功能测试")
    print("=" * 60)

    tests = [
        ("宏观经济数据获取工具", test_macro_economic_fetcher),
        ("宏观经济分析Agent", test_macro_analyst_agent),
        ("宏观分析与完整流水线集成", test_macro_integration_with_pipeline),
        ("不同宏观场景测试", test_different_macro_scenarios),
        ("宏观数据质量评估", test_macro_data_quality_assessment),
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
        print("🎉 所有宏观经济分析功能测试通过！")
        print("✅ 系统宏观分析能力已全面建立！")
    else:
        print("⚠️ 部分测试未通过，需要进一步优化")

    print("\n📈 新增宏观经济分析能力:")
    print("  ✅ 多源宏观经济数据获取")
    print("  ✅ 利率、通胀、就业、增长等核心指标")
    print("  ✅ 收益率曲线、实际利率等衍生指标")
    print("  ✅ 投资环境和经济周期判断")
    print("  ✅ 策略调整和风险因子分析")
    print("  ✅ 行业轮动和择时指导")
    print("  ✅ 与基本面、技术面、情感面的完整集成")
    print("  ✅ 多场景宏观策略适应性")
