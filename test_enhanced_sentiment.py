#!/usr/bin/env python3
"""
测试增强的新闻情感分析功能
"""

from app.agent.sentiment_analyzer import SentimentAnalyzerAgent
from app.tool.enhanced_news_fetcher import EnhancedNewsFetcher


def test_enhanced_news_api():
    """测试增强的NewsAPI工具"""
    print("🔍 测试增强的NewsAPI工具...")

    news_tool = EnhancedNewsFetcher()

    # 测试获取多只股票的新闻
    symbols = ["AAPL", "MSFT", "TSLA"]

    result = news_tool.execute(symbols=symbols, max_articles=5, days_back=7)

    if result.get("success"):
        print("✅ NewsAPI工具执行成功")
        data = result["data"]

        for symbol in symbols:
            symbol_data = data.get(symbol, {})
            print(f"\n📊 {symbol} 分析结果:")
            print(f"  情感分数: {symbol_data.get('sentiment_score', 0):.3f}")
            print(
                f"  时间权重分数: {symbol_data.get('time_weighted_sentiment', 0):.3f}"
            )
            print(f"  相关性分数: {symbol_data.get('relevance_score', 0):.3f}")
            print(f"  新闻数量: {symbol_data.get('article_count', 0)}")

            # 显示情感分布
            breakdown = symbol_data.get("sentiment_breakdown", {})
            print(
                f"  情感分布: 积极{breakdown.get('positive', 0):.1%} | "
                f"中性{breakdown.get('neutral', 0):.1%} | "
                f"消极{breakdown.get('negative', 0):.1%}"
            )

            # 显示关键词分析
            keyword_analysis = symbol_data.get("keyword_sentiment", {})
            if keyword_analysis:
                print(f"  关键词分析: {len(keyword_analysis)} 个关键词")
                for keyword, data in list(keyword_analysis.items())[:3]:
                    print(
                        f"    {keyword}: {data['avg_sentiment']:.3f} (提及{data['mentions']}次)"
                    )

            # 显示新闻标题
            articles = symbol_data.get("articles", [])
            print(f"  新闻标题:")
            for i, article in enumerate(articles[:3]):
                title = article.get("title", "")
                source = article.get("source", "")
                print(f"    {i+1}. {title[:80]}... ({source})")

    else:
        print(f"❌ NewsAPI工具执行失败: {result.get('error')}")
        return False

    return True


def test_enhanced_sentiment_agent():
    """测试增强的情感分析Agent"""
    print("\n🤖 测试增强的SentimentAnalyzer Agent...")

    # 模拟从MarketAnalyst传来的数据
    mock_analyst_data = {
        "analyst": {
            "tickers": [
                {
                    "symbol": "AAPL",
                    "industry": "Technology",
                    "market_cap": 3000000000000,
                },
                {
                    "symbol": "MSFT",
                    "industry": "Technology",
                    "market_cap": 2800000000000,
                },
            ],
            "rationale": {
                "AAPL": "Growth stock with strong fundamentals",
                "MSFT": "Stable growth with cloud expansion",
            },
        }
    }

    sentiment_agent = SentimentAnalyzerAgent()
    result = sentiment_agent.step(mock_analyst_data)

    if "sentiment" in result:
        sentiment_data = result["sentiment"]

        print("✅ SentimentAnalyzer Agent 执行成功")
        print(f"📊 情感分数: {sentiment_data.get('scores', {})}")

        # 显示增强的源信息
        sources = sentiment_data.get("sources", {})
        for symbol, source_list in sources.items():
            print(f"\n📰 {symbol} 新闻源:")
            for source in source_list:
                print(f"  {source}")

        # 显示元数据（如果有）
        metadata = sentiment_data.get("metadata", {})
        if metadata:
            print(f"\n📈 元数据分析:")
            for symbol, meta in metadata.items():
                print(f"  {symbol}:")
                print(f"    置信度: {meta.get('confidence', 0):.3f}")
                print(f"    文章数量: {meta.get('article_count', 0)}")
                breakdown = meta.get("breakdown", {})
                if breakdown:
                    print(
                        f"    情感分布: 积极{breakdown.get('positive', 0):.1%} | "
                        f"中性{breakdown.get('neutral', 0):.1%} | "
                        f"消极{breakdown.get('negative', 0):.1%}"
                    )

        return True
    else:
        print("❌ SentimentAnalyzer Agent 执行失败")
        return False


def test_sentiment_integration_with_pipeline():
    """测试情感分析在完整流水线中的集成"""
    print("\n🔄 测试情感分析流水线集成...")

    from app.agent.market_analyst import MarketAnalystAgent
    from app.agent.technical_trader import TechnicalTraderAgent

    # 1. MarketAnalyst 选股
    market_agent = MarketAnalystAgent()
    analyst_result = market_agent.step(
        {"universe": {"max_candidates": 2, "objective": "growth"}}
    )

    print(
        f"✅ 市场分析完成，选择股票: {[t['symbol'] for t in analyst_result['analyst']['tickers']]}"
    )

    # 2. SentimentAnalyzer 情感分析
    sentiment_agent = SentimentAnalyzerAgent()
    sentiment_result = sentiment_agent.step(analyst_result)

    sentiment_scores = sentiment_result["sentiment"]["scores"]
    print(f"✅ 情感分析完成，分数: {sentiment_scores}")

    # 3. TechnicalTrader 技术分析
    tech_agent = TechnicalTraderAgent()
    combined_input = {**analyst_result, **sentiment_result}
    tech_result = tech_agent.step(combined_input)

    tech_signals = tech_result["technical"]["signals"]
    print(f"✅ 技术分析完成，信号: {tech_signals}")

    # 4. 综合分析
    print(f"\n📊 综合分析结果:")
    for symbol in sentiment_scores.keys():
        sentiment_score = sentiment_scores[symbol]
        tech_signal = tech_signals.get(symbol, {})

        print(f"  {symbol}:")
        print(f"    情感分数: {sentiment_score:.3f}")
        print(
            f"    技术信号: {tech_signal.get('action', 'N/A')} (置信度: {tech_signal.get('confidence', 0):.3f})"
        )

        # 简单的综合评分
        sentiment_weight = 0.3
        tech_weight = 0.7

        tech_score = {"long": 1, "flat": 0, "short": -1}.get(
            tech_signal.get("action"), 0
        )
        tech_confidence = tech_signal.get("confidence", 0)

        combined_score = (
            sentiment_score * sentiment_weight
            + tech_score * tech_confidence * tech_weight
        )

        print(f"    综合评分: {combined_score:.3f}")

        if combined_score > 0.2:
            recommendation = "买入"
        elif combined_score < -0.2:
            recommendation = "卖出"
        else:
            recommendation = "持有"

        print(f"    建议: {recommendation}")

    return True


def test_sentiment_accuracy_validation():
    """验证情感分析的准确性"""
    print("\n🎯 验证情感分析准确性...")

    news_tool = EnhancedNewsFetcher()

    # 测试已知情感倾向的文本
    test_cases = [
        {
            "text": "Company reports record profits and beats expectations significantly",
            "expected_sentiment": "positive",
        },
        {
            "text": "Stock plunges amid concerns over regulatory challenges and market headwinds",
            "expected_sentiment": "negative",
        },
        {
            "text": "Quarterly results meet analyst expectations with stable performance",
            "expected_sentiment": "neutral",
        },
    ]

    correct_predictions = 0

    for i, case in enumerate(test_cases):
        # 模拟文章数据
        mock_articles = [
            {
                "title": case["text"],
                "description": case["text"],
                "content": case["text"],
                "published_at": "2024-01-01T00:00:00Z",
                "source": "Test Source",
            }
        ]

        # 分析情感
        sentiment_analysis = news_tool._analyze_sentiment_enhanced(
            mock_articles, "TEST"
        )
        sentiment_score = sentiment_analysis["overall_score"]

        # 判断预测结果
        if sentiment_score > 0.1:
            predicted = "positive"
        elif sentiment_score < -0.1:
            predicted = "negative"
        else:
            predicted = "neutral"

        is_correct = predicted == case["expected_sentiment"]
        if is_correct:
            correct_predictions += 1

        print(f"  测试 {i+1}: {case['text'][:50]}...")
        print(
            f"    预期: {case['expected_sentiment']}, 实际: {predicted} (分数: {sentiment_score:.3f})"
        )
        print(f"    结果: {'✅ 正确' if is_correct else '❌ 错误'}")

    accuracy = correct_predictions / len(test_cases)
    print(f"\n📊 准确率: {accuracy:.1%} ({correct_predictions}/{len(test_cases)})")

    return accuracy >= 0.6  # 要求至少60%准确率


if __name__ == "__main__":
    print("🚀 增强新闻情感分析测试")
    print("=" * 60)

    tests = [
        ("NewsAPI工具测试", test_enhanced_news_api),
        ("SentimentAnalyzer Agent测试", test_enhanced_sentiment_agent),
        ("流水线集成测试", test_sentiment_integration_with_pipeline),
        ("情感分析准确性验证", test_sentiment_accuracy_validation),
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
        print("🎉 所有增强功能测试通过！")
        print("✅ 新闻情感分析系统已大幅提升！")
    else:
        print("⚠️ 部分测试未通过，需要进一步优化")

    print("\n📈 新增情感分析能力:")
    print("  ✅ 多源新闻聚合")
    print("  ✅ 时间权重情感分析")
    print("  ✅ 关键词情感分析")
    print("  ✅ 相关性评分")
    print("  ✅ 情感置信度评估")
    print("  ✅ 增强的上下文理解")
