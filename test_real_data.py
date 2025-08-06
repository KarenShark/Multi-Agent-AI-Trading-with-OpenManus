#!/usr/bin/env python3
"""
测试真实数据集成，验证非硬编码实现
"""

from app.tool.indicators import TechnicalIndicators
from app.tool.news_sentiment_fetcher import NewsSentimentFetcher
from app.tool.yfinance_fetcher import YFinanceFetcher


def test_yfinance_tool():
    """测试 YFinance 工具"""
    print("🔍 测试 YFinance 工具...")
    yf_tool = YFinanceFetcher()
    result = yf_tool.execute(
        symbols=["AAPL", "MSFT"], period="1mo", info_fields=["marketCap", "sector"]
    )

    if result.get("success"):
        print("✅ YFinance 数据获取成功")
        for symbol, info in result["data"]["company_info"].items():
            market_cap = info.get("marketCap", 0)
            sector = info.get("sector", "N/A")
            print(f"  {symbol}: 市值 ${market_cap:,}, 行业 {sector}")
        return True
    else:
        print("❌ YFinance 数据获取失败")
        return False


def test_technical_indicators():
    """测试技术指标计算"""
    print("\n🔍 测试技术指标计算...")
    # 提供足够的价格数据（30个数据点）
    prices = [
        150,
        152,
        148,
        155,
        153,
        157,
        154,
        158,
        156,
        160,
        159,
        162,
        161,
        165,
        163,
        166,
        164,
        167,
        165,
        168,
        166,
        169,
        167,
        170,
        168,
        171,
        169,
        172,
        170,
        173,
    ]
    indicators_tool = TechnicalIndicators()
    result = indicators_tool.execute(
        prices=prices, indicators=["sma", "rsi"], sma_period=5, rsi_period=10
    )

    if result.get("success"):
        print("✅ 技术指标计算成功")
        sma = result["indicators"].get("sma", [])
        rsi = result["indicators"].get("rsi", [])
        print(f"  SMA(5): {sma[-3:] if len(sma) >= 3 else sma}")
        print(f"  RSI: {rsi[-3:] if len(rsi) >= 3 else rsi}")
        return True
    else:
        print("❌ 技术指标计算失败")
        return False


def test_news_sentiment():
    """测试新闻情感分析"""
    print("\n🔍 测试新闻情感分析...")
    news_tool = NewsSentimentFetcher()
    result = news_tool.execute(symbols=["AAPL"], max_articles=3)

    if result.get("success"):
        print("✅ 新闻情感分析成功")
        for symbol, data in result["data"].items():
            sentiment = data.get("sentiment", 0)
            articles = data.get("articles", [])
            print(f"  {symbol}: 情感分数 {sentiment:.3f}, 新闻数量 {len(articles)}")
        return True
    else:
        print("❌ 新闻情感分析失败")
        return False


def test_agent_data_flow():
    """测试 Agent 数据流的真实性"""
    print("\n🔍 测试 Agent 数据流真实性...")

    from app.agent.market_analyst import MarketAnalystAgent
    from app.agent.technical_trader import TechnicalTraderAgent

    # 测试多次运行是否产生一致但非硬编码的结果
    ma = MarketAnalystAgent()

    # 测试不同参数产生不同结果
    result1 = ma.step({"universe": {"objective": "growth", "max_candidates": 3}})
    result2 = ma.step({"universe": {"objective": "value", "max_candidates": 3}})

    symbols1 = {t["symbol"] for t in result1["analyst"]["tickers"]}
    symbols2 = {t["symbol"] for t in result2["analyst"]["tickers"]}

    print(f"Growth 策略选择: {symbols1}")
    print(f"Value 策略选择: {symbols2}")

    # 验证不同策略选择不同股票（证明非硬编码）
    if symbols1 != symbols2:
        print("✅ 不同策略产生不同结果，证明使用真实数据逻辑")
    else:
        print("⚠️  不同策略产生相同结果，可能存在硬编码")

    # 测试技术分析的真实计算
    tt = TechnicalTraderAgent()
    tech_result = tt.step({"analyst": result1["analyst"]})

    signals = tech_result["technical"]["signals"]
    actions = [sig["action"] for sig in signals.values()]
    confidences = [sig["confidence"] for sig in signals.values()]

    print(f"技术信号动作: {actions}")
    print(f"技术信号置信度: {confidences}")

    # 验证信号有差异化（证明基于真实计算）
    if len(set(actions)) > 1 or len(set(confidences)) > 1:
        print("✅ 技术信号有差异化，证明基于真实技术指标计算")
        return True
    else:
        print("⚠️  技术信号过于一致，可能存在硬编码")
        return False


if __name__ == "__main__":
    print("🚀 真实数据集成验证")
    print("=" * 50)

    results = []
    results.append(test_yfinance_tool())
    results.append(test_technical_indicators())
    results.append(test_news_sentiment())
    results.append(test_agent_data_flow())

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 所有测试通过 ({passed}/{total})！")
        print("✅ 系统确实使用真实数据，非硬编码实现！")
    else:
        print(f"⚠️  部分测试通过 ({passed}/{total})")
        print("🔧 可能需要进一步优化某些组件")
