#!/usr/bin/env python3
"""
测试增强后的技术指标库
"""

import numpy as np

from app.tool.indicators import TechnicalIndicators


def test_enhanced_indicators():
    """测试新增的技术指标"""
    print("🔍 测试增强后的技术指标库...")

    # 生成模拟的OHLC数据
    np.random.seed(42)  # 确保结果可重现
    n_points = 50

    base_price = 100
    prices = []
    highs = []
    lows = []

    for i in range(n_points):
        # 模拟价格随机游走
        change = np.random.normal(0, 2)
        base_price += change

        # 生成OHLC数据
        high = base_price + abs(np.random.normal(0, 1))
        low = base_price - abs(np.random.normal(0, 1))
        close = base_price + np.random.normal(0, 0.5)

        prices.append(close)
        highs.append(high)
        lows.append(low)

    print(f"✅ 生成了 {len(prices)} 个价格数据点")
    print(f"  价格范围: ${min(prices):.2f} - ${max(prices):.2f}")

    # 测试增强的技术指标
    indicators_tool = TechnicalIndicators()

    # 测试1: 基础指标
    print("\n📊 测试基础指标...")
    result1 = indicators_tool.execute(
        prices=prices, indicators=["sma", "ema", "rsi", "macd"]
    )

    if result1.get("success"):
        print("✅ 基础指标计算成功:")
        print(f"  SMA 最后值: {result1['indicators']['sma'][-1]:.2f}")
        print(f"  EMA 最后值: {result1['indicators']['ema'][-1]:.2f}")
        print(f"  RSI 最后值: {result1['indicators']['rsi'][-1]:.2f}")
        print(f"  MACD 最后值: {result1['indicators']['macd']['macd'][-1]:.4f}")
    else:
        print(f"❌ 基础指标计算失败: {result1.get('error')}")
        return False

    # 测试2: 布林带
    print("\n📈 测试布林带...")
    result2 = indicators_tool.execute(
        prices=prices, indicators=["bollinger"], bb_period=20, bb_std=2.0
    )

    if result2.get("success"):
        bb = result2["indicators"]["bollinger"]
        print("✅ 布林带计算成功:")
        print(f"  上轨: {bb['upper'][-1]:.2f}")
        print(f"  中轨: {bb['middle'][-1]:.2f}")
        print(f"  下轨: {bb['lower'][-1]:.2f}")

        # 验证布林带逻辑
        assert bb["upper"][-1] > bb["middle"][-1] > bb["lower"][-1], "布林带顺序错误"
        print("✅ 布林带逻辑验证通过")
    else:
        print(f"❌ 布林带计算失败: {result2.get('error')}")
        return False

    # 测试3: 随机指标 (需要高低价)
    print("\n📉 测试随机指标...")
    result3 = indicators_tool.execute(
        prices=prices,
        high_prices=highs,
        low_prices=lows,
        indicators=["stochastic"],
        stoch_k=14,
        stoch_d=3,
    )

    if result3.get("success"):
        stoch = result3["indicators"]["stochastic"]
        print("✅ 随机指标计算成功:")
        print(f"  %K 最后值: {stoch['k'][-1]:.2f}")
        print(f"  %D 最后值: {stoch['d'][-1]:.2f}")

        # 验证随机指标范围
        last_k = stoch["k"][-1]
        last_d = stoch["d"][-1]
        if not pd.isna(last_k):
            assert 0 <= last_k <= 100, f"K值超出范围: {last_k}"
        if not pd.isna(last_d):
            assert 0 <= last_d <= 100, f"D值超出范围: {last_d}"
        print("✅ 随机指标范围验证通过")
    else:
        print(f"❌ 随机指标计算失败: {result3.get('error')}")
        return False

    # 测试4: Williams %R
    print("\n📊 测试威廉姆斯%R...")
    result4 = indicators_tool.execute(
        prices=prices,
        high_prices=highs,
        low_prices=lows,
        indicators=["williams_r"],
        williams_r=14,
    )

    if result4.get("success"):
        williams = result4["indicators"]["williams_r"]
        print("✅ Williams %R 计算成功:")
        print(f"  最后值: {williams[-1]:.2f}")

        # 验证Williams %R范围
        last_wr = williams[-1]
        if not pd.isna(last_wr):
            assert -100 <= last_wr <= 0, f"Williams %R超出范围: {last_wr}"
        print("✅ Williams %R 范围验证通过")
    else:
        print(f"❌ Williams %R 计算失败: {result4.get('error')}")
        return False

    # 测试5: MACD信号分析
    print("\n🔄 测试MACD信号分析...")
    result5 = indicators_tool.execute(prices=prices, indicators=["macd_signals"])

    if result5.get("success"):
        macd_signals = result5["indicators"]["macd_signals"]
        print("✅ MACD信号分析成功:")
        print(f"  最新信号: {macd_signals['latest_signal']}")
        print(f"  信号强度: {macd_signals['signal_strength']:.4f}")
        print(f"  信号数量: {len(macd_signals['signals'])}")

        # 验证信号类型
        valid_signals = [
            "neutral",
            "bullish_crossover",
            "bearish_crossover",
            "bullish_momentum",
            "bearish_momentum",
        ]
        assert macd_signals["latest_signal"] in valid_signals, "无效的MACD信号"
        print("✅ MACD信号类型验证通过")
    else:
        print(f"❌ MACD信号分析失败: {result5.get('error')}")
        return False

    # 测试6: 综合指标测试
    print("\n🎯 测试综合指标...")
    result6 = indicators_tool.execute(
        prices=prices,
        high_prices=highs,
        low_prices=lows,
        indicators=["sma", "bollinger", "stochastic", "williams_r", "macd_signals"],
    )

    if result6.get("success"):
        indicators_data = result6["indicators"]
        print("✅ 综合指标计算成功:")
        print(f"  包含指标: {list(indicators_data.keys())}")

        # 生成交易建议
        suggestions = generate_trading_suggestions(indicators_data, prices[-1])
        print(f"  交易建议: {suggestions}")

    else:
        print(f"❌ 综合指标计算失败: {result6.get('error')}")
        return False

    return True


def generate_trading_suggestions(indicators: dict, current_price: float) -> str:
    """基于多个技术指标生成交易建议"""
    signals = []

    # 布林带信号
    if "bollinger" in indicators:
        bb = indicators["bollinger"]
        if current_price > bb["upper"][-1]:
            signals.append("超买(布林带)")
        elif current_price < bb["lower"][-1]:
            signals.append("超卖(布林带)")

    # 随机指标信号
    if "stochastic" in indicators:
        stoch = indicators["stochastic"]
        k_val = stoch["k"][-1]
        if not pd.isna(k_val):
            if k_val > 80:
                signals.append("超买(随机指标)")
            elif k_val < 20:
                signals.append("超卖(随机指标)")

    # Williams %R信号
    if "williams_r" in indicators:
        wr = indicators["williams_r"][-1]
        if not pd.isna(wr):
            if wr > -20:
                signals.append("超买(Williams%R)")
            elif wr < -80:
                signals.append("超卖(Williams%R)")

    # MACD信号
    if "macd_signals" in indicators:
        macd_signal = indicators["macd_signals"]["latest_signal"]
        if "bullish" in macd_signal:
            signals.append("看涨(MACD)")
        elif "bearish" in macd_signal:
            signals.append("看跌(MACD)")

    if not signals:
        return "中性"
    elif len([s for s in signals if "超买" in s or "看跌" in s]) > len(
        [s for s in signals if "超卖" in s or "看涨" in s]
    ):
        return f"偏空 ({', '.join(signals)})"
    elif len([s for s in signals if "超卖" in s or "看涨" in s]) > len(
        [s for s in signals if "超买" in s or "看跌" in s]
    ):
        return f"偏多 ({', '.join(signals)})"
    else:
        return f"中性 ({', '.join(signals)})"


if __name__ == "__main__":
    import pandas as pd

    print("🚀 技术指标增强功能测试")
    print("=" * 50)

    success = test_enhanced_indicators()

    print("\n" + "=" * 50)
    if success:
        print("🎉 所有技术指标测试通过！")
        print("✅ 系统技术分析能力已大幅增强！")
    else:
        print("❌ 部分测试失败")

    print("\n📈 新增指标能力:")
    print("  ✅ 布林带 (Bollinger Bands)")
    print("  ✅ 随机指标 (Stochastic Oscillator)")
    print("  ✅ 威廉姆斯%R (Williams %R)")
    print("  ✅ MACD信号分析 (Enhanced MACD)")
    print("  ✅ 综合交易建议生成")
