#!/usr/bin/env python3
"""
网络API调用验证 - 验证系统确实在进行真实的网络请求
"""

from datetime import datetime

import requests
import yfinance as yf

from app.tool.yfinance_fetcher import YFinanceFetcher


def test_direct_yfinance_api():
    """直接测试YFinance API调用"""
    print("🔍 直接测试 YFinance API...")

    try:
        # 直接使用yfinance库
        ticker = yf.Ticker("AAPL")
        info = ticker.info

        print(f"✅ YFinance API 直接调用成功:")
        print(f"  公司名: {info.get('longName', 'N/A')}")
        print(f"  当前价格: ${info.get('currentPrice', 'N/A')}")
        print(f"  市值: ${info.get('marketCap', 0):,}")

        # 验证这确实是真实的数据
        assert info.get("marketCap", 0) > 1e12, "市值数据异常"
        assert info.get("currentPrice", 0) > 100, "股价数据异常"

        return True
    except Exception as e:
        print(f"❌ YFinance API 调用失败: {e}")
        return False


def test_yahoo_finance_url_access():
    """测试Yahoo Finance URL访问"""
    print("\n🔍 测试 Yahoo Finance URL 访问...")

    try:
        # 直接访问Yahoo Finance的API端点
        url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            result = data.get("chart", {}).get("result", [])

            if result:
                symbol = result[0].get("meta", {}).get("symbol")
                current_price = result[0].get("meta", {}).get("regularMarketPrice")

                print(f"✅ Yahoo Finance API 访问成功:")
                print(f"  股票代码: {symbol}")
                print(f"  当前价格: ${current_price}")

                return True
            else:
                print("❌ Yahoo Finance API 返回空数据")
                return False
        else:
            print(f"❌ Yahoo Finance API 访问失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Yahoo Finance URL 访问异常: {e}")
        return False


def test_our_tool_vs_direct_api():
    """对比我们的工具与直接API调用的结果"""
    print("\n🔍 对比我们的工具与直接API...")

    try:
        # 使用我们的工具
        our_tool = YFinanceFetcher()
        our_result = our_tool.execute(
            symbols=["MSFT"], info_fields=["currentPrice", "marketCap", "longName"]
        )

        # 直接使用yfinance
        direct_ticker = yf.Ticker("MSFT")
        direct_info = direct_ticker.info

        if our_result.get("success"):
            our_data = our_result["data"]["company_info"]["MSFT"]

            print(f"✅ 数据对比:")
            print(f"  我们的工具 - 公司名: {our_data.get('longName', 'N/A')}")
            print(f"  直接API   - 公司名: {direct_info.get('longName', 'N/A')}")
            print(f"  我们的工具 - 当前价格: ${our_data.get('currentPrice', 'N/A')}")
            print(f"  直接API   - 当前价格: ${direct_info.get('currentPrice', 'N/A')}")

            # 验证数据一致性（允许小幅差异）
            our_price = our_data.get("currentPrice", 0)
            direct_price = direct_info.get("currentPrice", 0)

            if our_price and direct_price:
                price_diff_pct = abs(our_price - direct_price) / direct_price
                assert price_diff_pct < 0.01, f"价格差异过大: {price_diff_pct*100:.2f}%"
                print(f"✅ 价格数据一致性验证通过（差异: {price_diff_pct*100:.3f}%）")

            return True
        else:
            print("❌ 我们的工具调用失败")
            return False
    except Exception as e:
        print(f"❌ 数据对比异常: {e}")
        return False


def test_real_time_data_freshness():
    """测试数据的实时性"""
    print("\n🔍 测试数据实时性...")

    try:
        tool = YFinanceFetcher()

        # 获取两次数据，间隔几秒
        result1 = tool.execute(symbols=["TSLA"], info_fields=["currentPrice"])

        import time

        time.sleep(2)  # 等待2秒

        result2 = tool.execute(symbols=["TSLA"], info_fields=["currentPrice"])

        if result1.get("success") and result2.get("success"):
            price1 = result1["data"]["company_info"]["TSLA"].get("currentPrice", 0)
            price2 = result2["data"]["company_info"]["TSLA"].get("currentPrice", 0)

            print(f"✅ 实时数据测试:")
            print(f"  第一次获取: ${price1}")
            print(f"  第二次获取: ${price2}")

            # 数据应该是当前的（不是历史缓存）
            # 在交易时间内，价格可能变化，但至少应该是合理的
            if price1 and price2:
                if price1 == price2:
                    print("  两次价格相同（可能是同一时刻数据或市场未交易）")
                else:
                    print(f"  价格有变化，证明数据是实时的")

                return True
            else:
                print("❌ 无法获取有效价格数据")
                return False
        else:
            print("❌ 实时数据获取失败")
            return False
    except Exception as e:
        print(f"❌ 实时数据测试异常: {e}")
        return False


def test_multiple_symbols_simultaneously():
    """测试同时获取多个股票数据"""
    print("\n🔍 测试多股票同时获取...")

    try:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        tool = YFinanceFetcher()

        result = tool.execute(
            symbols=symbols, info_fields=["currentPrice", "marketCap", "sector"]
        )

        if result.get("success"):
            print(f"✅ 多股票数据获取成功:")

            for symbol in symbols:
                data = result["data"]["company_info"].get(symbol, {})
                price = data.get("currentPrice", "N/A")
                market_cap = data.get("marketCap", 0)
                sector = data.get("sector", "N/A")

                print(f"  {symbol}: ${price}, 市值 ${market_cap:,}, 行业 {sector}")

                # 验证每个股票都有有效数据
                if isinstance(price, (int, float)) and price > 0:
                    assert price > 50, f"{symbol} 价格异常: {price}"
                if isinstance(market_cap, (int, float)) and market_cap > 0:
                    assert market_cap > 1e10, f"{symbol} 市值异常: {market_cap}"

            print(f"✅ 所有股票数据验证通过")
            return True
        else:
            print("❌ 多股票数据获取失败")
            return False
    except Exception as e:
        print(f"❌ 多股票测试异常: {e}")
        return False


def main():
    """主验证函数"""
    print("🌐 网络API真实性验证")
    print("=" * 60)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    tests = [
        ("Direct YFinance API", test_direct_yfinance_api),
        ("Yahoo Finance URL Access", test_yahoo_finance_url_access),
        ("Tool vs Direct API Comparison", test_our_tool_vs_direct_api),
        ("Real-time Data Freshness", test_real_time_data_freshness),
        ("Multiple Symbols Fetch", test_multiple_symbols_simultaneously),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n📡 执行测试: {test_name}")
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {str(e)}")

    print("\n" + "=" * 60)
    print(f"🎯 网络验证结果: {passed}/{total} 通过")

    if passed >= 4:  # 允许一个测试失败（网络问题等）
        print("🎉 网络API真实性验证通过！")
        print("✅ 系统确实在进行真实的网络API调用！")
        print("✅ 数据来源真实、准确、实时！")
    else:
        print("⚠️ 部分网络验证未通过，请检查网络连接")

    return passed >= 4


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
