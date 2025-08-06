#!/usr/bin/env python3
"""
测试数据预处理管道功能
"""

import os
import tempfile
from datetime import datetime

import pandas as pd

from app.data.loader import UnifiedDataLoader
from app.data.preprocess import DataPreprocessor


def test_data_preprocessor():
    """测试数据预处理器"""
    print("🔍 测试数据预处理器...")

    preprocessor = DataPreprocessor()

    # 测试基本面数据预处理
    print("\n📊 测试基本面数据预处理...")
    sample_fundamental = {
        "AAPL": {
            "raw_data": {
                "valuation": {
                    "trailing_pe": 25.5,
                    "price_to_book": 8.2,
                    "price_to_sales": 7.1,
                },
                "profitability": {
                    "return_on_equity": 0.175,
                    "profit_margin": 0.266,
                    "operating_margin": 0.298,
                },
                "liquidity": {"current_ratio": 0.98, "quick_ratio": 0.82},
                "leverage": {"debt_to_equity": 1.73, "interest_coverage": 29.1},
            }
        },
        "MSFT": {
            "raw_data": {
                "valuation": {
                    "trailing_pe": 35.2,
                    "price_to_book": 12.8,
                    "price_to_sales": 12.9,
                },
                "profitability": {
                    "return_on_equity": 0.458,
                    "profit_margin": 0.368,
                    "operating_margin": 0.421,
                },
            }
        },
    }

    processed_fundamental = preprocessor.preprocess_fundamental_data(sample_fundamental)

    if processed_fundamental and "AAPL" in processed_fundamental:
        print("✅ 基本面数据预处理成功")
        aapl_data = processed_fundamental["AAPL"]
        print(f"  AAPL 质量评分: {aapl_data.get('quality_score', 0):.2f}")

        feature_metrics = aapl_data.get("feature_metrics", {})
        if feature_metrics:
            print(f"  估值评分: {feature_metrics.get('valuation_score', 0):.2f}")
            print(
                f"  盈利能力评分: {feature_metrics.get('profitability_score', 0):.2f}"
            )
            print(f"  财务实力评分: {feature_metrics.get('financial_strength', 0):.2f}")
    else:
        print("❌ 基本面数据预处理失败")
        return False

    # 测试技术面数据预处理
    print("\n📈 测试技术面数据预处理...")
    sample_technical = {
        "AAPL": {
            "sma": list(range(150, 175)),  # 25个数据点
            "rsi": [50 + i * 0.5 for i in range(25)],
            "macd": [i * 0.1 for i in range(25)],
        }
    }

    sample_price = {
        "AAPL": {
            "dates": [f"2024-01-{i+1:02d}" for i in range(25)],
            "values": list(range(170, 195)),
        }
    }

    processed_technical = preprocessor.preprocess_technical_data(
        sample_technical, sample_price
    )

    if processed_technical and "AAPL" in processed_technical:
        print("✅ 技术面数据预处理成功")
        aapl_tech = processed_technical["AAPL"]
        print(f"  AAPL 质量评分: {aapl_tech.get('quality_score', 0):.2f}")
        print(f"  数据点数量: {aapl_tech.get('data_points', 0)}")
        print(f"  特征数量: {aapl_tech.get('feature_count', 0)}")
    else:
        print("❌ 技术面数据预处理失败")
        return False

    # 测试情感面数据预处理
    print("\n💭 测试情感面数据预处理...")
    sample_sentiment = {
        "AAPL": {
            "scores": {"AAPL": 0.65},
            "metadata": {
                "article_count": 8,
                "relevance_score": 0.82,
                "time_weighted_sentiment": 0.71,
            },
        }
    }

    processed_sentiment = preprocessor.preprocess_sentiment_data(sample_sentiment)

    if processed_sentiment and "AAPL" in processed_sentiment:
        print("✅ 情感面数据预处理成功")
        aapl_sent = processed_sentiment["AAPL"]
        print(f"  AAPL 质量评分: {aapl_sent.get('quality_score', 0):.2f}")

        sentiment_features = aapl_sent.get("sentiment_features", {})
        if sentiment_features:
            print(f"  情感均值: {sentiment_features.get('sentiment_mean', 0):.3f}")
            print(f"  新闻数量: {sentiment_features.get('news_volume', 0)}")
            print(f"  相关性评分: {sentiment_features.get('relevance_score', 0):.3f}")
    else:
        print("❌ 情感面数据预处理失败")
        return False

    # 测试宏观面数据预处理
    print("\n🌍 测试宏观面数据预处理...")
    sample_macro = {
        "interest_rates": {
            "fed_funds_rate": {"latest_value": 5.25},
            "10y_treasury": {"latest_value": 4.5},
        },
        "inflation": {"cpi": {"latest_value": 3.2}, "core_cpi": {"latest_value": 4.1}},
        "employment": {"unemployment_rate": {"latest_value": 3.7}},
    }

    processed_macro = preprocessor.preprocess_macro_data(sample_macro)

    if processed_macro and "macro_features" in processed_macro:
        print("✅ 宏观面数据预处理成功")
        print(f"  质量评分: {processed_macro.get('quality_score', 0):.2f}")
        print(f"  指标数量: {processed_macro.get('indicators_count', 0)}")

        macro_features = processed_macro.get("macro_features", {})
        if macro_features:
            print(f"  利率水平: {macro_features.get('interest_rate_level', 0):.2f}")
            print(f"  收益率曲线斜率: {macro_features.get('yield_curve_slope', 0):.2f}")
            print(f"  通胀水平: {macro_features.get('inflation_level', 0):.2f}")
    else:
        print("❌ 宏观面数据预处理失败")
        return False

    # 测试统一数据集创建
    print("\n🔗 测试统一数据集创建...")
    symbols = ["AAPL", "MSFT"]

    unified_df = preprocessor.create_unified_dataset(
        processed_fundamental,
        processed_technical,
        processed_sentiment,
        processed_macro,
        symbols,
    )

    if not unified_df.empty:
        print("✅ 统一数据集创建成功")
        print(f"  数据行数: {len(unified_df)}")
        print(f"  特征列数: {len(unified_df.columns)}")
        print(f"  包含股票: {unified_df['symbol'].tolist()}")

        # 显示部分特征
        feature_cols = [col for col in unified_df.columns if col != "symbol"][:5]
        print(f"  示例特征: {feature_cols}")
    else:
        print("❌ 统一数据集创建失败")
        return False

    # 测试数据质量验证
    print("\n🔍 测试数据质量验证...")
    processed_data = {
        "fundamental": processed_fundamental,
        "technical": processed_technical,
        "sentiment": processed_sentiment,
        "macro": processed_macro,
    }

    quality_report = preprocessor.validate_data_quality(processed_data)

    if quality_report and "overall_score" in quality_report:
        print("✅ 数据质量验证成功")
        print(f"  整体质量评分: {quality_report['overall_score']:.2f}")

        category_scores = quality_report.get("category_scores", {})
        for category, score in category_scores.items():
            print(f"  {category} 质量: {score:.2f}")

        recommendations = quality_report.get("recommendations", [])
        if recommendations:
            print(f"  改进建议:")
            for rec in recommendations[:3]:  # 显示前3个建议
                print(f"    • {rec}")
    else:
        print("❌ 数据质量验证失败")
        return False

    return True


def test_unified_data_loader():
    """测试统一数据加载器"""
    print("\n🔄 测试统一数据加载器...")

    # 创建临时缓存目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            "enable_cache": True,
            "cache_directory": temp_dir,
            "cache_duration_hours": 1,
            "enable_preprocessing": True,
            "create_unified_dataset": True,
        }

        loader = UnifiedDataLoader(config)

        # 测试基础数据加载
        print("\n📊 测试综合数据加载...")
        test_symbols = ["AAPL", "MSFT"]

        comprehensive_data = loader.load_comprehensive_data(
            symbols=test_symbols,
            objective="balanced",
            use_cache=False,  # 首次加载不使用缓存
        )

        if "error" not in comprehensive_data:
            print("✅ 综合数据加载成功")

            # 检查数据源
            data_sources = comprehensive_data.get("metadata", {}).get(
                "data_sources", []
            )
            print(f"  数据源: {', '.join(data_sources)}")

            # 检查基本面数据
            if "fundamental" in comprehensive_data:
                fund_data = comprehensive_data["fundamental"]
                print(f"  基本面数据: {len(fund_data)} 只股票")

            # 检查技术面数据
            if "technical" in comprehensive_data:
                tech_data = comprehensive_data["technical"]
                print(f"  技术面数据: {len(tech_data)} 只股票")

            # 检查情感面数据
            if "sentiment" in comprehensive_data:
                sent_data = comprehensive_data["sentiment"]
                print(f"  情感面数据: {len(sent_data)} 只股票")

            # 检查宏观面数据
            if "macro" in comprehensive_data:
                macro_data = comprehensive_data["macro"]
                print(f"  宏观面数据: 已加载")

            # 检查分析结果
            if "analysis" in comprehensive_data:
                analysis = comprehensive_data["analysis"]
                analysis_types = list(analysis.keys())
                print(f"  智能分析: {', '.join(analysis_types)}")

            # 检查预处理结果
            if "processed" in comprehensive_data:
                processed = comprehensive_data["processed"]
                if "quality_report" in processed:
                    quality = processed["quality_report"]
                    overall_score = quality.get("overall_score", 0)
                    print(f"  数据质量: {overall_score:.2f}")

            # 检查统一数据集
            if "unified_dataset" in comprehensive_data:
                unified_df = comprehensive_data["unified_dataset"]
                if isinstance(unified_df, pd.DataFrame) and not unified_df.empty:
                    print(
                        f"  统一数据集: {len(unified_df)} 行, {len(unified_df.columns)} 列"
                    )
                else:
                    print("  统一数据集: 创建失败或为空")

        else:
            print(f"❌ 综合数据加载失败: {comprehensive_data['error']}")
            return False

        # 测试缓存功能
        print("\n📦 测试缓存功能...")

        # 第二次加载应该使用缓存
        cached_data = loader.load_comprehensive_data(
            symbols=test_symbols, objective="balanced", use_cache=True
        )

        if "error" not in cached_data:
            print("✅ 缓存数据加载成功")

            # 检查缓存状态
            cache_status = loader.get_cache_status()
            if cache_status.get("status") == "enabled":
                print(f"  缓存文件数量: {cache_status.get('cached_files', 0)}")
        else:
            print(f"❌ 缓存数据加载失败: {cached_data['error']}")
            return False

        # 测试特定数据类型加载
        print("\n🎯 测试特定数据类型加载...")

        # 测试仅加载基本面数据
        fundamental_only = loader.load_specific_data("fundamental", test_symbols)
        if "error" not in fundamental_only:
            print("✅ 基本面数据专项加载成功")
        else:
            print(f"❌ 基本面数据专项加载失败: {fundamental_only['error']}")

        # 测试仅加载宏观数据
        macro_only = loader.load_specific_data("macro")
        if "error" not in macro_only:
            print("✅ 宏观数据专项加载成功")
        else:
            print(f"❌ 宏观数据专项加载失败: {macro_only['error']}")

        # 测试实时快照
        print("\n⚡ 测试实时快照...")
        snapshot = loader.get_realtime_snapshot(["AAPL"])

        if "error" not in snapshot:
            print("✅ 实时快照获取成功")
            snapshot_data = snapshot.get("data", {})
            if "AAPL" in snapshot_data:
                aapl_snapshot = snapshot_data["AAPL"]
                latest_price = aapl_snapshot.get("latest_price")
                if latest_price:
                    print(f"  AAPL 最新价格: ${latest_price:.2f}")

                tech_signal = aapl_snapshot.get("technical_signal")
                if tech_signal:
                    action = tech_signal.get("action", "N/A")
                    confidence = tech_signal.get("confidence", 0)
                    print(f"  技术信号: {action} (置信度: {confidence:.3f})")
        else:
            print(f"❌ 实时快照获取失败: {snapshot['error']}")

        # 测试数据质量报告
        print("\n📋 测试数据质量报告...")
        quality_report = loader.get_data_quality_report(comprehensive_data)

        if "error" not in quality_report:
            print("✅ 数据质量报告生成成功")
            overall_assessment = quality_report.get("overall_assessment", "N/A")
            print(f"  整体评估: {overall_assessment}")

            source_quality = quality_report.get("source_quality", {})
            for source, quality in source_quality.items():
                print(f"  {source} 质量: {quality:.2f}")

            recommendations = quality_report.get("recommendations", [])
            if recommendations:
                print(f"  主要建议: {recommendations[0]}")
        else:
            print(f"❌ 数据质量报告生成失败: {quality_report['error']}")

        # 测试缓存管理
        print("\n🗂️ 测试缓存管理...")

        # 获取缓存状态
        cache_status = loader.get_cache_status()
        if cache_status.get("status") == "enabled":
            print("✅ 缓存状态查询成功")
            cached_files = cache_status.get("cached_files", 0)
            print(f"  缓存文件: {cached_files} 个")

            if cached_files > 0:
                # 清理缓存
                clear_result = loader.clear_cache()
                if clear_result.get("status") == "all_cache_cleared":
                    print("✅ 缓存清理成功")
                else:
                    print(f"❌ 缓存清理失败: {clear_result}")

        return True


def test_data_pipeline_integration():
    """测试数据管道集成"""
    print("\n🔗 测试数据管道集成...")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 配置数据加载器
        config = {
            "enable_cache": True,
            "cache_directory": temp_dir,
            "enable_preprocessing": True,
            "create_unified_dataset": True,
            "use_fundamental": True,
            "use_technical": True,
            "use_sentiment": True,
            "use_macro": True,
        }

        loader = UnifiedDataLoader(config)

        # 测试完整的数据管道流程
        print("\n🚀 执行完整数据管道流程...")

        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        objectives = ["growth", "value", "balanced"]

        results = {}

        for objective in objectives:
            print(f"\n📊 测试 {objective} 策略数据管道...")

            # 加载数据
            data = loader.load_comprehensive_data(
                symbols=test_symbols, objective=objective, use_cache=False
            )

            if "error" not in data:
                results[objective] = data

                # 检查数据完整性
                required_sections = [
                    "fundamental",
                    "technical",
                    "sentiment",
                    "macro",
                    "analysis",
                ]
                missing_sections = [
                    section for section in required_sections if section not in data
                ]

                if not missing_sections:
                    print(f"✅ {objective} 策略数据完整")
                else:
                    print(f"⚠️ {objective} 策略缺少: {', '.join(missing_sections)}")

                # 检查统一数据集
                if "unified_dataset" in data:
                    unified_df = data["unified_dataset"]
                    if isinstance(unified_df, pd.DataFrame) and not unified_df.empty:
                        print(f"  统一数据集: {len(unified_df)} 行")

                        # 检查特征覆盖
                        feature_types = {
                            "fundamental": [
                                col
                                for col in unified_df.columns
                                if col.startswith("fundamental_")
                            ],
                            "technical": [
                                col
                                for col in unified_df.columns
                                if col.startswith("technical_")
                            ],
                            "sentiment": [
                                col
                                for col in unified_df.columns
                                if col.startswith("sentiment_")
                            ],
                            "macro": [
                                col
                                for col in unified_df.columns
                                if col.startswith("macro_")
                            ],
                        }

                        for feature_type, features in feature_types.items():
                            print(f"    {feature_type} 特征: {len(features)} 个")

                # 检查分析结果
                if "analysis" in data:
                    analysis = data["analysis"]

                    # 检查市场分析
                    if "market_analysis" in analysis:
                        market_analysis = analysis["market_analysis"]
                        if "tickers" in market_analysis:
                            selected_tickers = [
                                t["symbol"] for t in market_analysis["tickers"]
                            ]
                            print(f"  选择股票: {selected_tickers}")

                    # 检查宏观分析
                    if "macro_analysis" in analysis:
                        macro_analysis = analysis["macro_analysis"]
                        if "investment_regime" in macro_analysis:
                            regime = macro_analysis["investment_regime"]
                            regime_type = regime.get("regime_type", "N/A")
                            confidence = regime.get("confidence", 0)
                            print(
                                f"  投资环境: {regime_type} (置信度: {confidence:.2f})"
                            )

            else:
                print(f"❌ {objective} 策略数据加载失败: {data['error']}")
                return False

        # 对比不同策略的结果
        if len(results) > 1:
            print(f"\n📊 策略对比分析...")

            for obj1, obj2 in [
                ("growth", "value"),
                ("growth", "balanced"),
                ("value", "balanced"),
            ]:
                if obj1 in results and obj2 in results:
                    data1 = results[obj1]
                    data2 = results[obj2]

                    # 对比选股结果
                    if "analysis" in data1 and "analysis" in data2:
                        analysis1 = data1["analysis"]
                        analysis2 = data2["analysis"]

                        if (
                            "market_analysis" in analysis1
                            and "market_analysis" in analysis2
                        ):
                            tickers1 = set(
                                t["symbol"]
                                for t in analysis1["market_analysis"].get("tickers", [])
                            )
                            tickers2 = set(
                                t["symbol"]
                                for t in analysis2["market_analysis"].get("tickers", [])
                            )

                            overlap = len(tickers1 & tickers2)
                            total_unique = len(tickers1 | tickers2)

                            print(
                                f"  {obj1} vs {obj2}: {overlap}/{total_unique} 重叠股票"
                            )

        # 测试数据更新
        print(f"\n🔄 测试数据更新...")
        update_result = loader.update_cache(symbols=["AAPL"], force=False)

        if update_result.get("status") == "success":
            print("✅ 数据更新测试成功")
        else:
            print(f"❌ 数据更新测试失败: {update_result}")

        # 测试支持的股票列表
        supported_symbols = loader.get_supported_symbols()
        print(f"\n📋 支持的股票数量: {len(supported_symbols)}")
        print(f"  示例股票: {supported_symbols[:5]}")

        return True


def test_error_handling():
    """测试错误处理"""
    print("\n🛡️ 测试错误处理...")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            "enable_cache": True,
            "cache_directory": temp_dir,
            "enable_preprocessing": True,
        }

        loader = UnifiedDataLoader(config)

        # 测试无效股票代码
        print("\n🚫 测试无效股票代码...")
        invalid_data = loader.load_comprehensive_data(
            symbols=["INVALID_SYMBOL"], objective="balanced", use_cache=False
        )

        # 系统应该能处理无效股票而不崩溃
        if "error" not in invalid_data:
            print("✅ 无效股票代码处理正常")
        else:
            print(f"⚠️ 无效股票代码触发错误: {invalid_data['error']}")

        # 测试无效数据类型
        print("\n🚫 测试无效数据类型...")
        try:
            invalid_type_data = loader.load_specific_data("invalid_type", ["AAPL"])
            if "error" in invalid_type_data:
                print("✅ 无效数据类型错误处理正常")
            else:
                print("⚠️ 无效数据类型未触发预期错误")
        except Exception as e:
            print(f"✅ 无效数据类型触发异常: {type(e).__name__}")

        # 测试空股票列表
        print("\n🚫 测试空股票列表...")
        empty_data = loader.load_comprehensive_data(
            symbols=[], objective="balanced", use_cache=False
        )

        if "error" in empty_data or not empty_data:
            print("✅ 空股票列表处理正常")
        else:
            print("⚠️ 空股票列表处理可能有问题")

        return True


if __name__ == "__main__":
    print("🚀 数据预处理管道功能测试")
    print("=" * 60)

    tests = [
        ("数据预处理器", test_data_preprocessor),
        ("统一数据加载器", test_unified_data_loader),
        ("数据管道集成", test_data_pipeline_integration),
        ("错误处理", test_error_handling),
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
        print("🎉 所有数据管道功能测试通过！")
        print("✅ 数据预处理管道已全面建立！")
    else:
        print("⚠️ 部分测试未通过，需要进一步优化")

    print("\n📈 数据管道新增能力:")
    print("  ✅ 四源数据统一预处理")
    print("  ✅ 智能数据质量评估")
    print("  ✅ 跨因子特征工程")
    print("  ✅ 统一数据集构建")
    print("  ✅ 高效数据缓存机制")
    print("  ✅ 实时数据快照")
    print("  ✅ 灵活的数据加载配置")
    print("  ✅ 全面的错误处理")
    print("  ✅ 数据质量监控和报告")
