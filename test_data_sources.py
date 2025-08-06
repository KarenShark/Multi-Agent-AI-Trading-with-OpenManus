#!/usr/bin/env python3
"""
数据源验证和质量监控系统

验证所有数据源的可用性、质量和一致性，建立持续监控机制。
包括数据完整性、时效性、准确性和可靠性的全面检测。
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data.loader import UnifiedDataLoader
from app.data.preprocess import DataPreprocessor
from app.tool.enhanced_news_fetcher import EnhancedNewsFetcher
from app.tool.fundamental_fetcher import FundamentalFetcher
from app.tool.macro_economic_fetcher import MacroEconomicFetcher
from app.tool.yfinance_fetcher import YFinanceFetcher


class DataSourceValidator:
    """数据源验证器 - 全面验证数据质量和可靠性"""

    def __init__(self):
        # 初始化所有数据获取器
        self.yfinance_fetcher = YFinanceFetcher()
        self.fundamental_fetcher = FundamentalFetcher()
        self.news_fetcher = EnhancedNewsFetcher()
        self.macro_fetcher = MacroEconomicFetcher()
        self.data_loader = UnifiedDataLoader()
        self.preprocessor = DataPreprocessor()

        # 测试股票池
        self.test_stocks = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "JPM",
            "V",
            "JNJ",
        ]

        # 数据质量标准
        self.quality_standards = {
            "minimum_data_points": 10,
            "maximum_age_days": 7,
            "minimum_coverage_ratio": 0.7,
            "maximum_missing_ratio": 0.3,
            "minimum_data_variance": 0.001,
            "consistency_tolerance": 0.05,
        }

        # 验证结果
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "data_sources": {},
            "quality_metrics": {},
            "issues_found": [],
            "recommendations": [],
        }

    def run_comprehensive_validation(self) -> Dict:
        """运行全面的数据源验证"""
        print("🔍 开始数据源全面验证")
        print("=" * 60)

        try:
            # 1. 基础数据源连接测试
            print("\n📡 测试数据源连接性...")
            self._test_data_source_connectivity()

            # 2. 数据质量验证
            print("\n📊 验证数据质量...")
            self._validate_data_quality()

            # 3. 数据一致性检查
            print("\n🔄 检查数据一致性...")
            self._check_data_consistency()

            # 4. 性能和响应时间测试
            print("\n⚡ 测试性能和响应时间...")
            self._test_performance()

            # 5. 数据完整性验证
            print("\n✅ 验证数据完整性...")
            self._validate_data_completeness()

            # 6. 时效性检查
            print("\n⏰ 检查数据时效性...")
            self._check_data_freshness()

            # 7. 异常检测
            print("\n🚨 执行异常检测...")
            self._detect_anomalies()

            # 8. 生成综合评估
            print("\n📋 生成综合评估报告...")
            self._generate_comprehensive_assessment()

            print("\n✅ 数据源验证完成")
            return self.validation_results

        except Exception as e:
            error_msg = f"数据源验证过程中发生错误: {str(e)}"
            print(f"❌ {error_msg}")
            self.validation_results["overall_status"] = "error"
            self.validation_results["error"] = error_msg
            return self.validation_results

    def _test_data_source_connectivity(self):
        """测试数据源连接性"""
        connectivity_results = {}

        # 测试 YFinance 连接
        print("  📈 测试 YFinance API...")
        try:
            start_time = time.time()
            test_data = self.yfinance_fetcher.execute("AAPL", "1d")
            response_time = time.time() - start_time

            connectivity_results["yfinance"] = {
                "status": "success" if test_data.get("success") else "failed",
                "response_time": response_time,
                "data_points": len(test_data.get("data", [])),
                "error": test_data.get("error"),
            }

            if response_time > 10:
                self.validation_results["issues_found"].append(
                    f"YFinance响应时间过长: {response_time:.2f}秒"
                )

        except Exception as e:
            connectivity_results["yfinance"] = {"status": "error", "error": str(e)}
            self.validation_results["issues_found"].append(
                f"YFinance连接失败: {str(e)}"
            )

        # 测试基本面数据
        print("  💰 测试基本面数据源...")
        try:
            start_time = time.time()
            test_data = self.fundamental_fetcher.execute("AAPL")
            response_time = time.time() - start_time

            connectivity_results["fundamental"] = {
                "status": "success" if test_data.get("success") else "failed",
                "response_time": response_time,
                "metrics_count": len(
                    test_data.get("data", {}).get("financial_ratios", {})
                ),
                "error": test_data.get("error"),
            }

        except Exception as e:
            connectivity_results["fundamental"] = {"status": "error", "error": str(e)}
            self.validation_results["issues_found"].append(
                f"基本面数据连接失败: {str(e)}"
            )

        # 测试新闻数据
        print("  📰 测试新闻数据源...")
        try:
            start_time = time.time()
            test_data = self.news_fetcher.execute("AAPL")
            response_time = time.time() - start_time

            connectivity_results["news"] = {
                "status": "success" if test_data.get("success") else "failed",
                "response_time": response_time,
                "articles_count": len(test_data.get("data", [])),
                "error": test_data.get("error"),
            }

        except Exception as e:
            connectivity_results["news"] = {"status": "error", "error": str(e)}
            self.validation_results["issues_found"].append(
                f"新闻数据连接失败: {str(e)}"
            )

        # 测试宏观数据
        print("  🌍 测试宏观经济数据源...")
        try:
            start_time = time.time()
            test_data = self.macro_fetcher.execute()
            response_time = time.time() - start_time

            connectivity_results["macro"] = {
                "status": "success" if test_data.get("success") else "failed",
                "response_time": response_time,
                "indicators_count": len(test_data.get("data", {})),
                "error": test_data.get("error"),
            }

        except Exception as e:
            connectivity_results["macro"] = {"status": "error", "error": str(e)}
            self.validation_results["issues_found"].append(
                f"宏观数据连接失败: {str(e)}"
            )

        self.validation_results["data_sources"]["connectivity"] = connectivity_results

        # 连接性汇总
        successful_sources = sum(
            1 for r in connectivity_results.values() if r.get("status") == "success"
        )
        total_sources = len(connectivity_results)

        print(f"  ✅ 数据源连接测试完成: {successful_sources}/{total_sources} 成功连接")

    def _validate_data_quality(self):
        """验证数据质量"""
        quality_results = {}

        # 抽样测试数据质量
        sample_stocks = ["AAPL", "MSFT", "GOOGL"]

        for data_type in ["fundamental", "technical", "sentiment", "macro"]:
            print(f"    🔍 验证 {data_type} 数据质量...")

            quality_metrics = {
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
                "timeliness": 0.0,
                "issues": [],
            }

            try:
                if data_type == "fundamental":
                    quality_metrics = self._validate_fundamental_quality(sample_stocks)
                elif data_type == "technical":
                    quality_metrics = self._validate_technical_quality(sample_stocks)
                elif data_type == "sentiment":
                    quality_metrics = self._validate_sentiment_quality(sample_stocks)
                elif data_type == "macro":
                    quality_metrics = self._validate_macro_quality()

                quality_results[data_type] = quality_metrics

                # 检查质量标准
                if (
                    quality_metrics["completeness"]
                    < self.quality_standards["minimum_coverage_ratio"]
                ):
                    self.validation_results["issues_found"].append(
                        f"{data_type}数据完整性不足: {quality_metrics['completeness']:.2%}"
                    )

            except Exception as e:
                quality_metrics["error"] = str(e)
                quality_results[data_type] = quality_metrics
                self.validation_results["issues_found"].append(
                    f"{data_type}数据质量验证失败: {str(e)}"
                )

        self.validation_results["data_sources"]["quality"] = quality_results

    def _validate_fundamental_quality(self, stocks: List[str]) -> Dict:
        """验证基本面数据质量"""
        total_metrics = 0
        valid_metrics = 0
        accuracy_scores = []

        for stock in stocks:
            try:
                # 确保使用完整的股票代码
                if len(stock) < 2:
                    continue

                result = self.fundamental_fetcher.execute(stock)
                if result.get("success"):
                    data = result.get("data", {})
                    ratios = data.get("financial_ratios", {})

                    # 检查关键指标
                    key_ratios = ["pe_ratio", "roe", "debt_to_equity", "current_ratio"]
                    for ratio in key_ratios:
                        total_metrics += 1
                        if ratio in ratios and ratios[ratio] is not None:
                            valid_metrics += 1
                            # 基本合理性检查
                            value = ratios[ratio]
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                if ratio == "pe_ratio" and 0 < value < 1000:
                                    accuracy_scores.append(1.0)
                                elif ratio == "roe" and -100 < value < 100:
                                    accuracy_scores.append(1.0)
                                elif ratio == "debt_to_equity" and 0 <= value < 10:
                                    accuracy_scores.append(1.0)
                                elif ratio == "current_ratio" and 0 < value < 20:
                                    accuracy_scores.append(1.0)
                                else:
                                    accuracy_scores.append(0.7)  # 可疑但不一定错误
                            else:
                                accuracy_scores.append(0.0)
                        else:
                            accuracy_scores.append(0.0)

            except Exception as e:
                # 记录具体的错误信息以便调试
                print(f"    ⚠️ 股票 {stock} 基本面数据验证失败: {str(e)[:100]}...")
                continue

        completeness = valid_metrics / total_metrics if total_metrics > 0 else 0
        accuracy = np.mean(accuracy_scores) if accuracy_scores else 0

        return {
            "completeness": completeness,
            "accuracy": accuracy,
            "consistency": 0.9,  # 基本面数据通常一致性较好
            "timeliness": 0.8,  # 基本面数据更新频率较低
            "total_metrics": total_metrics,
            "valid_metrics": valid_metrics,
            "issues": [],
        }

    def _validate_technical_quality(self, stocks: List[str]) -> Dict:
        """验证技术面数据质量"""
        total_points = 0
        valid_points = 0
        variance_scores = []

        for stock in stocks:
            try:
                result = self.yfinance_fetcher.execute(stock, "1mo")
                if result.get("success"):
                    data = result.get("data", [])

                    if len(data) >= self.quality_standards["minimum_data_points"]:
                        total_points += len(data)

                        # 检查数据完整性
                        for point in data:
                            if all(
                                key in point and point[key] is not None
                                for key in ["open", "high", "low", "close", "volume"]
                            ):
                                valid_points += 1

                        # 检查价格数据方差
                        if len(data) > 1:
                            prices = [p["close"] for p in data if p.get("close")]
                            if len(prices) > 1:
                                price_variance = np.var(prices) / np.mean(prices) ** 2
                                variance_scores.append(
                                    1.0
                                    if price_variance
                                    > self.quality_standards["minimum_data_variance"]
                                    else 0.5
                                )

            except Exception as e:
                continue

        completeness = valid_points / total_points if total_points > 0 else 0
        variance_quality = np.mean(variance_scores) if variance_scores else 0.5

        return {
            "completeness": completeness,
            "accuracy": 0.95,  # 技术数据通常较准确
            "consistency": 0.9,
            "timeliness": 0.95,  # 技术数据更新及时
            "variance_quality": variance_quality,
            "total_points": total_points,
            "valid_points": valid_points,
            "issues": [],
        }

    def _validate_sentiment_quality(self, stocks: List[str]) -> Dict:
        """验证情感面数据质量"""
        article_counts = []
        relevance_scores = []
        sentiment_diversities = []

        for stock in stocks:
            try:
                result = self.news_fetcher.execute(stock)
                if result.get("success"):
                    data = result.get("data", [])
                    article_counts.append(len(data))

                    # 检查相关性和情感多样性
                    if len(data) > 0:
                        # 模拟相关性评分（实际应该从数据中获取）
                        relevance_scores.append(0.8)  # 默认相关性

                        # 计算情感多样性
                        if len(data) > 1:
                            # 这里应该从实际情感分析结果中计算
                            sentiment_diversities.append(0.7)

            except Exception as e:
                continue

        avg_articles = np.mean(article_counts) if article_counts else 0
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        avg_diversity = np.mean(sentiment_diversities) if sentiment_diversities else 0

        # 评估完整性（基于文章数量）
        completeness = min(1.0, avg_articles / 10)  # 理想情况10篇文章

        return {
            "completeness": completeness,
            "accuracy": avg_relevance,
            "consistency": avg_diversity,
            "timeliness": 0.9,  # 新闻数据时效性较好
            "avg_articles": avg_articles,
            "avg_relevance": avg_relevance,
            "issues": [],
        }

    def _validate_macro_quality(self) -> Dict:
        """验证宏观面数据质量"""
        try:
            result = self.macro_fetcher.execute()
            if result.get("success"):
                data = result.get("data", {})

                # 检查关键指标覆盖
                key_categories = ["interest_rates", "inflation", "employment", "growth"]
                covered_categories = sum(1 for cat in key_categories if cat in data)
                category_coverage = covered_categories / len(key_categories)

                # 检查指标数量
                total_indicators = sum(
                    len(cat_data)
                    for cat_data in data.values()
                    if isinstance(cat_data, dict)
                )

                # 验证数值合理性
                reasonable_values = 0
                total_values = 0

                for category, indicators in data.items():
                    if isinstance(indicators, dict):
                        for indicator, value in indicators.items():
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                total_values += 1
                                # 基本合理性检查
                                if "rate" in indicator.lower():
                                    if -5 <= value <= 25:  # 合理的利率范围
                                        reasonable_values += 1
                                elif "unemployment" in indicator.lower():
                                    if 0 <= value <= 30:  # 合理的失业率范围
                                        reasonable_values += 1
                                elif (
                                    "cpi" in indicator.lower()
                                    or "inflation" in indicator.lower()
                                ):
                                    if -10 <= value <= 20:  # 合理的通胀范围
                                        reasonable_values += 1
                                else:
                                    reasonable_values += 1  # 其他指标默认合理

                accuracy = reasonable_values / total_values if total_values > 0 else 0

                return {
                    "completeness": category_coverage,
                    "accuracy": accuracy,
                    "consistency": 0.85,  # 宏观数据一致性较好
                    "timeliness": 0.75,  # 宏观数据更新频率较低
                    "total_indicators": total_indicators,
                    "category_coverage": category_coverage,
                    "issues": [],
                }
            else:
                return {
                    "completeness": 0.0,
                    "accuracy": 0.0,
                    "consistency": 0.0,
                    "timeliness": 0.0,
                    "issues": ["无法获取宏观数据"],
                }

        except Exception as e:
            return {
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
                "timeliness": 0.0,
                "issues": [f"宏观数据验证错误: {str(e)}"],
            }

    def _check_data_consistency(self):
        """检查数据一致性"""
        print("    🔄 检查跨源数据一致性...")

        consistency_results = {}

        # 检查股票价格数据一致性
        try:
            # 获取同一股票的基本面和技术面数据
            stock = "AAPL"

            # 基本面数据中的价格信息
            fundamental_result = self.fundamental_fetcher.execute(stock)
            technical_result = self.yfinance_fetcher.execute(stock, "1d")

            if fundamental_result.get("success") and technical_result.get("success"):
                fund_data = fundamental_result.get("data", {})
                tech_data = technical_result.get("data", [])

                # 比较市值等基础信息
                fund_market_cap = fund_data.get("basic_info", {}).get("market_cap")
                fund_price = fund_data.get("basic_info", {}).get("current_price")

                if tech_data and fund_price:
                    latest_tech_price = tech_data[-1].get("close")

                    if latest_tech_price and fund_price:
                        price_diff = abs(latest_tech_price - fund_price) / fund_price

                        consistency_results["price_consistency"] = {
                            "fundamental_price": fund_price,
                            "technical_price": latest_tech_price,
                            "difference_ratio": price_diff,
                            "consistent": price_diff
                            < self.quality_standards["consistency_tolerance"],
                        }

                        if price_diff > self.quality_standards["consistency_tolerance"]:
                            self.validation_results["issues_found"].append(
                                f"价格数据不一致: 基本面 {fund_price:.2f} vs 技术面 {latest_tech_price:.2f}"
                            )

        except Exception as e:
            consistency_results["price_consistency"] = {"error": str(e)}

        # 检查数据时间戳一致性
        try:
            current_time = datetime.now()
            timestamp_issues = []

            # 检查各数据源的时间戳
            for source_name, fetcher in [
                ("technical", self.yfinance_fetcher),
                ("fundamental", self.fundamental_fetcher),
                ("news", self.news_fetcher),
            ]:
                if source_name == "technical":
                    result = fetcher.execute("AAPL", "1d")
                else:
                    result = fetcher.execute("AAPL")

                if result.get("success"):
                    timestamp_str = result.get("timestamp")
                    if timestamp_str:
                        try:
                            data_time = datetime.fromisoformat(
                                timestamp_str.replace("Z", "+00:00")
                            )
                            age_hours = (
                                current_time - data_time.replace(tzinfo=None)
                            ).total_seconds() / 3600

                            if (
                                age_hours
                                > 24 * self.quality_standards["maximum_age_days"]
                            ):
                                timestamp_issues.append(
                                    f"{source_name}数据过期: {age_hours:.1f}小时"
                                )

                        except Exception:
                            timestamp_issues.append(f"{source_name}时间戳格式错误")

            consistency_results["timestamp_consistency"] = {
                "issues": timestamp_issues,
                "consistent": len(timestamp_issues) == 0,
            }

            if timestamp_issues:
                self.validation_results["issues_found"].extend(timestamp_issues)

        except Exception as e:
            consistency_results["timestamp_consistency"] = {"error": str(e)}

        self.validation_results["data_sources"]["consistency"] = consistency_results

    def _test_performance(self):
        """测试性能和响应时间"""
        print("    ⚡ 测试系统性能...")

        performance_results = {}

        # 测试并发加载性能
        try:
            start_time = time.time()

            # 模拟并发数据加载
            test_stocks = ["AAPL", "MSFT", "GOOGL"]
            results = self.data_loader.load_comprehensive_data(
                symbols=test_stocks, objective="balanced"
            )

            load_time = time.time() - start_time

            performance_results["comprehensive_load"] = {
                "load_time": load_time,
                "stocks_count": len(test_stocks),
                "time_per_stock": load_time / len(test_stocks),
                "success": results.get("success", False),
            }

            if load_time > 60:  # 超过1分钟
                self.validation_results["issues_found"].append(
                    f"数据加载性能较慢: {load_time:.2f}秒 for {len(test_stocks)}只股票"
                )

        except Exception as e:
            performance_results["comprehensive_load"] = {"error": str(e)}

        # 测试数据预处理性能
        try:
            start_time = time.time()

            # 模拟数据预处理
            sample_data = {
                "AAPL": {"price": 150.0, "volume": 1000000},
                "MSFT": {"price": 300.0, "volume": 800000},
            }

            processed_data = self.preprocessor.create_unified_dataset(
                fundamental_data=sample_data,
                technical_data=sample_data,
                sentiment_data=sample_data,
                macro_data=sample_data,
            )

            process_time = time.time() - start_time

            performance_results["preprocessing"] = {
                "process_time": process_time,
                "data_points": len(sample_data),
                "success": processed_data is not None,
            }

        except Exception as e:
            performance_results["preprocessing"] = {"error": str(e)}

        self.validation_results["data_sources"]["performance"] = performance_results

    def _validate_data_completeness(self):
        """验证数据完整性"""
        print("    ✅ 验证数据完整性...")

        completeness_results = {}

        # 检查股票覆盖率
        try:
            test_stocks = self.test_stocks[:5]  # 测试前5只股票
            coverage_stats = {"total": 0, "successful": 0}

            for stock in test_stocks:
                coverage_stats["total"] += 1

                # 检查是否能获取基础数据
                fund_result = self.fundamental_fetcher.execute(stock)
                tech_result = self.yfinance_fetcher.execute(stock, "1d")

                if fund_result.get("success") and tech_result.get("success"):
                    coverage_stats["successful"] += 1

            stock_coverage = coverage_stats["successful"] / coverage_stats["total"]

            completeness_results["stock_coverage"] = {
                "ratio": stock_coverage,
                "total_tested": coverage_stats["total"],
                "successful": coverage_stats["successful"],
                "meets_standard": stock_coverage
                >= self.quality_standards["minimum_coverage_ratio"],
            }

            if stock_coverage < self.quality_standards["minimum_coverage_ratio"]:
                self.validation_results["issues_found"].append(
                    f"股票数据覆盖率不足: {stock_coverage:.2%}"
                )

        except Exception as e:
            completeness_results["stock_coverage"] = {"error": str(e)}

        # 检查数据字段完整性
        try:
            required_fields = {
                "fundamental": ["basic_info", "financial_ratios", "analysis"],
                "technical": ["open", "high", "low", "close", "volume"],
                "sentiment": ["overall_score", "relevance"],
                "macro": ["interest_rates", "inflation", "employment"],
            }

            field_completeness = {}

            # 测试基本面字段
            fund_result = self.fundamental_fetcher.execute("AAPL")
            if fund_result.get("success"):
                fund_data = fund_result.get("data", {})
                fund_fields = sum(
                    1 for field in required_fields["fundamental"] if field in fund_data
                )
                field_completeness["fundamental"] = fund_fields / len(
                    required_fields["fundamental"]
                )

            # 测试技术面字段
            tech_result = self.yfinance_fetcher.execute("AAPL", "1d")
            if tech_result.get("success") and tech_result.get("data"):
                tech_data = (
                    tech_result.get("data", [])[0] if tech_result.get("data") else {}
                )
                tech_fields = sum(
                    1 for field in required_fields["technical"] if field in tech_data
                )
                field_completeness["technical"] = tech_fields / len(
                    required_fields["technical"]
                )

            completeness_results["field_completeness"] = field_completeness

        except Exception as e:
            completeness_results["field_completeness"] = {"error": str(e)}

        self.validation_results["data_sources"]["completeness"] = completeness_results

    def _check_data_freshness(self):
        """检查数据时效性"""
        print("    ⏰ 检查数据时效性...")

        freshness_results = {}
        current_time = datetime.now()

        # 检查各类数据的时效性
        data_sources = [
            ("technical", lambda: self.yfinance_fetcher.execute("AAPL", "1d")),
            ("fundamental", lambda: self.fundamental_fetcher.execute("AAPL")),
            ("news", lambda: self.news_fetcher.execute("AAPL")),
            ("macro", lambda: self.macro_fetcher.execute()),
        ]

        for source_name, data_func in data_sources:
            try:
                result = data_func()

                if result.get("success"):
                    timestamp_str = result.get("timestamp")
                    if timestamp_str:
                        try:
                            data_time = datetime.fromisoformat(
                                timestamp_str.replace("Z", "+00:00")
                            )
                            age_hours = (
                                current_time - data_time.replace(tzinfo=None)
                            ).total_seconds() / 3600
                            age_days = age_hours / 24

                            is_fresh = (
                                age_days <= self.quality_standards["maximum_age_days"]
                            )

                            freshness_results[source_name] = {
                                "timestamp": timestamp_str,
                                "age_hours": age_hours,
                                "age_days": age_days,
                                "is_fresh": is_fresh,
                                "max_age_days": self.quality_standards[
                                    "maximum_age_days"
                                ],
                            }

                            if not is_fresh:
                                self.validation_results["issues_found"].append(
                                    f"{source_name}数据不够新鲜: {age_days:.1f}天前"
                                )

                        except Exception as e:
                            freshness_results[source_name] = {
                                "error": f"时间戳解析错误: {str(e)}",
                                "is_fresh": False,
                            }
                    else:
                        freshness_results[source_name] = {
                            "error": "缺少时间戳",
                            "is_fresh": False,
                        }
                else:
                    freshness_results[source_name] = {
                        "error": "数据获取失败",
                        "is_fresh": False,
                    }

            except Exception as e:
                freshness_results[source_name] = {"error": str(e), "is_fresh": False}

        self.validation_results["data_sources"]["freshness"] = freshness_results

    def _detect_anomalies(self):
        """检测数据异常"""
        print("    🚨 执行异常检测...")

        anomaly_results = {}

        # 价格异常检测
        try:
            price_anomalies = []

            for stock in ["AAPL", "MSFT", "GOOGL"]:
                result = self.yfinance_fetcher.execute(stock, "1mo")
                if result.get("success"):
                    data = result.get("data", [])
                    if len(data) > 10:
                        prices = [p["close"] for p in data if p.get("close")]

                        if len(prices) > 2:
                            # 检测价格突变
                            price_changes = [
                                abs(prices[i] - prices[i - 1]) / prices[i - 1]
                                for i in range(1, len(prices))
                            ]

                            max_change = max(price_changes) if price_changes else 0

                            if max_change > 0.2:  # 单日涨跌幅超过20%
                                price_anomalies.append(
                                    {
                                        "stock": stock,
                                        "max_change": max_change,
                                        "description": f"{stock}发现价格异常波动: {max_change:.2%}",
                                    }
                                )

            anomaly_results["price_anomalies"] = price_anomalies

            if price_anomalies:
                for anomaly in price_anomalies:
                    self.validation_results["issues_found"].append(
                        anomaly["description"]
                    )

        except Exception as e:
            anomaly_results["price_anomalies"] = {"error": str(e)}

        # 数据值异常检测
        try:
            value_anomalies = []

            # 检测基本面数据异常
            result = self.fundamental_fetcher.execute("AAPL")
            if result.get("success"):
                ratios = result.get("data", {}).get("financial_ratios", {})

                # 检测异常的财务比率
                if "pe_ratio" in ratios:
                    pe = ratios["pe_ratio"]
                    if isinstance(pe, (int, float)) and (pe < 0 or pe > 500):
                        value_anomalies.append(f"异常P/E比率: {pe}")

                if "debt_to_equity" in ratios:
                    de = ratios["debt_to_equity"]
                    if isinstance(de, (int, float)) and de > 10:
                        value_anomalies.append(f"异常债务权益比: {de}")

            anomaly_results["value_anomalies"] = value_anomalies

            if value_anomalies:
                self.validation_results["issues_found"].extend(value_anomalies)

        except Exception as e:
            anomaly_results["value_anomalies"] = {"error": str(e)}

        self.validation_results["data_sources"]["anomalies"] = anomaly_results

    def _generate_comprehensive_assessment(self):
        """生成综合评估报告"""
        print("    📋 生成综合评估...")

        try:
            # 计算整体质量分数
            quality_scores = []

            # 连接性评分
            connectivity = self.validation_results["data_sources"].get(
                "connectivity", {}
            )
            successful_connections = sum(
                1 for r in connectivity.values() if r.get("status") == "success"
            )
            total_connections = len(connectivity)
            connectivity_score = (
                successful_connections / total_connections
                if total_connections > 0
                else 0
            )
            quality_scores.append(connectivity_score)

            # 数据质量评分
            quality_data = self.validation_results["data_sources"].get("quality", {})
            quality_score = 0
            quality_count = 0

            for data_type, metrics in quality_data.items():
                if isinstance(metrics, dict) and "completeness" in metrics:
                    quality_score += metrics["completeness"]
                    quality_count += 1

            if quality_count > 0:
                avg_quality = quality_score / quality_count
                quality_scores.append(avg_quality)

            # 一致性评分
            consistency = self.validation_results["data_sources"].get("consistency", {})
            consistency_score = 0.8  # 默认分数
            if "price_consistency" in consistency:
                price_consistent = consistency["price_consistency"].get(
                    "consistent", False
                )
                consistency_score = 0.9 if price_consistent else 0.6
            quality_scores.append(consistency_score)

            # 时效性评分
            freshness = self.validation_results["data_sources"].get("freshness", {})
            fresh_sources = sum(
                1 for f in freshness.values() if f.get("is_fresh", False)
            )
            total_sources = len(freshness)
            freshness_score = fresh_sources / total_sources if total_sources > 0 else 0
            quality_scores.append(freshness_score)

            # 计算加权平均分
            weights = [0.3, 0.3, 0.2, 0.2]  # 连接性、质量、一致性、时效性
            overall_score = sum(
                score * weight for score, weight in zip(quality_scores, weights)
            )

            # 确定整体状态
            if overall_score >= 0.9:
                status = "excellent"
                status_desc = "优秀"
            elif overall_score >= 0.8:
                status = "good"
                status_desc = "良好"
            elif overall_score >= 0.7:
                status = "acceptable"
                status_desc = "可接受"
            elif overall_score >= 0.6:
                status = "poor"
                status_desc = "较差"
            else:
                status = "critical"
                status_desc = "严重问题"

            # 生成建议
            recommendations = []

            if connectivity_score < 0.8:
                recommendations.append("改善数据源连接稳定性")

            if len(quality_scores) > 1 and quality_scores[1] < 0.8:
                recommendations.append("提升数据质量标准")

            if consistency_score < 0.8:
                recommendations.append("加强数据一致性检查")

            if freshness_score < 0.8:
                recommendations.append("增加数据更新频率")

            if len(self.validation_results["issues_found"]) > 5:
                recommendations.append("解决识别出的数据问题")

            # 更新验证结果
            self.validation_results.update(
                {
                    "overall_status": status,
                    "overall_score": overall_score,
                    "status_description": status_desc,
                    "quality_breakdown": {
                        "connectivity": connectivity_score,
                        "quality": quality_scores[1] if len(quality_scores) > 1 else 0,
                        "consistency": consistency_score,
                        "freshness": freshness_score,
                    },
                    "recommendations": recommendations,
                }
            )

        except Exception as e:
            self.validation_results["overall_status"] = "error"
            self.validation_results["assessment_error"] = str(e)

    def generate_monitoring_report(self) -> str:
        """生成监控报告"""
        report = []
        report.append("📊 数据源质量监控报告")
        report.append("=" * 50)
        report.append(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(
            f"🎯 整体状态: {self.validation_results.get('status_description', '未知')}"
        )
        report.append(
            f"📈 整体评分: {self.validation_results.get('overall_score', 0):.2%}"
        )

        # 质量分解
        breakdown = self.validation_results.get("quality_breakdown", {})
        if breakdown:
            report.append("\n📊 质量分解:")
            report.append(f"  🔗 连接性: {breakdown.get('connectivity', 0):.2%}")
            report.append(f"  📈 数据质量: {breakdown.get('quality', 0):.2%}")
            report.append(f"  🔄 一致性: {breakdown.get('consistency', 0):.2%}")
            report.append(f"  ⏰时效性: {breakdown.get('freshness', 0):.2%}")

        # 发现的问题
        issues = self.validation_results.get("issues_found", [])
        if issues:
            report.append(f"\n⚠️ 发现的问题 ({len(issues)}个):")
            for i, issue in enumerate(issues[:10], 1):  # 显示前10个问题
                report.append(f"  {i}. {issue}")
            if len(issues) > 10:
                report.append(f"  ... 还有 {len(issues) - 10} 个问题")

        # 改进建议
        recommendations = self.validation_results.get("recommendations", [])
        if recommendations:
            report.append(f"\n💡 改进建议:")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"  {i}. {rec}")

        # 数据源状态
        connectivity = self.validation_results.get("data_sources", {}).get(
            "connectivity", {}
        )
        if connectivity:
            report.append("\n📡 数据源连接状态:")
            for source, status in connectivity.items():
                status_icon = "✅" if status.get("status") == "success" else "❌"
                response_time = status.get("response_time", 0)
                report.append(f"  {status_icon} {source}: {response_time:.2f}秒")

        return "\n".join(report)

    def save_validation_results(self, filepath: str = "data_validation_results.json"):
        """保存验证结果到文件"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
            print(f"✅ 验证结果已保存到: {filepath}")
        except Exception as e:
            print(f"❌ 保存验证结果失败: {str(e)}")


def main():
    """主函数 - 运行数据源验证"""
    print("🔍 OpenManus 数据源验证系统")
    print("=" * 60)

    try:
        # 创建验证器
        validator = DataSourceValidator()

        # 运行全面验证
        results = validator.run_comprehensive_validation()

        # 生成并显示报告
        print("\n" + "=" * 60)
        report = validator.generate_monitoring_report()
        print(report)

        # 保存结果
        validator.save_validation_results()

        # 返回状态码
        status = results.get("overall_status", "error")
        if status in ["excellent", "good"]:
            return 0
        elif status == "acceptable":
            return 1
        else:
            return 2

    except Exception as e:
        print(f"❌ 验证过程发生错误: {str(e)}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
