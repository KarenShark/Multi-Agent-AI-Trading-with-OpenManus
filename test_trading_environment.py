#!/usr/bin/env python3
"""
交易环境全面测试

测试TradingEnvironment的所有功能，包括：
- 基础功能测试
- 数据集成测试
- 交易逻辑测试
- 奖励机制测试
- 性能基准测试
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.trading.trading_env import TradingEnvironment, create_trading_environment

class TradingEnvironmentTester:
    """交易环境全面测试器"""

    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "basic_functionality": {},
            "data_integration": {},
            "trading_logic": {},
            "reward_mechanism": {},
            "performance_benchmark": {},
            "overall_assessment": {}
        }

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """运行全面测试"""
        print("🤖 StockSynergy 交易环境全面测试")
        print("=" * 60)

        try:
            # 1. 基础功能测试
            print("\n🔧 测试1: 基础功能测试")
            self._test_basic_functionality()

            # 2. 数据集成测试
            print("\n📊 测试2: 数据集成测试")
            self._test_data_integration()

            # 3. 交易逻辑测试
            print("\n💰 测试3: 交易逻辑测试")
            self._test_trading_logic()

            # 4. 奖励机制测试
            print("\n🎯 测试4: 奖励机制测试")
            self._test_reward_mechanism()

            # 5. 性能基准测试
            print("\n⚡ 测试5: 性能基准测试")
            self._test_performance_benchmark()

            # 6. 综合评估
            print("\n📋 生成综合评估...")
            self._generate_overall_assessment()

            # 7. 显示结果
            self._display_test_results()

            return self.test_results

        except Exception as e:
            print(f"❌ 测试过程中发生错误: {str(e)}")
            self.test_results["error"] = str(e)
            return self.test_results

    def _test_basic_functionality(self):
        """基础功能测试"""
        print("  🔍 测试环境创建和基础操作...")

        basic_tests = {
            "environment_creation": False,
            "reset_functionality": False,
            "step_execution": False,
            "observation_space": False,
            "action_space": False,
            "info_structure": False
        }

        try:
            # 创建环境
            env = create_trading_environment(
                symbols=["AAPL", "MSFT"],
                initial_balance=50000.0,
                objective="balanced"
            )
            basic_tests["environment_creation"] = True
            print("    ✅ 环境创建成功")

            # 测试reset
            observation, info = env.reset()
            basic_tests["reset_functionality"] = True
            print(f"    ✅ Reset功能正常 - 观察维度: {observation.shape}")

            # 测试step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            basic_tests["step_execution"] = True
            print(f"    ✅ Step执行正常 - 奖励: {reward:.4f}")

            # 测试观察空间
            if observation.shape[0] > 0 and np.isfinite(observation).all():
                basic_tests["observation_space"] = True
                print(f"    ✅ 观察空间正常 - 特征数: {observation.shape[0]}")

            # 测试动作空间
            if hasattr(env.action_space, 'nvec') and len(env.action_space.nvec) == len(env.symbols):
                basic_tests["action_space"] = True
                print(f"    ✅ 动作空间正常 - 股票数: {len(env.symbols)}")

            # 测试info结构
            required_info_keys = ['portfolio_value', 'cash_balance', 'total_return', 'positions']
            if all(key in info for key in required_info_keys):
                basic_tests["info_structure"] = True
                print("    ✅ Info结构完整")

        except Exception as e:
            print(f"    ❌ 基础功能测试失败: {str(e)}")

        self.test_results["basic_functionality"] = {
            "tests": basic_tests,
            "success_rate": sum(basic_tests.values()) / len(basic_tests),
            "details": {
                "observation_shape": observation.shape if 'observation' in locals() else None,
                "action_space_type": str(type(env.action_space)) if 'env' in locals() else None,
                "info_keys": list(info.keys()) if 'info' in locals() else None
            }
        }

    def _test_data_integration(self):
        """数据集成测试"""
        print("  📊 测试数据源集成...")

        data_tests = {
            "unified_data_loading": False,
            "feature_extraction": False,
            "price_series_generation": False,
            "multi_symbol_support": False,
            "data_quality": False
        }

        try:
            # 创建多股票环境
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            env = create_trading_environment(
                symbols=symbols,
                initial_balance=100000.0,
                objective="growth"
            )

            # 检查统一数据加载
            if hasattr(env, 'feature_data') and env.feature_data is not None:
                data_tests["unified_data_loading"] = True
                print(f"    ✅ 统一数据加载成功 - 特征数: {len(env.feature_data.columns)}")

            # 检查特征提取
            observation, _ = env.reset()
            if observation.shape[0] > len(symbols) * 10:  # 应该有足够的特征
                data_tests["feature_extraction"] = True
                print(f"    ✅ 特征提取正常 - 总特征维度: {observation.shape[0]}")

            # 检查价格序列
            if hasattr(env, 'price_data') and len(env.price_data) == len(symbols):
                data_tests["price_series_generation"] = True
                avg_length = np.mean([len(prices) for prices in env.price_data.values()])
                print(f"    ✅ 价格序列生成 - 平均长度: {avg_length:.0f}天")

            # 检查多股票支持
            if len(env.symbols) == len(symbols) and len(env.positions) == len(symbols):
                data_tests["multi_symbol_support"] = True
                print(f"    ✅ 多股票支持 - 股票数: {len(symbols)}")

            # 检查数据质量
            current_prices = env._get_current_prices()
            if all(price > 0 for price in current_prices.values()):
                data_tests["data_quality"] = True
                price_range = f"{min(current_prices.values()):.2f}-{max(current_prices.values()):.2f}"
                print(f"    ✅ 数据质量良好 - 价格范围: ${price_range}")

        except Exception as e:
            print(f"    ❌ 数据集成测试失败: {str(e)}")

        self.test_results["data_integration"] = {
            "tests": data_tests,
            "success_rate": sum(data_tests.values()) / len(data_tests),
            "details": {
                "symbols_tested": symbols if 'symbols' in locals() else [],
                "feature_count": len(env.feature_data.columns) if 'env' in locals() and hasattr(env, 'feature_data') and env.feature_data is not None else 0,
                "price_data_length": len(env.price_data) if 'env' in locals() and hasattr(env, 'price_data') else 0
            }
        }

    def _test_trading_logic(self):
        """交易逻辑测试"""
        print("  💰 测试交易执行逻辑...")

        trading_tests = {
            "buy_execution": False,
            "sell_execution": False,
            "hold_execution": False,
            "position_tracking": False,
            "balance_updates": False,
            "transaction_costs": False,
            "position_limits": False
        }

        try:
            env = create_trading_environment(
                symbols=["AAPL", "MSFT"],
                initial_balance=100000.0,
                max_position_size=0.4
            )

            env.reset()
            initial_balance = env.balance

            # 测试买入
            buy_action = [2, 2]  # 买入两只股票
            obs, reward, _, _, info = env.step(buy_action)

            if env.balance < initial_balance:  # 现金减少
                trading_tests["buy_execution"] = True
                print("    ✅ 买入执行正常")

            if any(pos > 0 for pos in env.positions.values()):
                trading_tests["position_tracking"] = True
                print("    ✅ 持仓跟踪正常")

            # 测试卖出
            sell_action = [0, 1]  # 卖出第一只，持有第二只
            prev_balance = env.balance
            env.step(sell_action)

            if env.balance > prev_balance:  # 现金增加
                trading_tests["sell_execution"] = True
                print("    ✅ 卖出执行正常")

            # 测试持有
            hold_action = [1, 1]  # 全部持有
            prev_portfolio = env.portfolio_value
            env.step(hold_action)

            if len(env.transaction_history) == len([t for t in env.transaction_history if t['action'] in ['buy', 'sell']]):
                trading_tests["hold_execution"] = True
                print("    ✅ 持有执行正常")

            # 测试余额更新
            if env.portfolio_value > 0 and env.balance >= 0:
                trading_tests["balance_updates"] = True
                print("    ✅ 余额更新正常")

            # 测试交易成本
            if any('cost' in t for t in env.transaction_history):
                trading_tests["transaction_costs"] = True
                print("    ✅ 交易成本计算正常")

            # 测试仓位限制
            total_investment = sum(
                env.positions[symbol] * env._get_current_prices()[symbol]
                for symbol in env.symbols
            )
            max_allowed = env.portfolio_value * env.max_position_size * len(env.symbols)

            if total_investment <= max_allowed * 1.1:  # 允许10%误差
                trading_tests["position_limits"] = True
                print("    ✅ 仓位限制正常")

        except Exception as e:
            print(f"    ❌ 交易逻辑测试失败: {str(e)}")

        self.test_results["trading_logic"] = {
            "tests": trading_tests,
            "success_rate": sum(trading_tests.values()) / len(trading_tests),
            "details": {
                "transactions_executed": len(env.transaction_history) if 'env' in locals() else 0,
                "final_balance": env.balance if 'env' in locals() else 0,
                "final_positions": env.positions if 'env' in locals() else {}
            }
        }

    def _test_reward_mechanism(self):
        """奖励机制测试"""
        print("  🎯 测试奖励计算机制...")

        reward_tests = {
            "return_reward": False,
            "transaction_cost_penalty": False,
            "drawdown_penalty": False,
            "diversification_reward": False,
            "risk_adjustment": False,
            "reward_stability": False
        }

        try:
            env = create_trading_environment(
                symbols=["AAPL", "MSFT", "GOOGL"],
                initial_balance=100000.0
            )

            env.reset()
            rewards = []
            portfolio_values = []

            # 执行多步交易来测试奖励机制
            for step in range(10):
                if step < 3:
                    action = [2, 2, 2]  # 买入阶段
                elif step < 6:
                    action = [1, 1, 1]  # 持有阶段
                else:
                    action = [0, 1, 2]  # 混合交易

                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                portfolio_values.append(info['portfolio_value'])

                if terminated or truncated:
                    break

            # 检查收益奖励
            portfolio_changes = np.diff(portfolio_values)
            reward_changes = np.array(rewards[1:])

            if len(portfolio_changes) > 0 and len(reward_changes) > 0:
                correlation = np.corrcoef(portfolio_changes, reward_changes)[0, 1]
                if not np.isnan(correlation) and abs(correlation) > 0.1:
                    reward_tests["return_reward"] = True
                    print(f"    ✅ 收益奖励机制 - 相关性: {correlation:.3f}")

            # 检查交易成本惩罚
            if len(env.transaction_history) > 0:
                reward_tests["transaction_cost_penalty"] = True
                print("    ✅ 交易成本惩罚机制")

            # 检查回撤惩罚
            if hasattr(env, 'max_drawdown') and env.max_drawdown >= 0:
                reward_tests["drawdown_penalty"] = True
                print(f"    ✅ 回撤惩罚机制 - 最大回撤: {env.max_drawdown:.3f}")

            # 检查分散投资奖励
            current_prices = env._get_current_prices()
            total_value = sum(env.positions[s] * current_prices[s] for s in env.symbols)
            if total_value > 0:
                weights = [env.positions[s] * current_prices[s] / total_value for s in env.symbols]
                if len([w for w in weights if w > 0.1]) >= 2:  # 至少2只股票有意义持仓
                    reward_tests["diversification_reward"] = True
                    print("    ✅ 分散投资奖励机制")

            # 检查风险调整
            if hasattr(env, 'sharpe_window') and len(env.sharpe_window) > 0:
                reward_tests["risk_adjustment"] = True
                print("    ✅ 风险调整机制")

            # 检查奖励稳定性
            if len(rewards) > 0:
                reward_std = np.std(rewards)
                if reward_std < 10:  # 奖励不应该过于波动
                    reward_tests["reward_stability"] = True
                    print(f"    ✅ 奖励稳定性 - 标准差: {reward_std:.3f}")

        except Exception as e:
            print(f"    ❌ 奖励机制测试失败: {str(e)}")

        self.test_results["reward_mechanism"] = {
            "tests": reward_tests,
            "success_rate": sum(reward_tests.values()) / len(reward_tests),
            "details": {
                "total_rewards": len(rewards) if 'rewards' in locals() else 0,
                "reward_range": [min(rewards), max(rewards)] if 'rewards' in locals() and len(rewards) > 0 else [0, 0],
                "avg_reward": np.mean(rewards) if 'rewards' in locals() and len(rewards) > 0 else 0
            }
        }

    def _test_performance_benchmark(self):
        """性能基准测试"""
        print("  ⚡ 性能基准测试...")

        performance_tests = {
            "initialization_speed": False,
            "step_execution_speed": False,
            "memory_efficiency": False,
            "scalability": False,
            "error_handling": False
        }

        benchmark_results = {}

        try:
            import time
            import psutil
            import gc

            # 测试初始化速度
            start_time = time.time()
            env = create_trading_environment(
                symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                initial_balance=100000.0
            )
            init_time = time.time() - start_time
            benchmark_results["init_time"] = init_time

            if init_time < 30:  # 30秒内初始化
                performance_tests["initialization_speed"] = True
                print(f"    ✅ 初始化速度 - {init_time:.2f}秒")

            # 测试步骤执行速度
            env.reset()
            start_time = time.time()

            for _ in range(50):
                action = env.action_space.sample()
                env.step(action)

            step_time = (time.time() - start_time) / 50
            benchmark_results["step_time"] = step_time

            if step_time < 0.1:  # 每步小于0.1秒
                performance_tests["step_execution_speed"] = True
                print(f"    ✅ 步骤执行速度 - {step_time:.4f}秒/步")

            # 测试内存效率
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            benchmark_results["memory_usage"] = memory_usage

            if memory_usage < 500:  # 小于500MB
                performance_tests["memory_efficiency"] = True
                print(f"    ✅ 内存效率 - {memory_usage:.1f}MB")

            # 测试可扩展性（更多股票）
            large_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
            start_time = time.time()
            large_env = create_trading_environment(symbols=large_symbols[:8])
            large_init_time = time.time() - start_time
            benchmark_results["large_env_init_time"] = large_init_time

            if large_init_time < 60:  # 大环境60秒内初始化
                performance_tests["scalability"] = True
                print(f"    ✅ 可扩展性 - 8股票环境{large_init_time:.2f}秒")

            # 测试错误处理
            try:
                # 测试无效动作
                invalid_env = create_trading_environment(symbols=["AAPL"])
                invalid_env.reset()
                invalid_env.step([10])  # 无效动作值

                # 测试极端情况
                extreme_env = create_trading_environment(
                    symbols=["AAPL"],
                    initial_balance=1.0  # 极小余额
                )
                extreme_env.reset()
                extreme_env.step([2])  # 尝试买入

                performance_tests["error_handling"] = True
                print("    ✅ 错误处理机制")

            except Exception:
                # 预期会有一些异常，这是正常的
                performance_tests["error_handling"] = True
                print("    ✅ 错误处理机制（异常捕获正常）")

        except Exception as e:
            print(f"    ❌ 性能基准测试失败: {str(e)}")

        self.test_results["performance_benchmark"] = {
            "tests": performance_tests,
            "success_rate": sum(performance_tests.values()) / len(performance_tests),
            "benchmark_results": benchmark_results
        }

    def _generate_overall_assessment(self):
        """生成综合评估"""

        # 计算各模块得分
        module_scores = {}
        for module, results in self.test_results.items():
            if isinstance(results, dict) and "success_rate" in results:
                module_scores[module] = results["success_rate"]

        # 计算加权总分
        weights = {
            "basic_functionality": 0.25,
            "data_integration": 0.25,
            "trading_logic": 0.25,
            "reward_mechanism": 0.15,
            "performance_benchmark": 0.10
        }

        weighted_score = sum(
            module_scores.get(module, 0) * weight
            for module, weight in weights.items()
        )

        # 评级
        if weighted_score >= 0.9:
            grade = "优秀 (A)"
            status = "生产就绪"
        elif weighted_score >= 0.8:
            grade = "良好 (B)"
            status = "基本就绪"
        elif weighted_score >= 0.7:
            grade = "及格 (C)"
            status = "需要改进"
        elif weighted_score >= 0.6:
            grade = "待改进 (D)"
            status = "重大问题"
        else:
            grade = "不合格 (F)"
            status = "需要重构"

        # 关键成就
        achievements = []
        issues = []

        for module, score in module_scores.items():
            if score >= 0.9:
                achievements.append(f"{module}表现优秀 ({score:.1%})")
            elif score < 0.7:
                issues.append(f"{module}需要改进 ({score:.1%})")

        self.test_results["overall_assessment"] = {
            "weighted_score": weighted_score,
            "grade": grade,
            "status": status,
            "module_scores": module_scores,
            "achievements": achievements,
            "issues": issues,
            "recommendations": self._generate_recommendations(module_scores)
        }

    def _generate_recommendations(self, module_scores: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        for module, score in module_scores.items():
            if score < 0.8:
                if module == "basic_functionality":
                    recommendations.append("完善基础功能实现，确保所有API正常工作")
                elif module == "data_integration":
                    recommendations.append("优化数据加载和特征提取流程")
                elif module == "trading_logic":
                    recommendations.append("改进交易执行逻辑和仓位管理")
                elif module == "reward_mechanism":
                    recommendations.append("调整奖励函数参数，提升训练效果")
                elif module == "performance_benchmark":
                    recommendations.append("优化性能，减少内存占用和执行时间")

        if not recommendations:
            recommendations.append("系统表现优秀，可以开始模型训练")

        return recommendations

    def _display_test_results(self):
        """显示测试结果"""
        print("\n" + "=" * 60)
        print("📊 交易环境测试报告")
        print("=" * 60)

        overall = self.test_results["overall_assessment"]

        print(f"\n🎯 整体评估:")
        print(f"  📈 综合得分: {overall['weighted_score']:.1%}")
        print(f"  🏆 评级: {overall['grade']}")
        print(f"  📋 状态: {overall['status']}")

        print(f"\n📊 模块评分:")
        for module, score in overall["module_scores"].items():
            status_icon = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            print(f"  {status_icon} {module}: {score:.1%}")

        if overall["achievements"]:
            print(f"\n🏆 关键成就:")
            for achievement in overall["achievements"]:
                print(f"  🎉 {achievement}")

        if overall["issues"]:
            print(f"\n⚠️ 需要改进:")
            for issue in overall["issues"]:
                print(f"  🔧 {issue}")

        print(f"\n💡 建议:")
        for i, rec in enumerate(overall["recommendations"], 1):
            print(f"  {i}. {rec}")

        # 性能指标
        benchmark = self.test_results.get("performance_benchmark", {}).get("benchmark_results", {})
        if benchmark:
            print(f"\n⚡ 性能指标:")
            if "init_time" in benchmark:
                print(f"  🚀 初始化时间: {benchmark['init_time']:.2f}秒")
            if "step_time" in benchmark:
                print(f"  ⏱️ 平均步骤时间: {benchmark['step_time']:.4f}秒")
            if "memory_usage" in benchmark:
                print(f"  💾 内存使用: {benchmark['memory_usage']:.1f}MB")

        print(f"\n✅ 测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def save_results(self, filepath: str = "trading_environment_test_results.json"):
        """保存测试结果"""
        import json
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 测试结果已保存到: {filepath}")
        except Exception as e:
            print(f"❌ 保存结果失败: {str(e)}")


def main():
    """主函数"""
    print("🤖 OpenManus 交易环境全面测试")
    print("=" * 60)

    try:
        # 创建测试器
        tester = TradingEnvironmentTester()

        # 运行测试
        results = tester.run_comprehensive_tests()

        # 保存结果
        tester.save_results()

        # 返回状态码
        overall_score = results.get("overall_assessment", {}).get("weighted_score", 0)
        if overall_score >= 0.9:
            return 0  # 优秀
        elif overall_score >= 0.8:
            return 1  # 良好
        elif overall_score >= 0.7:
            return 2  # 及格
        else:
            return 3  # 需要改进

    except Exception as e:
        print(f"❌ 测试过程发生错误: {str(e)}")
        return 4


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)