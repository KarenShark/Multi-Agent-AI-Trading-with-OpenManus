"""
StockSynergy Trading Environment

基于OpenAI Gym的交易环境，整合四维数据源进行智能交易决策。
支持强化学习和监督学习算法的训练与验证。
"""

import sys
import os
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.data.loader import UnifiedDataLoader
from app.data.preprocess import DataPreprocessor

class TradingEnvironment(gym.Env):
    """
    基于Gym的交易环境

    State Space: 多维特征向量 (基本面 + 技术面 + 情感面 + 宏观面)
    Action Space: 离散动作 [0=卖出, 1=持有, 2=买入]
    Reward: 基于收益率、风险调整收益和交易成本的综合奖励
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: str = "2024-01-01",
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        objective: str = "balanced",
        lookback_window: int = 30,
        max_position_size: float = 0.3,
        risk_free_rate: float = 0.04
    ):
        super().__init__()

        # 环境基础配置
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.objective = objective
        self.lookback_window = lookback_window
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate

        # 数据组件
        self.data_loader = UnifiedDataLoader()
        self.preprocessor = DataPreprocessor()

        # 环境状态
        self.current_step = 0
        self.balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in symbols}  # 持股数量
        self.portfolio_value = initial_balance
        self.transaction_history = []
        self.reward_history = []
        self.state_history = []

        # 数据存储
        self.market_data = None
        self.feature_data = None
        self.price_data = None

        # 奖励计算组件
        self.previous_portfolio_value = initial_balance
        self.sharpe_window = []
        self.max_drawdown = 0.0
        self.peak_value = initial_balance

        # 初始化环境
        self._load_market_data()
        self._setup_spaces()

        print(f"🤖 交易环境初始化完成:")
        print(f"  📊 股票池: {len(self.symbols)}只股票")
        print(f"  📅 交易周期: {self.start_date} 至 {self.end_date}")
        print(f"  💰 初始资金: ${self.initial_balance:,.2f}")
        print(f"  🎯 投资目标: {self.objective}")
        print(f"  📈 特征维度: {self.observation_space.shape[0]}")

    def _load_market_data(self):
        """加载市场数据"""
        print("📊 加载交易环境数据...")

        try:
            # 使用统一数据加载器获取综合数据
            data_result = self.data_loader.load_comprehensive_data(
                symbols=self.symbols,
                objective=self.objective
            )

            if not data_result.get("success"):
                raise Exception(f"数据加载失败: {data_result.get('error', '未知错误')}")

            # 提取统一数据集
            self.feature_data = data_result.get("unified_dataset")
            if self.feature_data is None or self.feature_data.empty:
                raise Exception("统一数据集为空")

            # 生成模拟价格序列用于回测
            self._generate_price_series()

            print(f"  ✅ 数据加载成功: {len(self.feature_data)}行 × {len(self.feature_data.columns)}列")
            print(f"  ✅ 价格序列生成: {len(self.price_data)}个交易日")

        except Exception as e:
            print(f"  ❌ 数据加载失败: {str(e)}")
            # 生成模拟数据作为fallback
            self._generate_mock_data()

    def _generate_price_series(self):
        """基于特征数据生成价格序列"""
        # 计算交易日数量
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        trading_days = int((end - start).days * 0.7)  # 假设70%的日子是交易日

        # 为每只股票生成价格序列
        self.price_data = {}

        for symbol in self.symbols:
            # 获取该股票的特征数据
            if symbol in self.feature_data.index:
                features = self.feature_data.loc[symbol]

                # 基于基本面和技术面特征估算价格趋势
                base_price = 100.0  # 基础价格

                # 基本面因子影响价格
                fundamental_factor = 1.0
                if 'fundamental_profitability_score' in features:
                    fundamental_factor *= (1 + features['fundamental_profitability_score'] * 0.1)
                if 'fundamental_valuation_score' in features:
                    fundamental_factor *= (1 + features['fundamental_valuation_score'] * 0.05)

                # 技术面因子影响波动
                volatility = 0.02  # 基础波动率
                if 'technical_price' in features and features['technical_price'] > 0:
                    base_price = features['technical_price']

                # 生成价格序列
                prices = []
                current_price = base_price * fundamental_factor

                for day in range(trading_days):
                    # 添加随机波动
                    daily_return = np.random.normal(0.0005, volatility)  # 略微正向偏移
                    current_price *= (1 + daily_return)
                    prices.append(current_price)

                self.price_data[symbol] = np.array(prices)
            else:
                # 如果没有特征数据，生成简单的随机游走
                prices = [100.0]
                for _ in range(trading_days - 1):
                    daily_return = np.random.normal(0.0005, 0.02)
                    prices.append(prices[-1] * (1 + daily_return))
                self.price_data[symbol] = np.array(prices)

        # 生成时间索引
        self.trading_dates = pd.date_range(
            start=self.start_date,
            periods=trading_days,
            freq='B'  # 工作日
        )

    def _generate_mock_data(self):
        """生成模拟数据作为fallback"""
        print("  🔄 生成模拟数据...")

        # 生成基础特征数据
        n_features = 20
        mock_features = {}

        for symbol in self.symbols:
            features = {}
            features.update({f'fundamental_feature_{i}': np.random.random() for i in range(5)})
            features.update({f'technical_feature_{i}': np.random.random() for i in range(5)})
            features.update({f'sentiment_feature_{i}': np.random.random() for i in range(5)})
            features.update({f'macro_feature_{i}': np.random.random() for i in range(5)})
            mock_features[symbol] = features

        self.feature_data = pd.DataFrame(mock_features).T

        # 生成价格数据
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        trading_days = int((end - start).days * 0.7)

        self.price_data = {}
        for symbol in self.symbols:
            prices = [100.0]
            for _ in range(trading_days - 1):
                daily_return = np.random.normal(0.0005, 0.02)
                prices.append(prices[-1] * (1 + daily_return))
            self.price_data[symbol] = np.array(prices)

        self.trading_dates = pd.date_range(
            start=self.start_date,
            periods=trading_days,
            freq='B'
        )

    def _setup_spaces(self):
        """设置观察空间和动作空间"""

        # 观察空间: 包含所有特征 + 投资组合状态 + 市场状态
        n_features = len(self.feature_data.columns) if self.feature_data is not None else 20
        n_symbols = len(self.symbols)

        # 特征维度计算:
        # - 股票特征: n_features * n_symbols
        # - 投资组合状态: n_symbols (当前持仓比例)
        # - 账户状态: 3 (余额比例, 总价值变化, 当前收益率)
        # - 市场状态: n_symbols (当前价格相对变化)
        obs_dim = n_features * n_symbols + n_symbols + 3 + n_symbols

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 动作空间: 每只股票的交易决策 [卖出=-1, 持有=0, 买入=1]
        # 使用MultiDiscrete来表示多只股票的同时决策
        self.action_space = gym.spaces.MultiDiscrete([3] * n_symbols)

        print(f"  📊 观察空间: {self.observation_space.shape}")
        print(f"  🎮 动作空间: {self.action_space.nvec}")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置环境状态
        self.current_step = self.lookback_window  # 从lookback窗口后开始
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance

        # 重置历史记录
        self.transaction_history = []
        self.reward_history = []
        self.state_history = []
        self.sharpe_window = []
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance

        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步交易"""

        # 执行交易动作
        transaction_cost = self._execute_trades(action)

        # 更新投资组合价值
        self._update_portfolio_value()

        # 计算奖励
        reward = self._calculate_reward(transaction_cost)

        # 检查是否结束
        self.current_step += 1
        terminated = self.current_step >= len(self.trading_dates) - 1
        truncated = False

        # 记录历史
        self.reward_history.append(reward)
        self.state_history.append(self._get_observation())

        # 获取下一个观察和信息
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_trades(self, actions: np.ndarray) -> float:
        """执行交易并返回交易成本"""
        total_cost = 0.0
        current_prices = self._get_current_prices()

        for i, symbol in enumerate(self.symbols):
            action = actions[i]  # 0=卖出, 1=持有, 2=买入
            current_price = current_prices[symbol]

            if action == 0:  # 卖出
                if self.positions[symbol] > 0:
                    # 卖出全部持股
                    sell_value = self.positions[symbol] * current_price
                    cost = sell_value * self.transaction_cost

                    self.balance += sell_value - cost
                    self.positions[symbol] = 0.0
                    total_cost += cost

                    self.transaction_history.append({
                        'step': self.current_step,
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': self.positions[symbol],
                        'price': current_price,
                        'value': sell_value,
                        'cost': cost
                    })

            elif action == 2:  # 买入
                # 计算可买入金额 (限制最大仓位)
                max_investment = self.portfolio_value * self.max_position_size
                current_value = self.positions[symbol] * current_price
                available_investment = min(self.balance, max_investment - current_value)

                if available_investment > current_price:  # 至少能买一股
                    buy_quantity = available_investment / current_price
                    buy_value = buy_quantity * current_price
                    cost = buy_value * self.transaction_cost

                    if self.balance >= buy_value + cost:
                        self.balance -= buy_value + cost
                        self.positions[symbol] += buy_quantity
                        total_cost += cost

                        self.transaction_history.append({
                            'step': self.current_step,
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': buy_quantity,
                            'price': current_price,
                            'value': buy_value,
                            'cost': cost
                        })

            # action == 1 (持有) 不需要任何操作

        return total_cost

    def _update_portfolio_value(self):
        """更新投资组合价值"""
        current_prices = self._get_current_prices()

        # 计算持股价值
        holdings_value = sum(
            self.positions[symbol] * current_prices[symbol]
            for symbol in self.symbols
        )

        # 更新投资组合总价值
        self.previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self.balance + holdings_value

        # 更新最大回撤
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def _calculate_reward(self, transaction_cost: float) -> float:
        """计算综合奖励"""

        # 1. 基础收益奖励
        portfolio_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        return_reward = portfolio_return * 100  # 放大收益信号

        # 2. 风险调整奖励 (Sharpe ratio)
        self.sharpe_window.append(portfolio_return)
        if len(self.sharpe_window) > 30:  # 保持30天窗口
            self.sharpe_window.pop(0)

        sharpe_reward = 0.0
        if len(self.sharpe_window) >= 10:  # 至少10天数据
            returns = np.array(self.sharpe_window)
            if np.std(returns) > 0:
                sharpe_ratio = (np.mean(returns) - self.risk_free_rate/252) / np.std(returns)
                sharpe_reward = sharpe_ratio * 0.1  # Sharpe比率奖励

        # 3. 交易成本惩罚
        cost_penalty = -transaction_cost / self.portfolio_value * 10  # 交易成本惩罚

        # 4. 回撤惩罚
        drawdown_penalty = -self.max_drawdown * 0.5

        # 5. 多样化奖励 (鼓励分散投资)
        diversity_reward = 0.0
        current_prices = self._get_current_prices()
        total_holdings_value = sum(
            self.positions[symbol] * current_prices[symbol]
            for symbol in self.symbols
        )

        if total_holdings_value > 0:
            weights = [
                self.positions[symbol] * current_prices[symbol] / total_holdings_value
                for symbol in self.symbols
            ]
            # 使用Herfindahl指数衡量集中度，奖励分散投资
            hhi = sum(w**2 for w in weights)
            diversity_reward = (1 - hhi) * 0.1  # 分散投资奖励

        # 综合奖励
        total_reward = (
            return_reward +          # 收益奖励 (主要)
            sharpe_reward +          # 风险调整收益
            cost_penalty +           # 交易成本控制
            drawdown_penalty +       # 回撤控制
            diversity_reward         # 分散投资
        )

        return total_reward

    def _get_observation(self) -> np.ndarray:
        """获取当前观察状态"""
        observations = []

        # 1. 股票特征 (基本面 + 技术面 + 情感面 + 宏观面)
        for symbol in self.symbols:
            if symbol in self.feature_data.index:
                symbol_features = self.feature_data.loc[symbol].values
            else:
                # 如果没有特征数据，使用零向量
                symbol_features = np.zeros(len(self.feature_data.columns))

            observations.extend(symbol_features)

        # 2. 投资组合状态 (当前持仓比例)
        current_prices = self._get_current_prices()
        total_holdings_value = sum(
            self.positions[symbol] * current_prices[symbol]
            for symbol in self.symbols
        )

        for symbol in self.symbols:
            if total_holdings_value > 0:
                weight = self.positions[symbol] * current_prices[symbol] / total_holdings_value
            else:
                weight = 0.0
            observations.append(weight)

        # 3. 账户状态
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 0
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        recent_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value if self.previous_portfolio_value > 0 else 0

        observations.extend([cash_ratio, total_return, recent_return])

        # 4. 市场状态 (价格相对变化)
        for symbol in self.symbols:
            if self.current_step > 0:
                prev_price = self._get_price_at_step(symbol, self.current_step - 1)
                current_price = current_prices[symbol]
                price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            else:
                price_change = 0.0
            observations.append(price_change)

        return np.array(observations, dtype=np.float32)

    def _get_current_prices(self) -> Dict[str, float]:
        """获取当前价格"""
        prices = {}
        for symbol in self.symbols:
            prices[symbol] = self._get_price_at_step(symbol, self.current_step)
        return prices

    def _get_price_at_step(self, symbol: str, step: int) -> float:
        """获取指定步骤的价格"""
        if symbol in self.price_data and 0 <= step < len(self.price_data[symbol]):
            return self.price_data[symbol][step]
        return 100.0  # 默认价格

    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        current_prices = self._get_current_prices()

        # 计算持股价值
        holdings_value = sum(
            self.positions[symbol] * current_prices[symbol]
            for symbol in self.symbols
        )

        # 计算收益率统计
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.balance,
            'holdings_value': holdings_value,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'num_transactions': len(self.transaction_history),
            'positions': self.positions.copy(),
            'current_prices': current_prices
        }

        # 添加性能指标
        if len(self.reward_history) > 0:
            info['avg_reward'] = np.mean(self.reward_history)
            info['total_reward'] = np.sum(self.reward_history)

        if len(self.sharpe_window) >= 10:
            returns = np.array(self.sharpe_window)
            if np.std(returns) > 0:
                info['sharpe_ratio'] = (np.mean(returns) - self.risk_free_rate/252) / np.std(returns)

        return info

    def render(self, mode: str = 'human') -> None:
        """渲染环境状态"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"投资组合价值: ${self.portfolio_value:,.2f}")
            print(f"现金余额: ${self.balance:,.2f}")
            print(f"总收益率: {((self.portfolio_value - self.initial_balance) / self.initial_balance) * 100:.2f}%")
            print(f"最大回撤: {self.max_drawdown * 100:.2f}%")

            current_prices = self._get_current_prices()
            print("\n持股情况:")
            for symbol in self.symbols:
                if self.positions[symbol] > 0:
                    value = self.positions[symbol] * current_prices[symbol]
                    weight = value / self.portfolio_value * 100
                    print(f"  {symbol}: {self.positions[symbol]:.2f}股 (${value:,.2f}, {weight:.1f}%)")

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """获取投资组合统计信息"""
        if len(self.reward_history) == 0:
            return {}

        # 计算收益率序列
        returns = []
        portfolio_values = [self.initial_balance]

        # 重建投资组合价值历史
        for i, state in enumerate(self.state_history):
            if i < len(self.state_history) - 1:
                # 简化计算：基于奖励近似收益率
                if i < len(self.reward_history):
                    portfolio_values.append(portfolio_values[-1] * (1 + self.reward_history[i] / 100))

        # 计算收益率
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] > 0:
                returns.append((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1])

        if len(returns) == 0:
            return {}

        returns = np.array(returns)

        # 计算各种指标
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0

        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # 计算Calmar比率 (年化收益率 / 最大回撤)
        calmar_ratio = annualized_return / self.max_drawdown if self.max_drawdown > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(self.transaction_history),
            'final_portfolio_value': self.portfolio_value
        }


def create_trading_environment(
    symbols: List[str] = ["AAPL", "MSFT", "GOOGL"],
    **kwargs
) -> TradingEnvironment:
    """创建交易环境的便捷函数"""
    return TradingEnvironment(symbols=symbols, **kwargs)


# 示例使用
if __name__ == "__main__":
    print("🤖 StockSynergy 交易环境测试")
    print("=" * 50)

    # 创建环境
    env = create_trading_environment(
        symbols=["AAPL", "MSFT", "GOOGL"],
        initial_balance=100000.0,
        objective="balanced"
    )

    # 测试环境
    print("\n📊 环境测试:")
    observation, info = env.reset()
    print(f"初始观察维度: {observation.shape}")
    print(f"动作空间: {env.action_space}")

    # 执行几步随机动作
    print("\n🎮 执行随机交易:")
    for step in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step + 1}:")
        print(f"  动作: {action}")
        print(f"  奖励: {reward:.4f}")
        print(f"  投资组合价值: ${info['portfolio_value']:,.2f}")
        print(f"  总收益率: {info['total_return']:.2%}")

        if terminated or truncated:
            break

    # 显示最终统计
    print("\n📈 最终统计:")
    stats = env.get_portfolio_stats()
    if stats:
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n✅ 交易环境测试完成！")
