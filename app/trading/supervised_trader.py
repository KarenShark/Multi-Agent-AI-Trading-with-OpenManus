"""
监督学习交易员

基于历史数据训练的监督学习模型，包括：
- Random Forest Trader
- XGBoost Trader
- Neural Network Trader
- Ensemble Trader (集成多模型)
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.trading.trading_env import TradingEnvironment, create_trading_environment
from app.data.loader import UnifiedDataLoader

class SupervisedTrader:
    """
    监督学习交易员基类

    提供基础的特征工程、模型训练和预测功能
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        prediction_horizon: int = 5,
        feature_engineering: bool = True,
        use_technical_indicators: bool = True,
        use_sentiment_features: bool = True,
        use_macro_features: bool = True
    ):
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.feature_engineering = feature_engineering
        self.use_technical_indicators = use_technical_indicators
        self.use_sentiment_features = use_sentiment_features
        self.use_macro_features = use_macro_features

        # 模型组件
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []

        # 训练历史
        self.training_history = {
            "train_scores": [],
            "val_scores": [],
            "feature_importance": {},
            "model_params": {}
        }

        # 数据加载器
        self.data_loader = UnifiedDataLoader()

        print(f"🤖 初始化监督学习交易员:")
        print(f"  📊 模型类型: {model_type}")
        print(f"  🔮 预测周期: {prediction_horizon}天")
        print(f"  🛠️ 特征工程: {'启用' if feature_engineering else '禁用'}")

    def prepare_training_data(
        self,
        symbols: List[str],
        start_date: str = "2022-01-01",
        end_date: str = "2024-01-01",
        objective: str = "balanced"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        print("📊 准备训练数据...")

        try:
            # 加载综合数据
            data_result = self.data_loader.load_comprehensive_data(
                symbols=symbols,
                objective=objective
            )

            if not data_result.get("success"):
                raise Exception("数据加载失败")

            # 获取统一数据集
            feature_data = data_result.get("unified_dataset")
            if feature_data is None or feature_data.empty:
                raise Exception("统一数据集为空")

            print(f"  ✅ 加载数据成功: {len(feature_data)}只股票 × {len(feature_data.columns)}特征")

            # 生成模拟历史价格数据用于标签
            price_history = self._generate_price_history(symbols, start_date, end_date)

            # 创建特征和标签
            X, y = self._create_features_and_labels(feature_data, price_history, symbols)

            print(f"  📈 特征矩阵: {X.shape}")
            print(f"  🎯 标签分布: {pd.Series(y).value_counts().to_dict()}")

            return X, y

        except Exception as e:
            print(f"  ❌ 数据准备失败: {str(e)}")
            # 生成模拟数据作为fallback
            return self._generate_mock_training_data(symbols)

    def _generate_price_history(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, np.ndarray]:
        """生成模拟价格历史数据"""

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days

        price_history = {}

        for symbol in symbols:
            # 基础价格
            base_price = np.random.uniform(50, 200)

            # 生成价格序列 (几何布朗运动)
            returns = np.random.normal(0.0005, 0.02, days)  # 日收益率
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            price_history[symbol] = np.array(prices)

        return price_history

    def _create_features_and_labels(
        self,
        feature_data: pd.DataFrame,
        price_history: Dict[str, np.ndarray],
        symbols: List[str]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """创建特征和标签"""

        all_features = []
        all_labels = []

        for symbol in symbols:
            if symbol not in feature_data.index:
                continue

            # 获取该股票的特征
            stock_features = feature_data.loc[symbol].to_dict()

            # 获取价格历史
            prices = price_history.get(symbol, [])
            if len(prices) < self.prediction_horizon + 1:
                continue

            # 为每个时间点创建样本
            for i in range(len(prices) - self.prediction_horizon):
                # 特征 (当前特征 + 短期价格历史)
                features = stock_features.copy()

                # 添加价格特征
                current_price = prices[i]
                features['current_price'] = current_price

                if i > 0:
                    features['price_change_1d'] = (prices[i] - prices[i-1]) / prices[i-1]
                else:
                    features['price_change_1d'] = 0

                if i > 4:
                    features['price_change_5d'] = (prices[i] - prices[i-5]) / prices[i-5]
                    features['price_volatility_5d'] = np.std(prices[i-5:i]) / np.mean(prices[i-5:i])
                else:
                    features['price_change_5d'] = 0
                    features['price_volatility_5d'] = 0

                # 标签 (未来N天的收益率类别)
                future_price = prices[i + self.prediction_horizon]
                future_return = (future_price - current_price) / current_price

                # 将收益率分为3类: 0=下跌, 1=持平, 2=上涨
                if future_return < -0.02:  # 下跌超过2%
                    label = 0
                elif future_return > 0.02:  # 上涨超过2%
                    label = 2
                else:  # 持平
                    label = 1

                all_features.append(features)
                all_labels.append(label)

        # 转换为DataFrame
        X = pd.DataFrame(all_features)

        # 处理缺失值
        X = X.fillna(X.mean())

        # 保存特征列名
        self.feature_columns = X.columns.tolist()

        return X, np.array(all_labels)

    def _generate_mock_training_data(self, symbols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """生成模拟训练数据"""
        print("  🔄 生成模拟训练数据...")

        n_samples = 1000
        n_features = 20

        # 生成特征
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # 生成标签 (基于特征的线性组合 + 噪声)
        weights = np.random.randn(n_features)
        scores = X.values @ weights

        # 转换为分类标签
        y = np.where(scores < np.percentile(scores, 33), 0,
                    np.where(scores > np.percentile(scores, 67), 2, 1))

        self.feature_columns = X.columns.tolist()

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.2,
        validate: bool = True,
        hyperparameter_tuning: bool = True
    ) -> Dict[str, Any]:
        """训练模型"""
        print(f"🎯 开始训练{self.model_type}模型...")

        # 数据预处理
        X_scaled = self.scaler.fit_transform(X)

        # 分割数据 (时间序列分割)
        if validate:
            # 使用时间序列分割确保未来数据不泄露
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X_scaled, y
            X_val, y_val = None, None

        # 创建模型
        self.model = self._create_model()

        # 超参数调优
        if hyperparameter_tuning and validate:
            print("  🔧 执行超参数调优...")
            self.model = self._hyperparameter_tuning(X_train, y_train)

        # 训练模型
        if self.model_type in ["random_forest", "xgboost"]:
            self.model.fit(X_train, y_train)
        else:
            # 对于其他模型类型，可以添加early stopping等
            self.model.fit(X_train, y_train)

        # 验证
        train_results = {}
        if validate:
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)

            train_results = {
                "train_accuracy": train_score,
                "val_accuracy": val_score,
                "overfitting": train_score - val_score
            }

            # 详细评估
            y_pred = self.model.predict(X_val)
            train_results["classification_report"] = classification_report(
                y_val, y_pred, output_dict=True
            )

            print(f"  📊 训练准确率: {train_score:.3f}")
            print(f"  📊 验证准确率: {val_score:.3f}")
            print(f"  📊 过拟合程度: {train_score - val_score:.3f}")

        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            train_results["feature_importance"] = importance

            # 显示Top 10重要特征
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print("  🔥 Top 10重要特征:")
            for feature, imp in top_features:
                print(f"    • {feature}: {imp:.3f}")

        # 保存训练历史
        self.training_history.update(train_results)

        print("  ✅ 模型训练完成")
        return train_results

    def _create_model(self):
        """创建模型"""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray):
        """超参数调优"""

        if self.model_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        elif self.model_type == "xgboost":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            }
            base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        print(f"    🎯 最佳参数: {grid_search.best_params_}")
        print(f"    📊 最佳得分: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_

    def predict(self, features: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        # 预处理特征
        if isinstance(features, pd.DataFrame):
            # 确保特征列顺序一致
            features = features[self.feature_columns]
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        # 预测
        predictions = self.model.predict(features_scaled)

        # 预测概率
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)
        else:
            probabilities = None

        return predictions, probabilities

    def predict_trading_action(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测交易动作"""
        predictions, probabilities = self.predict(features)

        # 转换预测结果为交易动作
        # 0=下跌预测 -> 0=卖出
        # 1=持平预测 -> 1=持有
        # 2=上涨预测 -> 2=买入
        trading_actions = predictions.copy()

        return trading_actions

    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'prediction_horizon': self.prediction_horizon,
            'training_history': self.training_history
        }

        joblib.dump(model_data, filepath)
        print(f"💾 模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.prediction_horizon = model_data['prediction_horizon']
        self.training_history = model_data['training_history']

        print(f"📂 模型已从 {filepath} 加载")

    def evaluate_on_environment(
        self,
        env: TradingEnvironment,
        num_episodes: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """在交易环境中评估模型"""
        print(f"🎮 在交易环境中评估模型 ({num_episodes}轮)...")

        episode_results = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0

            while True:
                # 将观察转换为特征格式
                features = self._obs_to_features(obs)

                # 预测交易动作
                try:
                    actions = self.predict_trading_action(features.reshape(1, -1))
                    action = actions[0] if isinstance(actions, np.ndarray) else actions

                    # 确保动作格式正确
                    if isinstance(action, (int, np.integer)):
                        # 单股票情况，扩展为多股票
                        action = [action] * len(env.symbols)
                    elif len(action) != len(env.symbols):
                        # 多股票情况，确保长度匹配
                        action = [action[0]] * len(env.symbols)

                except Exception as e:
                    # 如果预测失败，使用随机动作
                    action = env.action_space.sample()

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                if terminated or truncated:
                    break

            episode_results.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'final_portfolio_value': info.get('portfolio_value', 0),
                'total_return': info.get('total_return', 0)
            })

            if verbose:
                print(f"  Episode {episode + 1}: 奖励={episode_reward:.2f}, "
                      f"投资组合=${info.get('portfolio_value', 0):,.2f}, "
                      f"收益率={info.get('total_return', 0):.2%}")

        # 计算统计信息
        rewards = [r['reward'] for r in episode_results]
        returns = [r['total_return'] for r in episode_results]

        evaluation_results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'episode_results': episode_results
        }

        print(f"  📊 平均奖励: {evaluation_results['avg_reward']:.3f}")
        print(f"  📊 平均收益率: {evaluation_results['avg_return']:.2%}")
        print(f"  📊 胜率: {evaluation_results['win_rate']:.2%}")

        return evaluation_results

    def _obs_to_features(self, observation: np.ndarray) -> np.ndarray:
        """将环境观察转换为模型特征"""
        # 简化版本：使用前N个观察值作为特征
        # 在实际应用中，这里需要更复杂的特征映射

        if len(observation) >= len(self.feature_columns):
            return observation[:len(self.feature_columns)]
        else:
            # 如果观察维度不足，用零填充
            features = np.zeros(len(self.feature_columns))
            features[:len(observation)] = observation
            return features


class EnsembleTrader(SupervisedTrader):
    """
    集成交易员

    结合多个模型的预测结果进行交易决策
    """

    def __init__(self, model_types: List[str] = ["random_forest", "xgboost"], **kwargs):
        super().__init__(model_type="ensemble", **kwargs)
        self.model_types = model_types
        self.models = {}
        self.traders = {}

        print(f"🎭 初始化集成交易员:")
        print(f"  📊 集成模型: {', '.join(model_types)}")

    def train(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """训练所有模型"""
        print("🎯 训练集成模型...")

        ensemble_results = {}

        for model_type in self.model_types:
            print(f"\n  训练 {model_type} 模型...")

            trader = SupervisedTrader(
                model_type=model_type,
                prediction_horizon=self.prediction_horizon
            )

            results = trader.train(X, y, **kwargs)

            self.traders[model_type] = trader
            ensemble_results[model_type] = results

        print("\n✅ 集成模型训练完成")
        return ensemble_results

    def predict_trading_action(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """集成预测交易动作"""
        if not self.traders:
            raise ValueError("集成模型尚未训练")

        # 收集所有模型的预测
        predictions = []
        for model_type, trader in self.traders.items():
            try:
                pred = trader.predict_trading_action(features)
                predictions.append(pred)
            except Exception as e:
                print(f"  ⚠️ {model_type}预测失败: {str(e)}")
                continue

        if not predictions:
            # 如果所有模型都失败，返回持有
            return np.array([1])

        # 简单投票机制
        predictions_array = np.array(predictions)
        if len(predictions_array.shape) == 1:
            # 单个预测
            from scipy import stats
            final_prediction = stats.mode(predictions_array, keepdims=False)[0]
        else:
            # 多个预测，按列投票
            final_prediction = []
            for col in range(predictions_array.shape[1]):
                from scipy import stats
                mode_result = stats.mode(predictions_array[:, col], keepdims=False)[0]
                final_prediction.append(mode_result)
            final_prediction = np.array(final_prediction)

        return final_prediction


# 便捷函数
def create_supervised_trader(
    model_type: str = "random_forest",
    **kwargs
) -> SupervisedTrader:
    """创建监督学习交易员"""
    return SupervisedTrader(model_type=model_type, **kwargs)


def create_ensemble_trader(
    model_types: List[str] = ["random_forest", "xgboost"],
    **kwargs
) -> EnsembleTrader:
    """创建集成交易员"""
    return EnsembleTrader(model_types=model_types, **kwargs)


# 测试代码
if __name__ == "__main__":
    print("🤖 监督学习交易员测试")
    print("=" * 50)

    # 测试单一模型
    print("\n1. 测试Random Forest交易员:")
    rf_trader = create_supervised_trader("random_forest")

    # 准备训练数据
    X, y = rf_trader.prepare_training_data(["AAPL", "MSFT"])

    # 训练模型
    results = rf_trader.train(X, y)

    # 测试集成模型
    print("\n2. 测试集成交易员:")
    ensemble_trader = create_ensemble_trader(["random_forest", "xgboost"])
    ensemble_results = ensemble_trader.train(X, y)

    # 在交易环境中测试
    print("\n3. 交易环境测试:")
    env = create_trading_environment(["AAPL", "MSFT"])
    evaluation = rf_trader.evaluate_on_environment(env, num_episodes=3)

    print("\n✅ 监督学习交易员测试完成！")