#!/usr/bin/env python3
"""
StockSynergy 快速模型优化 - 5分钟版本

专注于关键优化点，快速提升模型性能：
1. ⚡ 精简参数网格 (减少90%搜索空间)
2. 🎯 智能特征选择 (Top 15特征)
3. 🔧 快速集成学习
4. 📊 高效评估流程

目标: 5分钟内完成，准确率提升至65%+
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class QuickOptimizer:
    """
    快速优化器 - 5分钟搞定
    """

    def __init__(self):
        self.results = {}

    def quick_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """快速特征工程 - 只添加最有效的特征"""
        print("⚡ 快速特征工程...")

        df = data.copy()

        # 只添加最有效的技术指标
        if 'Returns' in df.columns:
            # 短期动量
            df['momentum_3'] = df['Returns'].rolling(3).mean()
            df['momentum_7'] = df['Returns'].rolling(7).mean()

            # 波动率
            df['volatility_5'] = df['Returns'].rolling(5).std()
            df['volatility_20'] = df['Returns'].rolling(20).std()

            # 价格加速度
            df['acceleration'] = df['Returns'].diff()

        # RSI相关
        if 'RSI' in df.columns:
            df['rsi_signal'] = np.where(df['RSI'] < 30, 2, np.where(df['RSI'] > 70, 0, 1))
            df['rsi_momentum'] = df['RSI'].diff()

        # 均线交叉
        if 'SMA_5' in df.columns and 'SMA_20' in df.columns:
            df['sma_ratio'] = df['SMA_5'] / df['SMA_20']
            df['sma_signal'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 0)

        # 成交量
        if 'Volume_MA' in df.columns:
            df['volume_signal'] = np.where(df['Volume_MA'] > df['Volume_MA'].rolling(10).mean(), 1, 0)

        print(f"  ✅ 新增特征: {len(df.columns) - len(data.columns)}个")
        return df.fillna(method='ffill').fillna(0)

    def smart_feature_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> List[str]:
        """智能特征选择 - 快速选出最重要的特征"""
        print(f"🎯 智能特征选择 (Top {n_features})...")

        # 移除非数值特征
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_features].fillna(0)

        # 使用F检验快速选择
        selector = SelectKBest(score_func=f_classif, k=min(n_features, len(numeric_features)))
        selector.fit(X_numeric, y)

        selected_features = X_numeric.columns[selector.get_support()].tolist()

        # 显示特征重要性
        feature_scores = selector.scores_[selector.get_support()]
        feature_importance = dict(zip(selected_features, feature_scores))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        print("  🔥 Top 10重要特征:")
        for feature, score in sorted_features[:10]:
            print(f"    • {feature}: {score:.1f}")

        return selected_features

    def balance_data_quick(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """快速数据平衡"""
        print("⚖️ 快速数据平衡...")

        from sklearn.utils import resample

        # 合并数据
        data = X.copy()
        data['target'] = y

        # 找到最小类别数量（避免数据膨胀）
        min_size = data['target'].value_counts().min()
        target_size = min(min_size * 2, 8000)  # 限制最大样本数

        # 平衡各类别
        balanced_groups = []
        for class_label in data['target'].unique():
            class_data = data[data['target'] == class_label]

            if len(class_data) > target_size:
                # 下采样
                sampled = resample(class_data,
                                 replace=False,
                                 n_samples=target_size,
                                 random_state=42)
            else:
                # 上采样
                sampled = resample(class_data,
                                 replace=True,
                                 n_samples=target_size,
                                 random_state=42)

            balanced_groups.append(sampled)

        balanced_data = pd.concat(balanced_groups, ignore_index=True)

        X_balanced = balanced_data.drop('target', axis=1)
        y_balanced = balanced_data['target']

        print(f"  ✅ 平衡完成: {len(X_balanced)}样本")
        print(f"  📊 类别分布: {y_balanced.value_counts().to_dict()}")

        return X_balanced, y_balanced

    def optimize_random_forest_quick(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """快速优化Random Forest"""
        print("🌲 快速优化Random Forest...")

        # 精简参数网格 (只搜索关键参数)
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # 随机搜索 (比网格搜索快很多)
        random_search = RandomizedSearchCV(
            rf, param_distributions,
            n_iter=20,  # 只试20个组合
            cv=3,  # 只做3折验证
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X, y)

        print(f"  ✅ 最佳参数: {random_search.best_params_}")
        print(f"  📊 最佳得分: {random_search.best_score_:.3f}")

        return {
            'model': random_search.best_estimator_,
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_
        }

    def optimize_xgboost_quick(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """快速优化XGBoost"""
        print("🚀 快速优化XGBoost...")

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        xgb_model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )

        random_search = RandomizedSearchCV(
            xgb_model, param_distributions,
            n_iter=20,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X, y)

        print(f"  ✅ 最佳参数: {random_search.best_params_}")
        print(f"  📊 最佳得分: {random_search.best_score_:.3f}")

        return {
            'model': random_search.best_estimator_,
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_
        }

    def create_gradient_boosting_quick(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """快速创建Gradient Boosting"""
        print("📈 快速优化Gradient Boosting...")

        param_distributions = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9]
        }

        gb = GradientBoostingClassifier(random_state=42)

        random_search = RandomizedSearchCV(
            gb, param_distributions,
            n_iter=15,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X, y)

        print(f"  ✅ 最佳参数: {random_search.best_params_}")
        print(f"  📊 最佳得分: {random_search.best_score_:.3f}")

        return {
            'model': random_search.best_estimator_,
            'best_score': random_search.best_score_,
            'best_params': random_search.best_params_
        }

    def create_quick_ensemble(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """快速创建集成模型"""
        print("🎭 快速集成学习...")

        # 简单投票集成
        estimators = [(name, model['model']) for name, model in models.items()]

        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',  # 软投票通常更好
            n_jobs=-1
        )

        voting_clf.fit(X, y)

        # 简单交叉验证
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(voting_clf, X, y, cv=3, scoring='accuracy')

        print(f"  ✅ 集成准确率: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        return {
            'model': voting_clf,
            'best_score': cv_scores.mean(),
            'cv_scores': cv_scores
        }

def run_quick_optimization():
    """运行快速优化 - 5分钟版本"""
    print("⚡ StockSynergy 快速模型优化 (5分钟版本)")
    print("=" * 60)

    start_time = datetime.now()

    # 1. 加载数据
    print("\n📊 加载历史数据...")
    try:
        with open('data/historical/complete_dataset_2020-01-01_2024-01-01.pkl', 'rb') as f:
            dataset = pickle.load(f)

        X_raw = dataset['features']
        y_raw = dataset['labels']

        print(f"  ✅ 原始数据: {len(X_raw)}样本, {len(X_raw.columns)}特征")

    except FileNotFoundError:
        print("  ❌ 未找到历史数据文件，请先运行 collect_historical_data.py")
        return

    # 2. 快速优化器
    optimizer = QuickOptimizer()

    # 3. 快速特征工程
    print("\n⚡ 快速特征工程...")
    X_enhanced = optimizer.quick_feature_engineering(X_raw)

    # 4. 智能特征选择
    selected_features = optimizer.smart_feature_selection(X_enhanced, y_raw, n_features=15)
    X_selected = X_enhanced[selected_features]

    # 5. 快速数据平衡
    X_balanced, y_balanced = optimizer.balance_data_quick(X_selected, y_raw)

    # 6. 数据分割
    print("\n✂️ 数据分割...")
    split_idx = int(len(X_balanced) * 0.8)
    X_train, X_test = X_balanced[:split_idx], X_balanced[split_idx:]
    y_train, y_test = y_balanced[:split_idx], y_balanced[split_idx:]

    print(f"  📊 训练集: {len(X_train)}样本")
    print(f"  📊 测试集: {len(X_test)}样本")

    # 7. 快速模型优化
    print("\n🚀 快速模型优化...")

    rf_result = optimizer.optimize_random_forest_quick(X_train, y_train)
    xgb_result = optimizer.optimize_xgboost_quick(X_train, y_train)
    gb_result = optimizer.create_gradient_boosting_quick(X_train, y_train)

    base_models = {
        'random_forest': rf_result,
        'xgboost': xgb_result,
        'gradient_boosting': gb_result
    }

    # 8. 快速集成
    ensemble_result = optimizer.create_quick_ensemble(base_models, X_train, y_train)

    # 9. 最终评估
    print("\n📊 最终评估...")

    all_models = {**base_models, 'ensemble': ensemble_result}

    # 找出最佳模型
    best_model_name = max(all_models.keys(), key=lambda x: all_models[x]['best_score'])
    best_model = all_models[best_model_name]

    # 测试集评估
    test_predictions = best_model['model'].predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # 计算运行时间
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    print(f"\n🏆 优化结果:")
    print(f"  ⏱️ 总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"  🎯 最佳模型: {best_model_name}")
    print(f"  📊 交叉验证得分: {best_model['best_score']:.3f}")
    print(f"  📊 测试集准确率: {test_accuracy:.3f}")

    # 性能提升分析
    original_accuracy = 0.52  # 原始最佳验证准确率
    improvement = test_accuracy - original_accuracy

    print(f"\n📈 性能提升分析:")
    print(f"  📊 原始准确率: {original_accuracy:.1%}")
    print(f"  📊 优化后准确率: {test_accuracy:.1%}")
    print(f"  🚀 性能提升: {improvement:+.1%}")

    success = test_accuracy >= 0.65
    print(f"  🎯 目标达成: {'✅ 是' if success else '❌ 否'} (目标: 65%+)")

    # 10. 生成快速报告
    print(f"\n📋 生成快速报告...")

    report_lines = [
        "=" * 60,
        "StockSynergy 快速模型优化报告",
        "=" * 60,
        f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总用时: {total_time:.1f}秒",
        "",
        "📊 优化结果:",
        f"- 最佳模型: {best_model_name}",
        f"- 测试准确率: {test_accuracy:.1%}",
        f"- 性能提升: {improvement:+.1%}",
        f"- 目标达成: {'✅' if success else '❌'}",
        "",
        "🔧 优化技术:",
        "- 智能特征选择 (Top 15)",
        "- 随机搜索优化 (vs 网格搜索)",
        "- 快速数据平衡",
        "- 集成学习",
        "",
        "⚡ 效率提升:",
        f"- 搜索空间: 减少90%+",
        f"- 运行时间: {total_time/60:.1f}分钟 (vs 预估3-12小时)",
        f"- 样本规模: {len(X_balanced):,} (优化后)",
        "",
        "📈 模型对比:"
    ]

    for name, result in all_models.items():
        test_pred = result['model'].predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        report_lines.append(f"- {name}: CV={result['best_score']:.3f}, Test={test_acc:.3f}")

    report_lines.extend([
        "",
        "💡 下一步建议:",
        "- 如已达标: 部署模型进行实盘测试",
        "- 如未达标: 收集更多数据或尝试深度学习",
        "",
        "=" * 60
    ])

    # 保存报告
    report_content = "\n".join(report_lines)
    report_file = f"quick_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"📄 报告已保存: {report_file}")

    # 保存最佳模型
    if success:
        model_file = f"best_model_quick_{best_model_name}_{datetime.now().strftime('%Y%m%d')}.pkl"

        model_package = {
            'model': best_model['model'],
            'feature_names': selected_features,
            'model_name': best_model_name,
            'test_accuracy': test_accuracy,
            'optimization_time': total_time
        }

        with open(model_file, 'wb') as f:
            pickle.dump(model_package, f)

        print(f"💾 最佳模型已保存: {model_file}")

    return {
        'best_model': best_model,
        'test_accuracy': test_accuracy,
        'optimization_time': total_time,
        'success': success
    }

if __name__ == "__main__":
    try:
        results = run_quick_optimization()

        print(f"\n🎉 快速优化完成！")
        print(f"⏱️ 用时: {results['optimization_time']:.1f}秒")
        print(f"📊 准确率: {results['test_accuracy']:.1%}")
        print(f"🎯 状态: {'✅ 成功' if results['success'] else '⚠️ 待改进'}")

    except Exception as e:
        print(f"❌ 快速优化失败: {str(e)}")
        import traceback
        traceback.print_exc()