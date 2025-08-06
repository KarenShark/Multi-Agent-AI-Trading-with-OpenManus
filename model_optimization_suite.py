#!/usr/bin/env python3
"""
StockSynergy 模型优化套件

系统性提升AI交易模型性能，包括：
1. 🔧 特征工程优化
2. 🤖 模型架构升级
3. 📊 数据质量提升
4. ⚡ 超参数精调
5. 🧠 集成学习强化
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 机器学习核心库
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from scipy import stats

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AdvancedFeatureEngineer:
    """
    高级特征工程器
    """

    def __init__(self):
        self.feature_importance = {}
        self.selected_features = []

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建高级技术指标特征"""
        print("🔧 创建高级技术指标...")

        df = data.copy()

        # 1. 价格动量特征
        for period in [3, 7, 14, 21]:
            df[f'price_momentum_{period}'] = df['Returns'].rolling(period).mean()
            df[f'price_acceleration_{period}'] = df['Returns'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
            )

        # 2. 波动率特征
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['Returns'].rolling(period).std()
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df['Returns'].rolling(60).std()

        # 3. 相对强弱指标
        if 'RSI' in df.columns:
            df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
            df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
            df['rsi_momentum'] = df['RSI'].diff()

        # 4. 均线交叉信号
        if 'SMA_5' in df.columns and 'SMA_20' in df.columns:
            df['sma_cross_signal'] = np.where(df['SMA_5'] > df['SMA_20'], 1, -1)
            df['sma_divergence'] = (df['SMA_5'] - df['SMA_20']) / df['SMA_20']

        # 5. 成交量特征
        if 'Volume_MA' in df.columns:
            df['volume_spike'] = (df['Volume_MA'] > df['Volume_MA'].rolling(20).mean() * 1.5).astype(int)
            df['volume_trend'] = df['Volume_MA'].rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
            )

        # 6. 市场状态特征
        df['trend_strength'] = abs(df['Returns'].rolling(10).mean()) / df['Returns'].rolling(10).std()
        df['market_regime'] = pd.cut(df['Returns'].rolling(20).mean(),
                                    bins=[-np.inf, -0.01, 0.01, np.inf],
                                    labels=[0, 1, 2])  # 下跌、震荡、上涨

        # 7. 统计特征
        df['returns_skewness'] = df['Returns'].rolling(20).skew()
        df['returns_kurtosis'] = df['Returns'].rolling(20).kurt()

        print(f"  ✅ 新增特征: {len(df.columns) - len(data.columns)}个")
        return df.fillna(method='ffill').fillna(0)

    def create_cross_asset_features(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """创建跨资产特征"""
        print("🌐 创建跨资产特征...")

        all_features = []

        for symbol, data in stock_data.items():
            # 基础特征
            features = data.copy()
            features['symbol'] = symbol

            # 相对表现特征
            market_returns = pd.concat([d['Returns'] for d in stock_data.values()], axis=1).mean(axis=1)
            features['relative_performance'] = features['Returns'] - market_returns
            features['beta'] = features['Returns'].rolling(60).corr(market_returns)

            # 排名特征
            all_returns = pd.concat([d['Returns'] for d in stock_data.values()], axis=1)
            features['return_rank'] = all_returns.rank(axis=1, pct=True)[symbol] if symbol in all_returns.columns else 0.5

            all_features.append(features)

        combined = pd.concat(all_features, ignore_index=True)
        print(f"  ✅ 跨资产特征创建完成")
        return combined

    def select_best_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> List[str]:
        """特征选择"""
        print(f"🎯 选择最佳特征 (目标: {n_features}个)...")

        # 移除非数值特征
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_features].fillna(0)

        # 方法1: 统计检验
        selector1 = SelectKBest(score_func=f_classif, k=min(n_features*2, len(numeric_features)))
        selector1.fit(X_numeric, y)
        selected1 = X_numeric.columns[selector1.get_support()].tolist()

        # 方法2: 递归特征消除 (使用Random Forest)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        selector2 = RFE(rf, n_features_to_select=min(n_features, len(numeric_features)))
        selector2.fit(X_numeric, y)
        selected2 = X_numeric.columns[selector2.get_support()].tolist()

        # 结合两种方法
        final_features = list(set(selected1 + selected2))[:n_features]

        print(f"  ✅ 选择特征: {len(final_features)}个")
        return final_features

class ModelOptimizer:
    """
    模型优化器
    """

    def __init__(self):
        self.best_models = {}
        self.optimization_history = []

    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """优化Random Forest"""
        print("🌲 优化Random Forest...")

        # 扩展参数网格
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.8, 0.9],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            rf, param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        print(f"  ✅ 最佳参数: {grid_search.best_params_}")
        print(f"  📊 最佳得分: {grid_search.best_score_:.3f}")

        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """优化XGBoost"""
        print("🚀 优化XGBoost...")

        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.5, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }

        xgb_model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )

        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            xgb_model, param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        print(f"  ✅ 最佳参数: {grid_search.best_params_}")
        print(f"  📊 最佳得分: {grid_search.best_score_:.3f}")

        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

    def create_gradient_boosting(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """创建Gradient Boosting模型"""
        print("📈 优化Gradient Boosting...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', 0.8]
        }

        gb = GradientBoostingClassifier(random_state=42)

        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            gb, param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        print(f"  ✅ 最佳参数: {grid_search.best_params_}")
        print(f"  📊 最佳得分: {grid_search.best_score_:.3f}")

        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

class AdvancedEnsemble:
    """
    高级集成学习
    """

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.meta_model = None

    def create_stacking_ensemble(self, base_models: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """创建Stacking集成"""
        print("🎭 创建Stacking集成...")

        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression

        estimators = [(name, model['model']) for name, model in base_models.items()]

        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=TimeSeriesSplit(n_splits=3),
            n_jobs=-1
        )

        stacking_clf.fit(X, y)

        # 交叉验证评估
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(stacking_clf, X, y, cv=tscv, scoring='accuracy')

        print(f"  ✅ Stacking准确率: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        return {
            'model': stacking_clf,
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean()
        }

    def create_voting_ensemble(self, base_models: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """创建Voting集成"""
        print("🗳️ 创建Voting集成...")

        from sklearn.ensemble import VotingClassifier

        estimators = [(name, model['model']) for name, model in base_models.items()]

        # 硬投票
        hard_voting = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
        hard_voting.fit(X, y)

        # 软投票
        soft_voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        soft_voting.fit(X, y)

        # 评估
        tscv = TimeSeriesSplit(n_splits=3)
        hard_scores = cross_val_score(hard_voting, X, y, cv=tscv, scoring='accuracy')
        soft_scores = cross_val_score(soft_voting, X, y, cv=tscv, scoring='accuracy')

        print(f"  ✅ 硬投票准确率: {hard_scores.mean():.3f}")
        print(f"  ✅ 软投票准确率: {soft_scores.mean():.3f}")

        best_voting = soft_voting if soft_scores.mean() > hard_scores.mean() else hard_voting
        best_scores = soft_scores if soft_scores.mean() > hard_scores.mean() else hard_scores

        return {
            'model': best_voting,
            'cv_scores': best_scores,
            'mean_score': best_scores.mean(),
            'type': 'soft' if soft_scores.mean() > hard_scores.mean() else 'hard'
        }

class DataQualityEnhancer:
    """
    数据质量增强器
    """

    def __init__(self):
        pass

    def clean_outliers(self, data: pd.DataFrame, method='iqr') -> pd.DataFrame:
        """清理异常值"""
        print(f"🧹 清理异常值 (方法: {method})...")

        df = data.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
                df[col] = df[col].where(z_scores < 3, df[col].median())

        print(f"  ✅ 异常值清理完成")
        return df

    def balance_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """平衡数据集"""
        print("⚖️ 平衡数据集...")

        from sklearn.utils import resample

        # 合并数据
        data = X.copy()
        data['target'] = y

        # 分离各类别
        class_groups = data.groupby('target')

        # 找到最大类别的样本数
        max_size = class_groups.size().max()

        # 对少数类进行上采样
        balanced_groups = []
        for class_label, group in class_groups:
            if len(group) < max_size:
                # 上采样
                upsampled = resample(group,
                                   replace=True,
                                   n_samples=max_size,
                                   random_state=42)
                balanced_groups.append(upsampled)
            else:
                balanced_groups.append(group)

        # 合并平衡后的数据
        balanced_data = pd.concat(balanced_groups, ignore_index=True)

        X_balanced = balanced_data.drop('target', axis=1)
        y_balanced = balanced_data['target']

        print(f"  ✅ 数据平衡完成: {len(X_balanced)}个样本")
        print(f"  📊 类别分布: {y_balanced.value_counts().to_dict()}")

        return X_balanced, y_balanced

def run_comprehensive_optimization():
    """运行综合优化"""
    print("🚀 StockSynergy 模型综合优化")
    print("=" * 60)

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

    # 2. 数据质量增强
    print("\n🔧 数据质量增强...")
    enhancer = DataQualityEnhancer()

    X_clean = enhancer.clean_outliers(X_raw)
    X_balanced, y_balanced = enhancer.balance_dataset(X_clean, y_raw)

    # 3. 高级特征工程
    print("\n⚙️ 高级特征工程...")
    feature_engineer = AdvancedFeatureEngineer()

    # 重新构造带有更多特征的数据
    enhanced_features = feature_engineer.create_technical_features(X_balanced)

    # 特征选择
    best_features = feature_engineer.select_best_features(enhanced_features, y_balanced, n_features=30)
    X_selected = enhanced_features[best_features]

    print(f"  ✅ 最终特征集: {len(best_features)}个特征")

    # 4. 数据分割
    print("\n✂️ 数据分割...")
    split_idx = int(len(X_selected) * 0.8)
    X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
    y_train, y_test = y_balanced[:split_idx], y_balanced[split_idx:]

    # 5. 模型优化
    print("\n🤖 模型优化...")
    optimizer = ModelOptimizer()

    # 优化各个模型
    rf_result = optimizer.optimize_random_forest(X_train, y_train)
    xgb_result = optimizer.optimize_xgboost(X_train, y_train)
    gb_result = optimizer.create_gradient_boosting(X_train, y_train)

    base_models = {
        'random_forest': rf_result,
        'xgboost': xgb_result,
        'gradient_boosting': gb_result
    }

    # 6. 高级集成
    print("\n🎭 高级集成学习...")
    ensemble = AdvancedEnsemble()

    stacking_result = ensemble.create_stacking_ensemble(base_models, X_train, y_train)
    voting_result = ensemble.create_voting_ensemble(base_models, X_train, y_train)

    # 7. 最终评估
    print("\n📊 最终模型评估...")

    all_models = {
        'Random Forest': rf_result,
        'XGBoost': xgb_result,
        'Gradient Boosting': gb_result,
        'Stacking Ensemble': stacking_result,
        'Voting Ensemble': voting_result
    }

    best_model_name = max(all_models.keys(), key=lambda x: all_models[x]['mean_score'])
    best_model = all_models[best_model_name]

    # 测试集评估
    test_predictions = best_model['model'].predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)

    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"📊 交叉验证得分: {best_model['mean_score']:.3f}")
    print(f"📊 测试集准确率: {test_accuracy:.3f}")

    # 8. 生成优化报告
    print("\n📋 生成优化报告...")

    report_lines = [
        "=" * 80,
        "StockSynergy 模型优化报告",
        "=" * 80,
        f"优化完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "📊 数据优化结果:",
        f"- 原始样本: {len(X_raw):,}",
        f"- 平衡后样本: {len(X_balanced):,}",
        f"- 特征工程: {len(X_raw.columns)} → {len(best_features)}",
        "",
        "🤖 模型性能对比:",
        "-" * 60,
        f"{'模型':<20} {'CV得分':<12} {'测试准确率':<12}",
        "-" * 60
    ]

    for name, result in all_models.items():
        if name == best_model_name:
            test_acc = test_accuracy
        else:
            try:
                test_pred = result['model'].predict(X_test)
                test_acc = np.mean(test_pred == y_test)
            except:
                test_acc = 0.0

        report_lines.append(f"{name:<20} {result['mean_score']:<12.3f} {test_acc:<12.3f}")

    report_lines.extend([
        "-" * 60,
        "",
        f"🏆 最佳模型: {best_model_name}",
        f"📈 性能提升: {test_accuracy:.1%} (目标: 65%+)",
        f"🎯 优化状态: {'✅ 达标' if test_accuracy >= 0.65 else '⚠️ 需进一步优化'}",
        "",
        "🔧 关键优化点:",
        "• 数据平衡和异常值处理",
        "• 高级技术指标特征工程",
        "• 多模型集成学习",
        "• 超参数网格搜索优化",
        "",
        "=" * 80
    ])

    # 保存报告
    report_content = "\n".join(report_lines)
    report_file = f"model_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"📄 优化报告已保存: {report_file}")

    # 9. 保存最佳模型
    model_file = f"best_model_{best_model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pkl"

    model_package = {
        'model': best_model['model'],
        'feature_names': best_features,
        'model_name': best_model_name,
        'performance': {
            'cv_score': best_model['mean_score'],
            'test_accuracy': test_accuracy
        },
        'optimization_date': datetime.now().isoformat()
    }

    with open(model_file, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"💾 最佳模型已保存: {model_file}")

    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'test_accuracy': test_accuracy,
        'feature_names': best_features
    }

if __name__ == "__main__":
    try:
        results = run_comprehensive_optimization()

        print(f"\n🎉 模型优化完成！")
        print(f"🏆 最佳模型: {results['best_model_name']}")
        print(f"📊 测试准确率: {results['test_accuracy']:.1%}")

        if results['test_accuracy'] >= 0.65:
            print("✅ 成功达到65%+目标准确率！")
        else:
            print("⚠️ 未达到目标，建议:")
            print("  • 收集更多历史数据")
            print("  • 加入更多外部特征")
            print("  • 尝试深度学习模型")

    except Exception as e:
        print(f"❌ 优化失败: {str(e)}")
        import traceback
        traceback.print_exc()