#!/usr/bin/env python3
"""
历史数据收集脚本 - FYP核心要求

快速收集2020-2024年S&P 500真实历史数据用于模型训练
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    """主要的数据收集流程"""
    print("🎓 StockSynergy FYP - 历史数据收集")
    print("=" * 50)

    # 配置
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    prediction_days = 5

    # S&P 500主要股票（精选30只）
    stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V",
        "PG", "HD", "UNH", "DIS", "MA", "BAC", "WMT", "PFE", "XOM", "KO",
        "MRK", "CVX", "PEP", "ABBV", "TMO", "COST", "NKE", "MCD", "DHR", "VZ"
    ]

    print(f"📅 收集时间: {start_date} 到 {end_date}")
    print(f"📈 股票数量: {len(stocks)}")

    # 1. 收集股票数据
    print(f"\n📥 下载股票数据...")
    all_data = []
    success_count = 0

    for i, symbol in enumerate(stocks, 1):
        print(f"  {i}/{len(stocks)} {symbol}...", end="")
        try:
            # 下载数据
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if len(data) < 100:  # 确保有足够数据
                print(" ❌ 数据不足")
                continue

            # 计算技术指标
            data['Returns'] = data['Close'].pct_change()
            data['SMA_5'] = data['Close'].rolling(5).mean()
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['RSI'] = calculate_rsi(data['Close'])
            data['Volume_MA'] = data['Volume'].rolling(20).mean()

            # 生成标签（未来5天收益率分类）
            data['Future_Return'] = data['Close'].shift(-prediction_days) / data['Close'] - 1
            data['Label'] = pd.cut(data['Future_Return'],
                                 bins=[-np.inf, -0.02, 0.02, np.inf],
                                 labels=[0, 1, 2])  # 0=下跌, 1=持平, 2=上涨

            # 添加股票标识
            data['Symbol'] = symbol
            data = data.dropna()

            all_data.append(data)
            success_count += 1
            print(f" ✅ {len(data)}行")

        except Exception as e:
            print(f" ❌ {str(e)[:30]}...")

    if not all_data:
        print("❌ 没有成功收集到数据")
        return False

    # 2. 合并数据
    print(f"\n⚙️ 合并数据...")
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"  📊 总样本: {len(combined_data):,}")
    print(f"  📈 成功股票: {success_count}/{len(stocks)}")

    # 3. 下载S&P 500基准
    print(f"\n📈 下载S&P 500基准...")
    try:
        sp500 = yf.Ticker("^GSPC")
        sp500_data = sp500.history(start=start_date, end=end_date)
        sp500_data['Daily_Return'] = sp500_data['Close'].pct_change()
        sp500_data['Cumulative_Return'] = (1 + sp500_data['Daily_Return']).cumprod() - 1

        total_return = sp500_data['Cumulative_Return'].iloc[-1]
        print(f"  ✅ S&P 500期间收益: {total_return:.2%}")

    except Exception as e:
        print(f"  ❌ S&P 500数据失败: {str(e)}")
        sp500_data = None

    # 4. 准备训练数据
    print(f"\n🔧 准备训练数据...")

    # 选择特征列
    feature_cols = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'Volume_MA']
    available_features = [col for col in feature_cols if col in combined_data.columns]

    features = combined_data[available_features].fillna(0)
    labels = combined_data['Label'].dropna()

    # 确保特征和标签长度一致
    min_length = min(len(features), len(labels))
    features = features.iloc[:min_length]
    labels = labels.iloc[:min_length]

    # 准备元数据（包含日期）
    metadata = combined_data[['Symbol']].iloc[:min_length].copy()
    metadata['date'] = combined_data.index[:min_length]  # 添加日期列

    print(f"  📊 特征维度: {features.shape}")
    print(f"  🎯 标签分布: {labels.value_counts().to_dict()}")

    # 5. 保存数据集
    print(f"\n💾 保存数据集...")

    dataset = {
        'features': features,
        'labels': labels,
        'metadata': metadata,  # 现在包含Symbol和date
        'sp500_benchmark': sp500_data,
        'dataset_stats': {
            'total_samples': len(features),
            'total_symbols': success_count,
            'feature_count': len(available_features),
            'label_distribution': labels.value_counts().to_dict()
        }
    }

    # 创建目录并保存
    os.makedirs("data/historical", exist_ok=True)
    filepath = f"data/historical/complete_dataset_{start_date}_{end_date}.pkl"

    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)

    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  ✅ 保存成功: {filepath}")
    print(f"  📁 文件大小: {file_size:.1f} MB")

    # 6. 总结
    print(f"\n🎉 历史数据收集完成！")
    print(f"📊 样本总数: {len(features):,}")
    print(f"📈 股票数量: {success_count}")
    print(f"🔧 特征数量: {len(available_features)}")
    print(f"🚀 可运行训练: python simple_historical_training.py")

    return True

def calculate_rsi(prices, window=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)