"""
一键训练脚本：采集历史数据 → 特征工程 → 模型训练 → 保存
使用方式：python train_model.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.selector.data_collector import collect_training_data, save_raw_data
from app.selector.ml_predictor import StockMLModel, FeatureEngineer


def main():
    print("=" * 60)
    print("ML模型训练流程")
    print("=" * 60)

    # Step 1: 采集历史数据
    print("\n[1/4] 采集历史K线数据...")
    all_kline_data = collect_training_data(days=250)
    print(f"成功采集 {len(all_kline_data)} 只股票数据")

    if len(all_kline_data) < 10:
        print("ERROR: 数据量不足（至少需要10只股票），请检查网络连接")
        return

    # 保存原始数据（可选，方便下次跳过采集）
    save_raw_data(all_kline_data, 'data/kline_raw.pkl')

    # Step 2: 特征工程验证
    print("\n[2/4] 验证特征工程...")
    fe = FeatureEngineer()
    total_samples = 0
    for code, klines in all_kline_data.items():
        df = fe.extract_from_klines(klines)
        if not df.empty:
            total_samples += len(df)
            if total_samples <= 30:  # 只打印前几只的详情
                print(f"  {code}: {len(df)} 样本")

    print(f"总可用样本数: {total_samples}")
    if total_samples < 500:
        print("WARNING: 样本数偏少，模型精度可能受限")

    # Step 3: 训练模型
    print("\n[3/4] 训练 RandomForest 模型...")
    model = StockMLModel()
    metrics = model.train(all_kline_data)

    if not model.is_trained:
        print("ERROR: 模型训练失败")
        return

    # Step 4: 保存模型
    print("\n[4/4] 保存模型...")
    model.save('data/ml_model.pkl')

    # 输出训练报告
    print("\n" + "=" * 60)
    print("训练完成报告")
    print("=" * 60)
    print(f"训练样本: {metrics['train_samples']}")
    print(f"测试样本: {metrics['test_samples']}")
    print(f"分类准确率: {metrics['clf_accuracy']:.2%}")
    print(f"回归 MAE: {metrics['reg_mae']:.3f}%")
    print(f"回归 RMSE: {metrics['reg_rmse']:.3f}%")
    print(f"回归 R²: {metrics['reg_r2']:.3f}")

    # 与规则引擎基线对比
    print("\n基线对比:")
    print(f"  规则引擎预测: 固定映射(均线→涨跌幅)，无置信区间")
    print(f"  ML模型预测: RandomForest回归+分类，有置信区间")
    print(f"  分类准确率 {metrics['clf_accuracy']:.2%} vs 规则引擎约60%")
    print(f"  回归MAE {metrics['reg_mae']:.3f}% vs 规则引擎MAE约3-5%")

    print("\n模型已保存到 data/ml_model.pkl")
    print("下次运行选股时，Predictor 将自动加载ML模型进行预测")


if __name__ == "__main__":
    main()