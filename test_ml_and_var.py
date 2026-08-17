"""
测试ML预测模型和VaR风险度量模块
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

# 测试ML预测模型
print("=" * 60)
print("测试 ML 预测模型")
print("=" * 60)

from app.selector.ml_predictor import (
    MLPredictor,
    FeatureEngineer,
    generate_ml_prediction_report
)

# 生成测试数据
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

# 模拟股票价格（带趋势的随机游走）
trend = np.linspace(0, 0.3, 200)  # 上涨趋势
noise = np.random.normal(0, 0.02, 200)
returns = 0.001 + noise  # 日均收益0.1%
prices = 100 * np.cumprod(1 + returns)

test_df = pd.DataFrame({
    'date': dates,
    'open': prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
    'high': prices * (1 + np.random.uniform(0, 0.02, 200)),
    'low': prices * (1 + np.random.uniform(-0.02, 0, 200)),
    'close': prices,
    'volume': np.random.randint(1000000, 5000000, 200)
})

# 测试特征工程
print("\n1. 测试特征工程...")
fe = FeatureEngineer()
df_features = fe.extract_features(test_df)
print(f"   提取特征数量: {len(fe.feature_names)}")
print(f"   主要特征: {fe.feature_names[:10]}")

# 测试ML预测器
print("\n2. 测试ML预测器（未训练模式）...")
predictor = MLPredictor(model_type='ensemble')
result = predictor.predict(test_df)

if result:
    print(f"   预测趋势: {result.predicted_trend}")
    print(f"   预测涨跌幅: {result.predicted_change_pct:+.2f}%")
    print(f"   置信度: {result.confidence:.1%}")
    print(f"   预测区间: [{result.predicted_change_range[0]:+.2f}%, {result.predicted_change_range[1]:+.2f}%]")

# 生成ML预测报告
print("\n3. 生成ML预测报告...")
print(generate_ml_prediction_report(result))

# 测试VaR风险度量
print("\n" + "=" * 60)
print("测试 VaR 风险度量")
print("=" * 60)

from app.selector.risk_metrics import (
    VaRCalculator,
    VaRMethod,
    VaRResult,
    PortfolioRiskAnalyzer,
    RiskMonitor,
    calculate_single_stock_var
)

# 计算收益率
returns_series = test_df['close'].pct_change().dropna()

# 测试VaR计算器
print("\n1. 测试VaR计算器...")
calculator = VaRCalculator(confidence_level=0.95)

for method in VaRMethod:
    result = calculator.calculate_var(returns_series, method=method, position_value=100000)
    print(f"\n   {method.value} 方法:")
    print(f"     VaR(95%): {result.var_value:.2f}% ({result.var_absolute:,.2f}元)")
    print(f"     CVaR: {result.cvar:.2f}%")
    print(f"     风险等级: {result.risk_level}")
    print(f"     收益分布: 均值={result.mean_return:.4f}%, 标准差={result.std_return:.4f}%")

# 测试组合风险分析
print("\n2. 测试组合风险分析...")

# 创建多只股票数据
positions = {
    '600519': 50000,  # 贵州茅台
    '000858': 30000,  # 五粮液
    '000333': 20000,  # 美的集团
}

price_data = {}
for i, code in enumerate(positions.keys()):
    np.random.seed(42 + i)
    ret = np.random.normal(0.0005, 0.02 * (1 + i * 0.15), 200)
    prices = 100 * np.cumprod(1 + ret)
    price_data[code] = pd.DataFrame({
        'date': dates,
        'close': prices
    })

analyzer = PortfolioRiskAnalyzer()
report = analyzer.analyze_portfolio(positions, price_data)

print(f"\n   组合价值: {report.portfolio_value:,.2f}元")
print(f"   风险评级: {report.risk_rating} ({report.overall_risk_score:.1f}分)")

if report.portfolio_var:
    print(f"   组合VaR: {report.portfolio_var.var_value:.2f}%")
    print(f"   组合CVaR: {report.portfolio_var.cvar:.2f}%")

print(f"   夏普比率: {report.sharpe_ratio:.4f}")
print(f"   索提诺比率: {report.sortino_ratio:.4f}")
print(f"   最大回撤: {report.max_drawdown:.2f}%")
print(f"   卡玛比率: {report.calmar_ratio:.4f}")

# 压力测试结果
print("\n   压力测试结果:")
for scenario, loss in report.stress_test_results.items():
    print(f"     {scenario}: {loss:,.2f}元 ({loss/report.portfolio_value*100:.2f}%)")

# 风险贡献
print("\n   风险贡献:")
for code, contrib in sorted(report.risk_contribution.items(), key=lambda x: x[1], reverse=True):
    print(f"     {code}: {contrib:.2f}%")

# 生成风险报告
print("\n3. 生成完整风险报告...")
monitor = RiskMonitor()
print(monitor.generate_risk_report(report))

# 测试风险限额检查
print("\n4. 测试风险限额检查...")
if report.portfolio_var:
    limit_check = monitor.check_risk_limits(report.portfolio_var, var_limit=5.0, cvar_limit=7.0)
    print(f"   VaR状态: {limit_check['var_status']}")
    print(f"   CVaR状态: {limit_check['cvar_status']}")
    if limit_check['alerts']:
        for alert in limit_check['alerts']:
            print(f"   ⚠️ {alert['message']}")
    else:
        print("   ✅ 风险在限额范围内")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
