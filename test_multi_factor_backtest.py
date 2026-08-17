"""
多因子模型和回测框架测试脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from app.selector.multi_factor_model import MultiFactorModel, StockScore
from app.selector.backtest import (
    BacktestEngine, BacktestRunner, BacktestResult,
    PredictionBasedStrategy, generate_backtest_report
)


def test_multi_factor_model():
    """测试多因子模型"""
    print("\n" + "="*60)
    print("测试多因子模型")
    print("="*60)

    model = MultiFactorModel()

    # 模拟数据测试
    test_cases = [
        # 强势上涨股票
        {
            'code': '300308',
            'name': '中际旭创',
            'data': {
                'close': 150,
                'ma5': 148,
                'ma10': 145,
                'ma20': 140,
                'ma60': 130,
                'ma5_slope': 3.5,
                'change_5d': 12,
                'rsi': 65,
                'macd': 1.2,
                'macd_signal': 0.8,
                'volume_change_pct': 80,
                'turnover_rate': 6,
                'change_pct': 3,
                'main_net_inflow_pct': 5,
                'big_order_inflow_pct': 3,
                'atr_pct': 3.5,
                'volatility': 28,
                'market_sentiment': 'bull',
                'is_hot_sector': True,
            }
        },
        # 震荡股票
        {
            'code': '600519',
            'name': '贵州茅台',
            'data': {
                'close': 1800,
                'ma5': 1795,
                'ma10': 1800,
                'ma20': 1805,
                'ma60': 1750,
                'ma5_slope': 0.5,
                'change_5d': 2,
                'rsi': 50,
                'macd': 0.1,
                'macd_signal': 0.2,
                'volume_change_pct': 20,
                'turnover_rate': 0.5,
                'change_pct': 0.5,
                'main_net_inflow_pct': 0,
                'big_order_inflow_pct': 0,
                'atr_pct': 2,
                'volatility': 20,
                'market_sentiment': 'neutral',
                'is_hot_sector': False,
            }
        },
        # 弱势下跌股票
        {
            'code': '000001',
            'name': '平安银行',
            'data': {
                'close': 10,
                'ma5': 10.5,
                'ma10': 11,
                'ma20': 11.5,
                'ma60': 12,
                'ma5_slope': -2,
                'change_5d': -8,
                'rsi': 35,
                'macd': -0.5,
                'macd_signal': -0.3,
                'volume_change_pct': -30,
                'turnover_rate': 1,
                'change_pct': -2,
                'main_net_inflow_pct': -5,
                'big_order_inflow_pct': -3,
                'atr_pct': 4,
                'volatility': 35,
                'market_sentiment': 'bear',
                'is_hot_sector': False,
            }
        }
    ]

    # 批量评分
    scores = model.score_stocks(test_cases)

    print("\n评分结果:")
    for score in scores:
        print(f"\n{score.rank}. {score.code} {score.name}")
        print(f"   总分: {score.total_score:.1f}")
        print(f"   推荐等级: {score.recommendation}")
        print(f"   置信度: {score.confidence:.0%}")
        print("   因子得分:")
        for fs in score.factor_scores:
            print(f"     - {fs.factor_name}: {fs.normalized_score:.1f}分 (权重{fs.weight:.0%})")


def test_backtest_engine():
    """测试回测引擎"""
    print("\n" + "="*60)
    print("测试回测引擎")
    print("="*60)

    engine = BacktestEngine(initial_capital=1000000)

    # 模拟交易
    trade_date = date.today()

    # 买入
    success = engine.buy(
        code='300308',
        name='中际旭创',
        trade_date=trade_date,
        price=150,
        amount=100000,
        reason="多因子评分85分"
    )
    print(f"买入成功: {success}")
    print(f"剩余现金: {engine.cash:.2f}")

    # 更新持仓
    engine.update_positions(trade_date, {'300308': 155})
    print(f"持仓市值: {engine.positions['300308'].market_value:.2f}")
    print(f"持仓盈亏: {engine.positions['300308'].profit_pct:.2f}%")

    # 卖出
    success = engine.sell(
        code='300308',
        trade_date=trade_date + timedelta(days=5),
        price=160,
        reason="止盈卖出"
    )
    print(f"卖出成功: {success}")
    print(f"最终现金: {engine.cash:.2f}")

    # 计算收益
    total_value = engine.get_total_value()
    print(f"总资产: {total_value:.2f}")
    print(f"收益率: {(total_value - 1000000) / 1000000 * 100:.2f}%")


def test_backtest_runner():
    """测试回测运行器"""
    print("\n" + "="*60)
    print("测试回测运行器")
    print("="*60)

    runner = BacktestRunner()

    # 运行空回测（没有历史数据）
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    strategy = PredictionBasedStrategy(score_threshold=60)

    result = runner.run_backtest(strategy, start_date, end_date)

    print(f"\n回测结果:")
    print(f"  开始日期: {result.start_date}")
    print(f"  结束日期: {result.end_date}")
    print(f"  初始资金: {result.initial_capital:,.0f}")
    print(f"  最终资金: {result.final_capital:,.0f}")
    print(f"  总收益率: {result.total_return:+.2f}%")
    print(f"  最大回撤: {result.max_drawdown:.2f}%")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  交易次数: {result.total_trades}")
    print(f"  胜率: {result.win_rate:.1f}%")


def test_backtest_report():
    """测试回测报告生成"""
    print("\n" + "="*60)
    print("测试回测报告生成")
    print("="*60)

    # 创建模拟回测结果
    result = BacktestResult(
        start_date=date.today() - timedelta(days=30),
        end_date=date.today(),
        initial_capital=1000000,
        final_capital=1050000,
        total_return=5.0,
        annual_return=60.0,
        benchmark_return=3.0,
        max_drawdown=2.5,
        volatility=15.0,
        sharpe_ratio=4.0,
        sortino_ratio=5.0,
        total_trades=20,
        win_trades=12,
        lose_trades=8,
        win_rate=60.0,
        avg_profit_trade=5000,
        avg_loss_trade=3000,
        profit_factor=1.67,
        avg_hold_days=5,
        max_hold_days=10
    )

    report = generate_backtest_report(result)
    print(report)


if __name__ == "__main__":
    print("="*60)
    print("多因子模型和回测框架测试")
    print("="*60)

    # 测试多因子模型
    test_multi_factor_model()

    # 测试回测引擎
    test_backtest_engine()

    # 测试回测运行器
    test_backtest_runner()

    # 测试回测报告
    test_backtest_report()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)