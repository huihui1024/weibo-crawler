"""
风格漂移/调仓时序追踪测试
"""
import sys
sys.path.insert(0, '/root/weibo_crawler')

from datetime import date, timedelta
from app.selector.style_drift_tracker import (
    StyleDriftTracker, PositionSnapshot, StyleMetrics,
    RebalanceSignalGenerator, StyleDriftReporter, get_style_tracker
)


def test_position_snapshot():
    """测试持仓快照保存和读取"""
    print("\n" + "=" * 60)
    print("测试持仓快照")
    print("=" * 60)

    tracker = get_style_tracker('/root/weibo_crawler/data/weibo.db')

    # 创建持仓快照
    snapshot = PositionSnapshot(
        date=date.today(),
        positions={
            '300308': 15.0,  # 中际旭创
            '600519': 12.0,  # 贵州茅台
            '000858': 8.0,   # 五粮液
            '002475': 6.0,   # 立讯精密
            '300750': 5.0,   # 宁德时代
        },
        sectors={
            '科技': 21.0,
            '消费': 20.0,
            '新能源': 5.0,
        },
        total_value=100.0,
        cash_ratio=46.0
    )

    # 保存快照
    tracker.save_position_snapshot('测试博主A', snapshot)
    print(f"已保存持仓快照: {snapshot.date}")

    # 读取历史
    history = tracker.get_position_history('测试博主A', days=7)
    print(f"历史快照数量: {len(history)}")
    for h in history:
        print(f"  {h.date}: {len(h.positions)}只股票, 现金{h.cash_ratio:.0f}%")


def test_style_metrics():
    """测试风格指标计算"""
    print("\n" + "=" * 60)
    print("测试风格指标计算")
    print("=" * 60)

    tracker = get_style_tracker('/root/weibo_crawler/data/weibo.db')

    snapshot = PositionSnapshot(
        date=date.today(),
        positions={
            '300308': 15.0,
            '600519': 12.0,
            '000858': 8.0,
            '002475': 6.0,
            '300750': 5.0,
        },
        sectors={'科技': 21.0, '消费': 20.0},
        total_value=100.0
    )

    # 模拟股票信息
    stock_info = {
        '300308': {'market_cap': 800, 'pe': 35, 'roe': 18, 'sector': '光模块'},
        '600519': {'market_cap': 2000, 'pe': 25, 'roe': 30, 'sector': '白酒'},
        '000858': {'market_cap': 500, 'pe': 22, 'roe': 25, 'sector': '白酒'},
        '002475': {'market_cap': 300, 'pe': 28, 'roe': 15, 'sector': '消费电子'},
        '300750': {'market_cap': 900, 'pe': 40, 'roe': 12, 'sector': '锂电池'},
    }

    # 计算风格指标
    metrics = tracker.calculate_style_metrics(snapshot, stock_info)

    print(f"规模风格:")
    print(f"  大盘: {metrics.large_cap_ratio:.0%}")
    print(f"  中盘: {metrics.mid_cap_ratio:.0%}")
    print(f"  小盘: {metrics.small_cap_ratio:.0%}")

    print(f"投资风格:")
    print(f"  价值: {metrics.value_ratio:.0%}")
    print(f"  成长: {metrics.growth_ratio:.0%}")
    print(f"  混合: {metrics.blend_ratio:.0%}")

    print(f"行业风格:")
    print(f"  科技: {metrics.tech_ratio:.0%}")
    print(f"  消费: {metrics.consume_ratio:.0%}")
    print(f"  金融: {metrics.finance_ratio:.0%}")
    print(f"  周期: {metrics.cycle_ratio:.0%}")

    print(f"持仓集中度: {metrics.concentration:.0%}")

    # 保存风格指标
    tracker.save_style_metrics('测试博主A', metrics)
    print("已保存风格指标")


def test_style_drift_detection():
    """测试风格漂移检测"""
    print("\n" + "=" * 60)
    print("测试风格漂移检测")
    print("=" * 60)

    tracker = get_style_tracker('/root/weibo_crawler/data/weibo.db')

    # 创建两期风格指标（模拟风格漂移）
    metrics1 = StyleMetrics(
        date=date.today() - timedelta(days=7),
        large_cap_ratio=0.3, mid_cap_ratio=0.4, small_cap_ratio=0.3,
        value_ratio=0.2, growth_ratio=0.5, blend_ratio=0.3,
        tech_ratio=0.4, consume_ratio=0.3, finance_ratio=0.1, cycle_ratio=0.1,
        concentration=0.6
    )

    metrics2 = StyleMetrics(
        date=date.today(),
        large_cap_ratio=0.5, mid_cap_ratio=0.3, small_cap_ratio=0.2,  # 大盘增加
        value_ratio=0.3, growth_ratio=0.4, blend_ratio=0.3,
        tech_ratio=0.2, consume_ratio=0.4, finance_ratio=0.2, cycle_ratio=0.1,  # 科技减少，消费增加
        concentration=0.7
    )

    tracker.save_style_metrics('测试博主B', metrics1)
    tracker.save_style_metrics('测试博主B', metrics2)

    # 检测风格漂移
    drift_events = tracker.detect_style_drift('测试博主B', lookback_days=30)

    print(f"检测到 {len(drift_events)} 个风格漂移事件:")
    for event in drift_events:
        print(f"  [{event.date}] {event.drift_type}: {event.description}")
        print(f"    变化: {event.from_value} -> {event.to_value}")
        print(f"    幅度: {event.magnitude:.0%}")


def test_position_adjustment_tracking():
    """测试调仓追踪"""
    print("\n" + "=" * 60)
    print("测试调仓追踪")
    print("=" * 60)

    tracker = get_style_tracker('/root/weibo_crawler/data/weibo.db')

    # 创建两期持仓快照
    previous = PositionSnapshot(
        date=date.today() - timedelta(days=1),
        positions={'300308': 15.0, '600519': 10.0, '000858': 8.0},
        sectors={'科技': 15.0, '消费': 18.0},
        total_value=100.0
    )

    current = PositionSnapshot(
        date=date.today(),
        positions={'300308': 12.0, '600519': 15.0, '002475': 5.0},  # 减持300308，加仓600519，新建002475，清仓000858
        sectors={'科技': 17.0, '消费': 15.0},
        total_value=100.0
    )

    # 追踪调仓
    adjustments = tracker.track_position_adjustments('测试博主C', current, previous)

    print(f"检测到 {len(adjustments)} 条调仓记录:")
    for adj in adjustments:
        print(f"  {adj.stock_code}: {adj.action} {adj.weight_change:.1f}% - {adj.reason}")

    # 获取调仓历史
    history = tracker.get_adjustment_history('测试博主C', days=7)
    print(f"\n调仓历史记录数: {len(history)}")


def test_rebalance_signal():
    """测试调仓信号生成"""
    print("\n" + "=" * 60)
    print("测试调仓信号生成")
    print("=" * 60)

    tracker = get_style_tracker('/root/weibo_crawler/data/weibo.db')
    generator = RebalanceSignalGenerator(tracker)

    # 当前持仓
    current_positions = {
        '300308': 15.0,
        '600519': 25.0,  # 权重过高
        '000858': 8.0,
    }

    # 模拟市场数据
    market_data = {
        '300308': {'name': '中际旭创', 'trend': 'up', 'change_pct': 3.5, 'change_5d': 8},
        '600519': {'name': '贵州茅台', 'trend': 'neutral', 'change_pct': 0.5, 'change_5d': 2},
        '000858': {'name': '五粮液', 'trend': 'down', 'change_pct': -4.0, 'change_5d': -12},  # 连续下跌
    }

    # 生成调仓信号
    signals = generator.generate_signals('测试博主D', current_positions, market_data)

    print(f"生成 {len(signals)} 个调仓信号:")
    for signal in signals:
        print(f"\n  信号类型: {signal.signal_type}")
        print(f"  置信度: {signal.confidence:.0%}")
        print(f"  原因: {signal.reason}")
        if signal.suggested_actions:
            print(f"  建议操作:")
            for action in signal.suggested_actions:
                print(f"    - {action.stock_code}: {action.action} {action.weight_change:.1f}% ({action.reason})")


def test_report_generation():
    """测试报告生成"""
    print("\n" + "=" * 60)
    print("测试报告生成")
    print("=" * 60)

    tracker = get_style_tracker('/root/weibo_crawler/data/weibo.db')
    reporter = StyleDriftReporter(tracker)

    # 生成报告
    report = reporter.generate_report('测试博主A', days=30)
    print(report)


def test_turnover_rate():
    """测试换手率计算"""
    print("\n" + "=" * 60)
    print("测试换手率计算")
    print("=" * 60)

    tracker = get_style_tracker('/root/weibo_crawler/data/weibo.db')

    # 计算换手率
    turnover = tracker.calculate_turnover_rate('测试博主C', days=30)
    print(f"近30天换手率: {turnover:.1f}%")


def main():
    """主测试函数"""
    print("=" * 60)
    print("风格漂移/调仓时序追踪测试")
    print("=" * 60)

    # 测试持仓快照
    test_position_snapshot()

    # 测试风格指标计算
    test_style_metrics()

    # 测试风格漂移检测
    test_style_drift_detection()

    # 测试调仓追踪
    test_position_adjustment_tracking()

    # 测试调仓信号生成
    test_rebalance_signal()

    # 测试换手率计算
    test_turnover_rate()

    # 测试报告生成
    test_report_generation()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
