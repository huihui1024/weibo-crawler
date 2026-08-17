"""
基本面数据源测试
测试财报、北向资金、龙虎榜数据获取
"""
import sys
sys.path.insert(0, '/root/weibo_crawler')

from app.market.fundamental_data import get_fundamental_client, FundamentalDataClient
from app.selector.multi_factor_model import MultiFactorModel, prepare_stock_data_for_scoring


def test_financial_report():
    """测试财报数据获取"""
    print("\n" + "=" * 60)
    print("测试财报数据获取")
    print("=" * 60)

    client = get_fundamental_client()

    test_codes = ['300308', '600519', '000001']

    for code in test_codes:
        print(f"\n获取 {code} 财报数据...")
        try:
            report = client.get_financial_report(code)
            if report:
                print(f"  股票名称: {report.name}")
                print(f"  报告期: {report.report_date}")
                print(f"  营业收入: {report.revenue:.2f}亿")
                print(f"  营收同比: {report.revenue_yoy:.1f}%")
                print(f"  净利润: {report.net_profit:.2f}亿")
                print(f"  利润同比: {report.net_profit_yoy:.1f}%")
                print(f"  ROE: {report.roe:.2f}%")
                print(f"  毛利率: {report.gross_margin:.2f}%")
                print(f"  净利率: {report.net_margin:.2f}%")
                print(f"  资产负债率: {report.asset_liability_ratio:.2f}%")
                print(f"  每股收益: {report.eps:.2f}")
                print(f"  每股净资产: {report.bvps:.2f}")
                print(f"  财务质量评分: {report.quality_score:.1f}")
                print(f"  成长性评分: {report.growth_score:.1f}")
                print(f"  估值评分: {report.valuation_score:.1f}")
                print(f"  综合评分: {report.total_score:.1f}")
            else:
                print(f"  获取失败")
        except Exception as e:
            print(f"  错误: {e}")


def test_north_bound():
    """测试北向资金数据获取"""
    print("\n" + "=" * 60)
    print("测试北向资金数据获取")
    print("=" * 60)

    client = get_fundamental_client()

    test_codes = ['300308', '600519', '000858']

    for code in test_codes:
        print(f"\n获取 {code} 北向资金数据...")
        try:
            flow = client.get_north_bound_flow(code)
            if flow:
                print(f"  股票名称: {flow.name}")
                print(f"  交易日期: {flow.trade_date}")
                print(f"  持股比例: {flow.hold_ratio:.2f}%")
                print(f"  持股市值: {flow.hold_value:.2f}亿")
                print(f"  5日持股变化: {flow.hold_change_5d:.2f}%")
                print(f"  5日净买入: {flow.net_buy_5d:.2f}亿")
                print(f"  连续买入天数: {flow.continuous_buy_days}")
                print(f"  资金流向评分: {flow.flow_score:.1f}")
            else:
                print(f"  获取失败")
        except Exception as e:
            print(f"  错误: {e}")

    # 测试北向资金整体流向
    print("\n获取北向资金整体流向...")
    try:
        market_flow = client.get_north_bound_market_flow()
        if market_flow:
            print(f"  日期: {market_flow.get('date', 'N/A')}")
            print(f"  沪股通净买入: {market_flow.get('sh_net_buy', 0):.2f}亿")
            print(f"  深股通净买入: {market_flow.get('sz_net_buy', 0):.2f}亿")
            print(f"  合计净买入: {market_flow.get('total_net_buy', 0):.2f}亿")
    except Exception as e:
        print(f"  错误: {e}")


def test_dragon_tiger():
    """测试龙虎榜数据获取"""
    print("\n" + "=" * 60)
    print("测试龙虎榜数据获取")
    print("=" * 60)

    client = get_fundamental_client()

    print("\n获取近期龙虎榜数据...")
    try:
        dragons = client.get_dragon_tiger_list(days=7)
        print(f"  近7日龙虎榜股票数: {len(dragons)}")

        # 显示TOP 10
        sorted_dragons = sorted(dragons, key=lambda x: x.net_buy, reverse=True)[:10]
        for i, d in enumerate(sorted_dragons, 1):
            print(f"  {i}. {d.code} {d.name}: 净买入 {d.net_buy:.0f}万, 涨跌幅 {d.change_pct:.2f}%, 评分 {d.dragon_score:.1f}")

    except Exception as e:
        print(f"  错误: {e}")

    # 测试游资活跃股票
    print("\n获取游资活跃股票...")
    try:
        hot_stocks = client.get_hot_money_stocks(days=7)
        for i, stock in enumerate(hot_stocks[:10], 1):
            print(f"  {i}. {stock['code']} {stock['name']}: 净买入 {stock['net_buy']:.0f}万")
    except Exception as e:
        print(f"  错误: {e}")


def test_comprehensive_fundamental():
    """测试综合基本面数据获取"""
    print("\n" + "=" * 60)
    print("测试综合基本面数据获取")
    print("=" * 60)

    client = get_fundamental_client()

    test_codes = ['300308', '600519']

    for code in test_codes:
        print(f"\n获取 {code} 综合基本面数据...")
        try:
            data = client.get_stock_fundamental(code)

            print(f"  === 财务数据 ===")
            if data.get('financial'):
                fin = data['financial']
                print(f"    报告期: {fin.get('report_date', 'N/A')}")
                print(f"    ROE: {fin.get('roe', 0):.2f}%")
                print(f"    营收同比: {fin.get('revenue_yoy', 0):.1f}%")
                print(f"    利润同比: {fin.get('net_profit_yoy', 0):.1f}%")
                print(f"    财务质量评分: {fin.get('quality_score', 0):.1f}")
                print(f"    成长性评分: {fin.get('growth_score', 0):.1f}")
                print(f"    估值评分: {fin.get('valuation_score', 0):.1f}")
            else:
                print(f"    无财务数据")

            print(f"  === 北向资金 ===")
            if data.get('north_bound'):
                north = data['north_bound']
                print(f"    持股比例: {north.get('hold_ratio', 0):.2f}%")
                print(f"    5日变化: {north.get('hold_change_5d', 0):.2f}%")
                print(f"    资金流向评分: {north.get('flow_score', 0):.1f}")
            else:
                print(f"    无北向资金数据")

            print(f"  === 龙虎榜 ===")
            if data.get('dragon_tiger'):
                dragons = data['dragon_tiger']
                print(f"    近期上榜次数: {len(dragons)}")
                for d in dragons[:3]:
                    print(f"      {d['trade_date']}: 净买入 {d['net_buy']:.0f}万, 评分 {d['dragon_score']:.1f}")
            else:
                print(f"    近期未上榜")

            print(f"  === 综合评分 ===")
            print(f"    基本面综合评分: {data.get('fundamental_score', 0):.1f}")
            print(f"    数据完整性: {data.get('data_completeness', 0):.0%}")

        except Exception as e:
            print(f"  错误: {e}")


def test_multi_factor_with_fundamental():
    """测试带基本面因子的多因子模型"""
    print("\n" + "=" * 60)
    print("测试多因子模型（含基本面因子）")
    print("=" * 60)

    # 创建启用基本面的模型
    model = MultiFactorModel(use_fundamental=True)

    print(f"\n因子列表 ({len(model.factors)} 个):")
    for f in model.factors:
        print(f"  - {f.name}: 权重 {f.weight:.0%}")

    # 模拟数据测试
    test_data = {
        'close': 100,
        'ma5': 98,
        'ma10': 96,
        'ma20': 94,
        'ma60': 90,
        'ma5_slope': 2.5,
        'change_5d': 8,
        'rsi': 55,
        'macd': 0.5,
        'macd_signal': 0.3,
        'volume_change_pct': 60,
        'turnover_rate': 5,
        'change_pct': 2,
        'main_net_inflow_pct': 3,
        'big_order_inflow_pct': 2,
        'atr_pct': 3.5,
        'volatility': 28,
        'market_sentiment': 'bull',
        'is_hot_sector': True,
        # 基本面数据
        'roe': 18.5,
        'gross_margin': 35.2,
        'net_margin': 12.8,
        'asset_liability_ratio': 45,
        'current_ratio': 1.8,
        'operating_cash_flow': 5.2,
        'revenue_yoy': 25.5,
        'net_profit_yoy': 35.2,
        'financial_quality_score': 72,
        'financial_growth_score': 78,
        'financial_valuation_score': 65,
        'north_hold_ratio': 3.5,
        'north_hold_change_5d': 2.1,
        'north_net_buy_5d': 0.8,
        'north_bound_score': 68,
        'on_dragon_tiger_list': True,
        'dragon_net_buy': 3500,
        'dragon_tiger_score': 72,
    }

    score = model.score_stock('300308', '中际旭创', test_data)

    print(f"\n评分结果:")
    print(f"  总分: {score.total_score:.1f}")
    print(f"  推荐等级: {score.recommendation}")
    print(f"  置信度: {score.confidence:.0%}")

    print(f"\n因子评分详情:")
    for fs in score.factor_scores:
        print(f"  {fs.factor_name}: {fs.normalized_score:.1f}分 (权重{fs.weight:.0%})")


def test_cache_functionality():
    """测试缓存功能"""
    print("\n" + "=" * 60)
    print("测试缓存功能")
    print("=" * 60)

    client = get_fundamental_client(use_cache=True)

    import time

    # 第一次获取（无缓存）
    print("\n第一次获取财报数据...")
    start = time.time()
    report1 = client.get_financial_report('300308')
    time1 = time.time() - start
    print(f"  耗时: {time1:.2f}秒")

    # 第二次获取（有缓存）
    print("\n第二次获取财报数据（应该使用缓存）...")
    start = time.time()
    report2 = client.get_financial_report('300308')
    time2 = time.time() - start
    print(f"  耗时: {time2:.2f}秒")

    if time2 < time1 * 0.5:
        print("  缓存生效！第二次获取明显更快")
    else:
        print("  缓存可能未生效")


def main():
    """主测试函数"""
    print("=" * 60)
    print("基本面数据源测试")
    print("=" * 60)

    # 测试财报数据
    test_financial_report()

    # 测试北向资金
    test_north_bound()

    # 测试龙虎榜
    test_dragon_tiger()

    # 测试综合基本面
    test_comprehensive_fundamental()

    # 测试多因子模型
    test_multi_factor_with_fundamental()

    # 测试缓存功能
    test_cache_functionality()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
