#!/usr/bin/env python3
"""
智能选股与复盘系统 - 主运行脚本
"""
import sys
import os
import asyncio
import argparse
from datetime import date, datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.analyzer.stock_analyzer import StockAnalyzer
from app.selector.stock_selector import StockSelector, run_stock_selector, get_stock_pool_by_blogger, update_all_blogger_pools
from app.selector.predictor import Predictor, generate_prediction_report
from app.reviewer.daily_reviewer import DailyReviewer, get_review_history, get_review_detail_by_date, generate_history_report
from app.database import search_weibo_posts, search_weibo_posts_count, get_all_bloggers, get_weibo_posts
import sqlite3


def analyze_blogger(blogger_name: str = "小棉袄"):
    """分析博主策略"""
    print("=" * 70)
    print(f"分析博主: {blogger_name}")
    print("=" * 70)

    conn = sqlite3.connect('data/weibo.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT b.nickname, w.content, w.publish_time, w.weibo_id
        FROM weibo_posts w
        LEFT JOIN bloggers b ON w.blogger_id = b.id
        WHERE b.nickname = ?
        ORDER BY w.crawl_time DESC
    ''', (blogger_name,))

    posts = []
    for row in cursor.fetchall():
        weibo_id = row[3] if row[3] else ''
        # 查找对应的图片
        image_paths = []
        if weibo_id:
            image_dir = os.path.join('data/images', weibo_id)
            if os.path.exists(image_dir):
                for f in os.listdir(image_dir):
                    if f.endswith(('.jpg', '.png', '.jpeg')):
                        image_paths.append(os.path.join(image_dir, f))

        posts.append({
            'content': row[1],
            'publish_time': str(row[2]) if row[2] else '',
            'weibo_id': weibo_id,
            'image_paths': image_paths
        })
    conn.close()

    if not posts:
        print(f"未找到博主 {blogger_name} 的微博数据")
        return None

    analyzer = StockAnalyzer()
    result = analyzer.analyze_posts(posts, blogger_name)

    print(f"\n关注板块: {result.hot_industries}")
    print(f"选股因子: {list(result.selection_factors.keys())}")
    print(f"交易周期: {result.trading_cycle_detail}")

    return result


def _print_risk_banner():
    """打印动态胜率风险横幅"""
    try:
        from app.selector.stock_selector import get_dynamic_win_rate, get_simulation_status
        dyn = get_dynamic_win_rate()
        wr = dyn.get('win_rate', 0) or dyn.get('backtest_win_rate', 64.0)
        signal = dyn.get('signal_strength', 'normal')
        if dyn.get('total_samples', 0) > 0:
            print(f"⚠️  风险提示: 系统近期胜率{wr}%，非确定性预测。单笔仓位≤5%，严格止损。")
        else:
            print(f"⚠️  风险提示: 系统回测胜率{wr}%（无实盘验证），非确定性预测。单笔仓位≤5%，严格止损。")
        if signal == 'suspended':
            print("🔴  荐股信号已暂停：实盘胜率严重偏离回测，请勿跟投！")
        elif signal == 'degraded':
            print("🟡  荐股信号降级：实盘表现弱于回测，建议减仓观望。")

        # P2: 强制实盘模拟警告
        sim = get_simulation_status()
        sim_days = sim.get('days', 0)
        if sim_days < 90:
            print(f"🔴  系统尚未完成3个月实盘模拟验证(当前{sim_days}天/需90天)，所有推荐仅供参考，请勿直接跟投！")
    except Exception:
        print("⚠️  风险提示: 系统非确定性预测。单笔仓位≤5%，严格止损。")
    print("⚠️  建议至少进行3个月实盘模拟验证后再考虑跟投。")


def run_selection(blogger_name: str = None):
    """运行选股"""
    print("\n" + "=" * 70)
    print(f"开始智能选股（博主: {blogger_name or '默认'}）— 博主信号仅作行业方向参考")
    print("=" * 70)
    _print_risk_banner()
    print("=" * 70)

    candidates = run_stock_selector(blogger_name)

    if candidates:
        print(f"\n共筛选出 {len(candidates)} 只候选股票")

        # 生成预测
        print("\n" + "=" * 70)
        print("生成预测")
        print("=" * 70)

        predictor = Predictor()
        codes = [c.code for c in candidates[:10]]
        predictions = predictor.batch_predict(codes)

        if predictions:
            predictor.save_predictions_to_db(predictions)
            print(f"\n已生成并保存 {len(predictions)} 条预测")

            # 打印预测报告
            print(generate_prediction_report(predictions))
    else:
        print("未筛选出候选股票")


def run_review():
    """运行复盘"""
    print("\n" + "=" * 70)
    print("开始每日复盘")
    print("=" * 70)

    reviewer = DailyReviewer()
    summary = reviewer.review_predictions()

    if summary.total_predictions > 0:
        print(f"\n复盘完成")
        print(f"准确率: {summary.accuracy_rate:.1f}%")
        print(f"平均评分: {summary.avg_prediction_score:.1f}分")

        # 打印复盘报告
        print(reviewer.generate_review_report(summary))

        # 保存报告
        report_path = f"data/review_report_{summary.review_date}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(reviewer.generate_review_report(summary))
        print(f"\n复盘报告已保存: {report_path}")
    else:
        print("没有需要复盘的预测记录")


def show_stock_pool(blogger_name: str = None):
    """显示股票池"""
    print("\n" + "=" * 70)
    print(f"当前股票池（博主: {blogger_name or '全部'}）")
    print("=" * 70)
    _print_risk_banner()

    # 显示模拟验证状态
    try:
        from app.selector.stock_selector import get_simulation_status
        sim = get_simulation_status(blogger_name)
        if sim.get('status') == 'in_progress':
            print(f"📋  实盘模拟进度: {sim['days']}/{sim['required_days']}天 (胜率: {sim['win_rate']}%)")
        elif sim.get('status') == 'not_started':
            print(f"📋  实盘模拟: 尚未开始 (需{sim['required_days']}天验证)")
        elif sim.get('status') == '3m_completed':
            print(f"📋  实盘模拟: 已完成{sim['days']}天验证 (胜率: {sim['win_rate']}%)")
    except Exception:
        pass

    pool = get_stock_pool_by_blogger(blogger_name)

    if pool:
        # 按博主分组显示
        if not blogger_name:
            bloggers = {}
            for stock in pool:
                bn = stock.get('blogger_name', '未知')
                if bn not in bloggers:
                    bloggers[bn] = []
                bloggers[bn].append(stock)

            for bn, stocks in bloggers.items():
                print(f"\n【{bn or '全部'}】")
                for i, stock in enumerate(stocks, 1):
                    src = stock.get('selection_source', '市场热点')
                    print(f"{i}. [{src}] {stock['code']} {stock['name']}")
                    print(f"   板块: {stock['sector']} | 价格: {stock['price']} | 涨跌: {stock['change_pct']:+.2f}% | 匹配度: {stock['style_match_score']:.1f}分")
                    print(f"   添加日期: {stock['added_date']}")
                    print(f"   原因: {stock['selection_reasons']}")
                    print()
        else:
            for i, stock in enumerate(pool, 1):
                print(f"{i}. {stock['code']} {stock['name']}")
                print(f"   板块: {stock['sector']} | 价格: {stock['price']} | 涨跌: {stock['change_pct']:+.2f}% | 匹配度: {stock['style_match_score']:.1f}分")
                print(f"   添加日期: {stock['added_date']}")
                print(f"   原因: {stock['selection_reasons']}")
                print()
    else:
        print("股票池为空，请先运行选股")


def show_predictions():
    """显示预测记录"""
    print("\n" + "=" * 70)
    print("当前预测记录")
    print("=" * 70)

    conn = sqlite3.connect('data/weibo.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT stock_code, stock_name, prediction_date, current_price,
               expected_close_change, actual_change, expected_amplitude,
               expected_trend, expected_hold_days, expected_hold_return,
               expected_buy_point, expected_sell_point, selection_reason, status
        FROM predictions
        ORDER BY prediction_date DESC, prediction_time DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()
    conn.close()

    if rows:
        for i, row in enumerate(rows, 1):
            status_icon = "⏳" if row[13] == 'pending' else "✅"
            current_price = row[3] if row[3] else 0
            actual_change = row[5] if row[5] else 0
            actual_str = f"实际: {actual_change:+.2f}%" if row[13] != 'pending' else ""
            amplitude = row[6] if row[6] else 0
            hold_days = row[8] if row[8] else 0
            hold_return = row[9] if row[9] else 0

            print(f"{status_icon} {i}. {row[0]} {row[1]} ({row[2]})")
            print(f"   当前价格: {current_price:.2f}元 | 预期涨跌: {row[4]:+.2f}% | {actual_str}")
            print(f"   预期振幅: {amplitude:.2f}% | 趋势: {row[7]}")
            print(f"   持股周期: {hold_days}天 | 预期收益: {hold_return:+.2f}%")
            print(f"   买入点位: {row[10] or '未设置'}")
            print(f"   卖出点位: {row[11] or '未设置'}")
            print(f"   原因: {row[12] or '无'}")
            print()
    else:
        print("暂无预测记录")


def show_bloggers():
    """显示所有博主"""
    print("\n" + "=" * 70)
    print("博主列表")
    print("=" * 70)

    conn = sqlite3.connect('data/weibo.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT b.nickname, COUNT(w.id) as post_count, MAX(w.crawl_time) as last_update
        FROM bloggers b
        LEFT JOIN weibo_posts w ON b.id = w.blogger_id
        GROUP BY b.id
        ORDER BY post_count DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    if rows:
        for i, row in enumerate(rows, 1):
            print(f"{i}. {row[0]}")
            print(f"   微博数量: {row[1]}")
            print(f"   最后更新: {row[2] or '未知'}")
            print()
    else:
        print("暂无博主数据")


def update_pool(blogger_name: str = None):
    """更新股票池"""
    if blogger_name:
        print(f"\n更新博主 {blogger_name} 的股票池...")
        run_selection(blogger_name)
    else:
        print("\n更新所有博主的股票池...")
        update_all_blogger_pools()


def show_review_history(days: int = 30):
    """显示历史复盘记录"""
    print("\n" + "=" * 70)
    print(f"历史复盘记录（最近 {days} 天）")
    print("=" * 70)

    history = get_review_history(days)

    if not history:
        print(f"最近 {days} 天没有复盘记录")
        return

    print(f"\n共找到 {len(history)} 天的复盘记录\n")

    # 按日期分组显示
    print("-" * 70)
    print(f"{'日期':<12} {'预测数':>6} {'正确数':>6} {'准确率':>8} {'评分':>6} {'涨跌幅偏差':>10} {'振幅偏差':>8}")
    print("-" * 70)

    for h in history:
        print(f"{h['review_date']:<12} {h['total_predictions']:>6} {h['correct_predictions']:>6} "
              f"{h['accuracy_rate']:>7.1f}% {h['avg_prediction_score']:>6.1f} "
              f"{h['avg_change_diff']:>+9.2f}% {h['avg_amplitude_diff']:>+7.2f}%")

    print("-" * 70)

    # 统计汇总
    total_predictions = sum(h['total_predictions'] for h in history)
    total_correct = sum(h['correct_predictions'] for h in history)
    avg_accuracy = sum(h['accuracy_rate'] for h in history) / len(history)
    avg_score = sum(h['avg_prediction_score'] for h in history) / len(history)

    print(f"\n汇总统计:")
    print(f"  总预测数: {total_predictions}")
    print(f"  总正确数: {total_correct}")
    print(f"  平均准确率: {avg_accuracy:.1f}%")
    print(f"  平均评分: {avg_score:.1f}分")


def show_review_detail(review_date: str = None):
    """显示指定日期的复盘详情"""
    if review_date:
        try:
            target_date = datetime.strptime(review_date, '%Y-%m-%d').date()
        except ValueError:
            print(f"日期格式错误，请使用 YYYY-MM-DD 格式")
            return
    else:
        target_date = date.today()

    print("\n" + "=" * 70)
    print(f"复盘详情 - {target_date}")
    print("=" * 70)

    details = get_review_detail_by_date(target_date)

    if not details:
        print(f"没有找到 {target_date} 的复盘详情")
        return

    print(f"\n共 {len(details)} 只股票的复盘记录\n")

    for i, d in enumerate(details, 1):
        status = "✅" if d['trend_match'] else "❌"
        print(f"{status} {i}. {d['stock_code']} {d['stock_name']}")
        print(f"   涨跌幅: 预测 {d['expected_change']:+.2f}% | 实际 {d['actual_change']:+.2f}% | 偏差 {d['change_diff']:+.2f}%")
        print(f"   振幅: 预测 {d['expected_amplitude']:.2f}% | 实际 {d['actual_amplitude']:.2f}% | 偏差 {d['amplitude_diff']:+.2f}%")
        print(f"   趋势: 预测 {d['expected_trend']} | 实际 {d['actual_trend']} | {'匹配' if d['trend_match'] else '不匹配'}")
        print(f"   评分: {d['prediction_score']}分")
        print(f"   差异原因: {d['diff_reason']}")
        print(f"   改进建议: {d['improvement_suggestion']}")
        print()


def show_review_summary(days: int = 30):
    """显示历史复盘汇总报告"""
    print("\n" + "=" * 70)
    print(f"历史复盘汇总报告（最近 {days} 天）")
    print("=" * 70)

    report = generate_history_report(days)
    print(report)


def show_weibo_posts(blogger_name: str = None, limit: int = 20):
    """显示微博内容（按博主分类）"""
    print("\n" + "=" * 70)
    print(f"微博内容（博主: {blogger_name or '全部'}）")
    print("=" * 70)

    conn = sqlite3.connect('data/weibo.db')
    cursor = conn.cursor()

    # 获取博主ID
    if blogger_name:
        cursor.execute("SELECT id, nickname FROM bloggers WHERE nickname = ?", (blogger_name,))
        blogger_row = cursor.fetchone()
        if not blogger_row:
            print(f"未找到博主: {blogger_name}")
            conn.close()
            return
        blogger_id = blogger_row[0]
        blogger_nickname = blogger_row[1]

        cursor.execute("""
            SELECT w.weibo_id, w.content, w.publish_time, w.is_repost, w.repost_content, w.repost_author
            FROM weibo_posts w
            WHERE w.blogger_id = ?
            ORDER BY w.is_pinned DESC, w.publish_time DESC
            LIMIT ?
        """, (blogger_id, limit))
    else:
        # 显示所有博主的微博
        cursor.execute("""
            SELECT b.nickname, w.weibo_id, w.content, w.publish_time, w.is_repost, w.repost_content, w.repost_author
            FROM weibo_posts w
            LEFT JOIN bloggers b ON w.blogger_id = b.id
            ORDER BY w.is_pinned DESC, w.publish_time DESC
            LIMIT ?
        """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    if rows:
        for i, row in enumerate(rows, 1):
            if blogger_name:
                nickname = blogger_nickname
                weibo_id = row[0]
                content = row[1]
                publish_time = row[2]
                is_repost = row[3]
                repost_content = row[4]
                repost_author = row[5]
            else:
                nickname = row[0] or '未知'
                weibo_id = row[1]
                content = row[2]
                publish_time = row[3]
                is_repost = row[4]
                repost_content = row[5]
                repost_author = row[6]

            print(f"\n{i}. 【{nickname}】 {publish_time or '未知时间'}")
            print(f"   微博ID: {weibo_id}")
            # 显示内容（截断长内容）
            if content:
                display_content = content[:200] + "..." if len(content) > 200 else content
                print(f"   内容: {display_content}")
            if is_repost:
                print(f"   [转发] @{repost_author}: {repost_content[:100] if repost_content else ''}...")
    else:
        print("没有找到微博内容")


def search_weibo(keyword: str, blogger_name: str = None, limit: int = 50):
    """搜索微博内容"""
    print("\n" + "=" * 70)
    print(f"微博搜索 - 关键词: '{keyword}'")
    print("=" * 70)

    if not keyword:
        print("请提供搜索关键词")
        return

    conn = sqlite3.connect('data/weibo.db')
    cursor = conn.cursor()

    # 构建搜索模式
    search_pattern = f"%{keyword}%"

    # 获取博主ID（如果指定）
    blogger_id = None
    if blogger_name:
        cursor.execute("SELECT id, nickname FROM bloggers WHERE nickname = ?", (blogger_name,))
        blogger_row = cursor.fetchone()
        if not blogger_row:
            print(f"未找到博主: {blogger_name}")
            conn.close()
            return
        blogger_id = blogger_row[0]
        print(f"限定博主: {blogger_name}")

    # 执行搜索
    if blogger_id:
        # 先获取总数
        cursor.execute("""
            SELECT COUNT(*) FROM weibo_posts
            WHERE blogger_id = ? AND (content LIKE ? OR repost_content LIKE ?)
        """, (blogger_id, search_pattern, search_pattern))
        total_count = cursor.fetchone()[0]

        # 再获取结果
        cursor.execute("""
            SELECT b.nickname, w.weibo_id, w.content, w.publish_time, w.is_repost, w.repost_content, w.repost_author
            FROM weibo_posts w
            LEFT JOIN bloggers b ON w.blogger_id = b.id
            WHERE w.blogger_id = ? AND (w.content LIKE ? OR w.repost_content LIKE ?)
            ORDER BY w.publish_time DESC
            LIMIT ?
        """, (blogger_id, search_pattern, search_pattern, limit))
    else:
        # 先获取总数
        cursor.execute("""
            SELECT COUNT(*) FROM weibo_posts
            WHERE content LIKE ? OR repost_content LIKE ?
        """, (search_pattern, search_pattern))
        total_count = cursor.fetchone()[0]

        # 再获取结果
        cursor.execute("""
            SELECT b.nickname, w.weibo_id, w.content, w.publish_time, w.is_repost, w.repost_content, w.repost_author
            FROM weibo_posts w
            LEFT JOIN bloggers b ON w.blogger_id = b.id
            WHERE w.content LIKE ? OR w.repost_content LIKE ?
            ORDER BY w.publish_time DESC
            LIMIT ?
        """, (search_pattern, search_pattern, limit))

    rows = cursor.fetchall()
    conn.close()

    print(f"\n找到 {total_count} 条匹配的微博，显示前 {len(rows)} 条")

    if rows:
        for i, row in enumerate(rows, 1):
            nickname = row[0] or '未知'
            weibo_id = row[1]
            content = row[2]
            publish_time = row[3]
            is_repost = row[4]
            repost_content = row[5]
            repost_author = row[6]

            print(f"\n{i}. 【{nickname}】 {publish_time or '未知时间'}")
            print(f"   微博ID: {weibo_id}")

            # 高亮显示匹配内容
            if content:
                # 找到关键词位置，显示上下文
                keyword_lower = keyword.lower()
                content_lower = content.lower()
                pos = content_lower.find(keyword_lower)

                if pos >= 0:
                    # 显示关键词前后各50个字符
                    start = max(0, pos - 50)
                    end = min(len(content), pos + len(keyword) + 50)
                    context = content[start:end]
                    if start > 0:
                        context = "..." + context
                    if end < len(content):
                        context = context + "..."
                    print(f"   内容: {context}")
                else:
                    display_content = content[:150] + "..." if len(content) > 150 else content
                    print(f"   内容: {display_content}")

            # 如果是转发且转发内容匹配
            if is_repost and repost_content:
                repost_lower = repost_content.lower()
                if keyword_lower in repost_lower:
                    pos = repost_lower.find(keyword_lower)
                    start = max(0, pos - 30)
                    end = min(len(repost_content), pos + len(keyword) + 30)
                    context = repost_content[start:end]
                    print(f"   [转发匹配] @{repost_author}: ...{context}...")
    else:
        print("没有找到匹配的微博")


def show_factor_health(weeks: int = 8):
    """显示因子健康度(IC衰减监控)"""
    print("\n" + "=" * 70)
    print("因子IC衰减监控")
    print("=" * 70)

    from app.selector.ic_decay_monitor import ICDecayMonitor
    monitor = ICDecayMonitor()
    report = monitor.run_weekly_ic_analysis(weeks=weeks)

    print(f"\n{report.summary}")
    print("-" * 70)

    # 按健康度排序：suspended > degraded > watch > healthy
    status_order = {'suspended': 0, 'degraded': 1, 'watch': 2, 'healthy': 3}
    sorted_factors = sorted(report.factors, key=lambda f: status_order.get(f.health_status, 4))

    for f in sorted_factors:
        icon = {'healthy': '✅', 'watch': '⚪', 'degraded': '🟡', 'suspended': '🔴'}.get(f.health_status, '❓')
        trend_icon = {'improving': '📈', 'stable': '➡️', 'decaying': '📉', 'negative': '⬇️'}.get(f.ic_trend, '❓')
        print(f"{icon} {f.factor_name}: IC={f.current_ic:+.4f} {trend_icon} 状态={f.health_status}")
        if f.consecutive_low_weeks > 0:
            print(f"   连续{f.consecutive_low_weeks}周IC<{monitor.IC_LOW_THRESHOLD}")
        if f.consecutive_neg_weeks > 0:
            print(f"   连续{f.consecutive_neg_weeks}周IC<0")

    if report.actions_taken:
        print(f"\n操作建议:")
        for action in report.actions_taken:
            print(f"  {action}")

    if report.weight_adjustments:
        print(f"\n权重调整:")
        for fname, new_w in report.weight_adjustments.items():
            print(f"  {fname}: {new_w:.4f}")


def show_quality_feedback(days: int = 30):
    """显示推荐质量反馈"""
    print("\n" + "=" * 70)
    print("推荐质量反馈闭环")
    print("=" * 70)

    from app.selector.recommendation_feedback import RecommendationFeedback
    fb = RecommendationFeedback()
    report = fb.run_batch_review(days=days)

    # 系统健康度
    health_icons = {'healthy': '✅', 'watch': '⚪', 'degraded': '🟡', 'suspended': '🔴'}
    icon = health_icons.get(report.system_health, '❓')
    print(f"\n{icon} 系统状态: {report.system_health}")

    if report.degradation_reasons:
        for r in report.degradation_reasons:
            print(f"  ⚠️  {r}")

    # 核心指标
    print(f"\n{'─' * 70}")
    print(f"回顾期: {report.period_days}天 | 样本: {report.total_reviews}条")
    print(f"方向准确率: {report.direction_accuracy:.1%}")
    print(f"平均质量分: {report.avg_quality_score:.1f}")
    print(f"胜率: {report.win_rate:.1%} ({report.win_count}胜/{report.loss_count}负)")
    print(f"平均幅度偏差: {report.avg_magnitude_error:.1f}%")

    # 等级分布
    if report.grade_distribution:
        grades = []
        for g in ['A', 'B', 'C', 'D', 'F']:
            if g in report.grade_distribution:
                grades.append(f"{g}:{report.grade_distribution[g]}")
        print(f"等级分布: {' | '.join(grades)}")

    # 按来源统计
    if report.source_stats:
        print(f"\n{'─' * 70}")
        print("按来源统计:")
        for src, stats in report.source_stats.items():
            print(f"  {src}: {stats['count']}条 | 胜率{stats.get('win_rate', 0):.0%} | 均质量{stats.get('avg_quality', 0):.0f}")

    # 详细回顾
    if report.reviews:
        print(f"\n{'─' * 70}")
        grade_icons = {'A': '🌟', 'B': '✅', 'C': '⚪', 'D': '🟡', 'F': '🔴'}
        for r in report.reviews[:10]:
            gi = grade_icons.get(r.quality_grade, '❓')
            dir_icon = '↑' if r.direction_correct else '↓'
            print(f"  {gi} {r.stock_code} {r.stock_name} [{r.quality_grade}] "
                  f"预期{r.expected_return_pct:+.1f}%→实际{r.actual_return_pct:+.1f}% {dir_icon} "
                  f"质量{r.quality_score:.0f}")
            if r.hit_stop_loss:
                print(f"     ⚠️ 触发止损")

    print()


def show_tracked_positions(state: str = None):
    """显示持仓跟踪状态"""
    print("\n" + "=" * 70)
    print("持仓跟踪状态机")
    print("=" * 70)

    from app.selector.position_tracker import PositionTracker, ALL_STATES
    tracker = PositionTracker()

    # 概览
    summary = tracker.get_position_summary()
    print(f"\n持仓概览:")
    state_labels = {
        'observing': '观察中', 'ready_to_buy': '待买入', 'bought': '已买入',
        'holding': '持有中', 'sold': '已卖出', 'expired': '已过期',
    }
    for s in ALL_STATES + ['expired']:
        if summary.get(s, 0) > 0:
            print(f"  {state_labels.get(s, s)}: {summary[s]}只")

    if summary.get('holding_count_with_entry', 0) > 0:
        print(f"\n  持仓盈亏: 总{summary['holding_total_profit_pct']:.1f}% | 均{summary['holding_avg_profit_pct']:.1f}%")
    if summary.get('sold_count', 0) > 0:
        print(f"  已卖出: {summary['sold_count']}笔 | 胜率{summary['sold_win_rate']:.0f}% | 均收益{summary['sold_avg_return']:.1f}%")

    # 活跃持仓
    if state in ('sold', 'expired'):
        positions = tracker.get_sold_positions(days=30)
        positions = [p for p in positions if p.state == state]
    else:
        positions = tracker.get_active_positions()
        if state:
            positions = [p for p in positions if p.state == state]

    if positions:
        print(f"\n{'─' * 70}")
        state_icons = {
            'observing': '👁️', 'ready_to_buy': '🎯', 'bought': '💰',
            'holding': '📊', 'sold': '✅', 'expired': '⏰',
        }
        for p in positions:
            icon = state_icons.get(p.state, '❓')
            print(f"\n{icon} {p.code} {p.name} [{state_labels.get(p.state, p.state)}]")
            print(f"   推荐: {p.recommend_date} 评分{p.recommend_score:.0f} 来源={p.recommend_source}")
            if p.entry_price > 0:
                profit_str = f"{p.profit_pct:+.1f}%" if p.profit_pct else "0%"
                print(f"   入场: {p.entry_price:.2f} → 现价{p.current_price:.2f} {profit_str} | 持{p.hold_days}天")
                if p.stop_loss_price > 0:
                    print(f"   止损{p.stop_loss_price:.2f} | 止盈{p.take_profit_price:.2f}")
            if p.sell_reason:
                print(f"   卖出原因: {p.sell_reason} | 实际收益{p.actual_return_pct:+.1f}%")
    else:
        print("\n暂无跟踪持仓")

    print()


def show_trading_decision(blogger_name: str = None):
    """显示交易决策"""
    print("\n" + "=" * 70)
    print("交易决策引擎")
    print("=" * 70)

    from app.selector.trading_decision import TradingDecisionEngine
    from app.selector.stock_selector import get_stock_pool_by_blogger

    pool = get_stock_pool_by_blogger(blogger_name)
    if not pool:
        print("股票池为空，请先运行选股")
        return

    candidate_codes = [s['code'] for s in pool[:20]]
    engine = TradingDecisionEngine()
    report = engine.generate_decision_report(
        candidate_codes=candidate_codes,
        regime=None,
    )

    print(f"\n市场状态: {report.regime} | 策略: {report.strategy_name}")

    if report.buy_signals:
        print(f"\n买入信号({len(report.buy_signals)}只, 时机评分>{engine.TIMING_SCORE_THRESHOLD}):")
        for s in sorted(report.buy_signals, key=lambda x: -x.timing_score):
            icon = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(s.confidence, '⚪')
            print(f"  {icon} {s.code} {s.name} [{s.signal_type}] 时机{s.timing_score:.0f}分")
            print(f"     入场{s.entry_price:.2f} | 止损{s.stop_loss_price:.2f} | 止盈{s.take_profit_price:.2f} | 仓位{s.position_pct:.1f}%")
    else:
        print("\n暂无满足条件的买入信号")

    if report.sell_signals:
        print(f"\n卖出信号({len(report.sell_signals)}只):")
        for s in report.sell_signals:
            icon = {'immediate': '🔴', 'normal': '🟡', 'watch': '⚪'}.get(s.urgency, '⚪')
            print(f"  {icon} {s.code} {s.name} [{s.signal_type}] {s.urgency}")
            print(f"     {'; '.join(s.reasons)}")

    if report.portfolio_actions:
        print(f"\n操作建议:")
        for action in report.portfolio_actions:
            print(f"  → {action}")

    if report.risk_warnings:
        for w in report.risk_warnings:
            print(f"  ⚠️  {w}")

    engine.save_decision_report(report, blogger_name=blogger_name)


def main():
    parser = argparse.ArgumentParser(description='智能荐股系统')
    parser.add_argument('command', choices=['analyze', 'select', 'review', 'pool', 'predictions', 'bloggers', 'update', 'history', 'detail', 'summary', 'weibo', 'search', 'decision', 'factor-health', 'positions', 'quality', 'all'],
                        help='执行命令: analyze=分析博主, select=选股, review=复盘, pool=股票池, predictions=预测, bloggers=博主, update=更新, history=历史复盘, detail=复盘详情, summary=复盘汇总, weibo=微博, search=搜索, decision=交易决策, factor-health=因子IC监控, positions=持仓跟踪, quality=推荐质量, all=全部')
    parser.add_argument('--blogger', default=None, help='博主名称')
    parser.add_argument('--date', default=None, help='指定日期 (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=30, help='历史天数')
    parser.add_argument('--keyword', default=None, help='搜索关键词')
    parser.add_argument('--limit', type=int, default=20, help='显示数量限制')

    args = parser.parse_args()

    if args.command == 'analyze':
        analyze_blogger(args.blogger or '小棉袄')

    elif args.command == 'select':
        run_selection(args.blogger)

    elif args.command == 'review':
        run_review()

    elif args.command == 'pool':
        show_stock_pool(args.blogger)

    elif args.command == 'predictions':
        show_predictions()

    elif args.command == 'bloggers':
        show_bloggers()

    elif args.command == 'update':
        update_pool(args.blogger)

    elif args.command == 'history':
        show_review_history(args.days)

    elif args.command == 'detail':
        show_review_detail(args.date)

    elif args.command == 'summary':
        show_review_summary(args.days)

    elif args.command == 'weibo':
        show_weibo_posts(args.blogger, args.limit)

    elif args.command == 'search':
        search_weibo(args.keyword, args.blogger, args.limit)

    elif args.command == 'decision':
        show_trading_decision(args.blogger)

    elif args.command == 'factor-health':
        show_factor_health(args.days)

    elif args.command == 'positions':
        show_tracked_positions()

    elif args.command == 'quality':
        show_quality_feedback(args.days)

    elif args.command == 'all':
        # 执行完整流程
        analyze_blogger(args.blogger or '小棉袄')
        run_selection(args.blogger)
        # 复盘需要在收盘后执行
        print("\n提示: 复盘任务将在收盘后(14:50)自动执行")


if __name__ == "__main__":
    main()
