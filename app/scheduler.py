"""
定时任务调度模块
包含：微博爬虫、智能选股、每日复盘
选股预测次日走势，次日复盘验证
"""
import asyncio
import traceback
from datetime import datetime, date, timedelta
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from .config import get_config
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore

from .database import get_all_bloggers, get_blogger, get_sync_db
from .crawler.weibo import crawl_blogger_task
from .market import MarketType


class CrawlerScheduler:
    """爬虫调度器"""

    def __init__(self):
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': MemoryJobStore()},
            timezone='Asia/Shanghai'
        )
        self._running = False

    def start(self):
        """启动调度器"""
        if not self._running:
            self.scheduler.start()
            self._running = True
            print("调度器已启动")

    def stop(self):
        """停止调度器"""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            print("调度器已停止")

    async def add_blogger_job(self, blogger_id: int, interval_minutes: int):
        """添加博主的定时任务"""
        job_id = f"crawl_blogger_{blogger_id}"

        # 如果任务已存在，先移除
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)

        # 添加新任务
        self.scheduler.add_job(
            crawl_blogger_task,
            IntervalTrigger(minutes=interval_minutes),
            id=job_id,
            args=[blogger_id],
            replace_existing=True,
            max_instances=1,
            coalesce=True
        )

        print(f"已添加博主 {blogger_id} 的定时任务，间隔 {interval_minutes} 分钟")

    def remove_blogger_job(self, blogger_id: int):
        """移除博主的定时任务"""
        job_id = f"crawl_blogger_{blogger_id}"
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            print(f"已移除博主 {blogger_id} 的定时任务")

    def update_blogger_job(self, blogger_id: int, interval_minutes: int, enabled: bool):
        """更新博主的定时任务"""
        if enabled:
            asyncio.create_task(self.add_blogger_job(blogger_id, interval_minutes))
        else:
            self.remove_blogger_job(blogger_id)

    async def init_jobs(self):
        """初始化所有博主的定时任务"""
        bloggers = await get_all_bloggers(enabled_only=True)

        for blogger in bloggers:
            if blogger.id and blogger.enabled:
                await self.add_blogger_job(blogger.id, blogger.crawl_interval)

        print(f"已初始化 {len(bloggers)} 个博主的定时任务")

    def get_jobs_info(self) -> list:
        """获取所有任务信息"""
        try:
            jobs = self.scheduler.get_jobs()
            result = []

            for job in jobs:
                next_run = job.next_run_time
                result.append({
                    'job_id': job.id,
                    'next_run_time': next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else '未调度',
                    'trigger': str(job.trigger)
                })

            return result
        except Exception as e:
            print(f"获取任务信息失败: {e}")
            return []


class StockScheduler:
    """股票选股与复盘调度器"""

    def __init__(self):
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': MemoryJobStore()},
            timezone='Asia/Shanghai'
        )
        self._running = False

    def start(self):
        """启动调度器"""
        if not self._running:
            self.scheduler.start()
            self._running = True
            print("股票调度器已启动")

    def stop(self):
        """停止调度器"""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            print("股票调度器已停止")

    def add_stock_selection_job(self):
        """
        添加选股任务
        每个交易日9:15执行（开盘前）
        预测次日走势
        """
        self.scheduler.add_job(
            run_stock_selection_task,
            CronTrigger(day_of_week='mon-fri', hour=9, minute=15),
            id='stock_selection',
            replace_existing=True,
            max_instances=1
        )
        print("已添加选股任务：每个交易日 9:15 执行（预测次日走势）")

    def add_daily_review_job(self):
        """
        添加每日复盘任务
        每个交易日14:50执行（尾盘）
        复盘前一日对今日的预测
        """
        self.scheduler.add_job(
            run_daily_review_task,
            CronTrigger(day_of_week='mon-fri', hour=14, minute=50),
            id='daily_review',
            replace_existing=True,
            max_instances=1
        )
        print("已添加复盘任务：每个交易日 14:50 执行（验证前一日预测）")

    def add_prediction_update_job(self):
        """
        添加预测更新任务
        每个交易日15:30执行（收盘后）
        """
        self.scheduler.add_job(
            run_prediction_update_task,
            CronTrigger(day_of_week='mon-fri', hour=15, minute=30),
            id='prediction_update',
            replace_existing=True,
            max_instances=1
        )
        print("已添加预测更新任务：每个交易日 15:30 执行")

    def add_stock_price_update_job(self, interval_seconds: int = 15):
        """
        添加股票价格实时更新任务
        每15秒更新一次股票池价格

        Args:
            interval_seconds: 更新间隔秒数，默认15秒
        """
        self.scheduler.add_job(
            run_stock_price_update_task,
            IntervalTrigger(seconds=interval_seconds),
            id='stock_price_update',
            replace_existing=True,
            max_instances=1,
            coalesce=True
        )
        print(f"已添加股票价格更新任务：每 {interval_seconds} 秒执行一次")

    def add_quality_review_job(self):
        """推荐质量回顾任务：每周一收盘后自动运行"""
        self.scheduler.add_job(
            run_quality_review_task,
            CronTrigger(day_of_week='mon', hour=15, minute=30),
            id='quality_review',
            replace_existing=True,
            max_instances=1
        )
        print("已添加质量回顾任务：每周一 15:30 执行")

    def init_stock_jobs(self):
        """初始化所有股票相关任务"""
        self.add_stock_selection_job()
        self.add_daily_review_job()
        self.add_prediction_update_job()
        self.add_stock_price_update_job()
        self.add_quality_review_job()
        self.add_hk_selection_job()
        self.add_us_selection_job()
        self.add_hk_review_job()
        self.add_us_review_job()

    def add_hk_selection_job(self):
        """港股选股任务：周一至周五 09:00 HKT（开盘前）"""
        self.scheduler.add_job(
            run_hk_selection_task,
            CronTrigger(day_of_week='mon-fri', hour=9, minute=0),
            id='hk_selection',
            replace_existing=True,
            max_instances=1
        )
        print("已添加港股选股任务：每个交易日 09:00 HKT")

    def add_us_selection_job(self):
        """美股选股任务：周一至周五 21:00 CST（盘前）"""
        self.scheduler.add_job(
            run_us_selection_task,
            CronTrigger(day_of_week='mon-fri', hour=21, minute=0),
            id='us_selection',
            replace_existing=True,
            max_instances=1
        )
        print("已添加美股选股任务：每个交易日 21:00 CST")

    def add_hk_review_job(self):
        """港股复盘任务：周一至周五 15:50 HKT（收盘后）"""
        self.scheduler.add_job(
            run_hk_review_task,
            CronTrigger(day_of_week='mon-fri', hour=15, minute=50),
            id='hk_review',
            replace_existing=True,
            max_instances=1
        )
        print("已添加港股复盘任务：每个交易日 15:50 HKT")

    def add_us_review_job(self):
        """美股复盘任务：周二至周六 04:00 CST（收盘后）"""
        self.scheduler.add_job(
            run_us_review_task,
            CronTrigger(day_of_week='tue-sat', hour=4, minute=0),
            id='us_review',
            replace_existing=True,
            max_instances=1
        )
        print("已添加美股复盘任务：每个交易日 04:00 CST")

    def add_alert_check_job(self, interval_seconds: int = 60):
        """添加预警检查定时任务"""
        self.scheduler.add_job(
            run_alert_check_task,
            IntervalTrigger(seconds=interval_seconds),
            id='alert_check',
            replace_existing=True,
            max_instances=1,
            coalesce=True
        )
        print(f"已添加预警检查任务：每 {interval_seconds} 秒执行一次")

    def get_jobs_info(self) -> list:
        """获取所有任务信息"""
        try:
            jobs = self.scheduler.get_jobs()
            result = []
            for job in jobs:
                next_run = job.next_run_time
                result.append({
                    'job_id': job.id,
                    'next_run_time': next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else '未调度',
                    'trigger': str(job.trigger)
                })
            return result
        except Exception as e:
            print(f"获取任务信息失败: {e}")
            return []


# 选股任务
async def run_stock_selection_task(blogger_name: str = None):
    """执行选股任务

    Args:
        blogger_name: 博主名称，如果为None则使用第一个有数据的博主

    说明:
        预测次日走势，prediction_date 设为次日
    """
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 开始执行选股任务（博主: {blogger_name or '自动选择'}）")
    print(f"预测目标日期: {date.today() + timedelta(days=1)}（次日）")
    print('='*60)

    try:
        from .selector.stock_selector import StockSelector, run_stock_selector

        # 运行选股
        candidates = run_stock_selector(blogger_name=blogger_name)

        print(f"\n选股完成，共筛选出 {len(candidates)} 只候选股票")

        # 生成预测
        if candidates:
            from .selector.predictor import Predictor
            predictor = Predictor()

            codes = [c.code for c in candidates]
            predictions = predictor.batch_predict(codes)

            if predictions:
                # 只保留预期有涨幅的股票（过滤看跌/横盘）
                bullish_predictions = []
                bearish_trends = {'横盘震荡', '震荡下跌', '弱势下跌'}
                for pred in predictions:
                    if pred.expected_close_change > 0 and pred.expected_trend not in bearish_trends:
                        bullish_predictions.append(pred)

                filtered_count = len(predictions) - len(bullish_predictions)
                if filtered_count > 0:
                    print(f"过滤掉 {filtered_count} 只预期无涨幅的股票（保留 {len(bullish_predictions)} 只看涨）")

                # 保存看涨预测
                predictor.save_predictions_to_db(bullish_predictions, blogger_name=blogger_name)
                print(f"已生成并保存 {len(bullish_predictions)} 条看涨预测（博主: {blogger_name or '默认'}）")

                # 从候选股票池中也移除预期无涨幅的股票
                bullish_codes = {p.code for p in bullish_predictions}
                bullish_candidates = [c for c in candidates if c.code in bullish_codes]
                if len(bullish_candidates) < len(candidates):
                    print(f"从股票池中移除 {len(candidates) - len(bullish_candidates)} 只预期无涨幅股票")
                    # 重新保存过滤后的股票池
                    from .selector.stock_selector import StockSelector
                    selector = StockSelector()
                    selector.save_candidates_to_db(bullish_candidates, blogger_name=blogger_name)

    except Exception as e:
        print(f"选股任务执行失败: {e}")
        import traceback
        traceback.print_exc()


# 复盘任务
async def run_daily_review_task():
    """执行每日复盘任务

    说明:
        复盘前一日对今日的预测，验证今日实际走势
    """
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 开始执行复盘任务")
    print(f"复盘目标: 验证前一日对今日({date.today()})的预测")
    print('='*60)

    try:
        from .reviewer.daily_reviewer import DailyReviewer

        reviewer = DailyReviewer()
        summary = reviewer.review_predictions()

        if summary.total_predictions > 0:
            print(f"\n复盘完成")
            print(f"准确率: {summary.accuracy_rate:.1f}%")
            print(f"平均评分: {summary.avg_prediction_score:.1f}分")

            # 保存复盘报告
            report = reviewer.generate_review_report(summary)
            report_path = f"data/review_report_{summary.review_date}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"复盘报告已保存: {report_path}")
        else:
            print("没有需要复盘的预测记录")

    except Exception as e:
        print(f"复盘任务执行失败: {e}")
        import traceback
        traceback.print_exc()


# 预测更新任务
async def run_prediction_update_task():
    """更新预测结果"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 开始更新预测结果")
    print('='*60)

    try:
        from .reviewer.daily_reviewer import DailyReviewer

        # 执行复盘（更新实际数据）
        reviewer = DailyReviewer()
        summary = reviewer.review_predictions()

        print(f"预测结果更新完成")

    except Exception as e:
        print(f"预测更新任务执行失败: {e}")
        import traceback
        traceback.print_exc()


# 股票价格实时更新任务
async def run_stock_price_update_task():
    """更新股票池实时价格"""
    try:
        import sqlite3
        from .market import get_market_client

        conn = get_sync_db()
        cursor = conn.cursor()

        # 获取所有活跃股票
        cursor.execute("""
            SELECT stock_code, stock_name FROM stock_pool
            WHERE status = 'active'
        """)
        stocks = cursor.fetchall()

        if not stocks:
            conn.close()
            return

        # 获取市场客户端
        client = get_market_client()

        # 批量获取实时行情（使用缓存，5分钟内有效）
        quotes = client._get_realtime_quotes_sina()

        updated_count = 0
        for code, name in stocks:
            if code in quotes:
                quote = quotes[code]
                price = quote.get('price', 0)
                change_pct = quote.get('change_pct', 0)

                if price > 0:
                    cursor.execute("""
                        UPDATE stock_pool
                        SET price = ?, change_pct = ?
                        WHERE stock_code = ?
                    """, (price, change_pct, code))
                    updated_count += 1

        conn.commit()
        conn.close()

        if updated_count > 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 股票价格已更新: {updated_count} 只")

    except Exception as e:
        print(f"股票价格更新失败: {e}")


# 全局调度器实例
_crawler_scheduler: Optional[CrawlerScheduler] = None
_stock_scheduler: Optional[StockScheduler] = None


def get_scheduler() -> CrawlerScheduler:
    """获取爬虫调度器实例"""
    global _crawler_scheduler
    if _crawler_scheduler is None:
        _crawler_scheduler = CrawlerScheduler()
    return _crawler_scheduler


def get_stock_scheduler() -> StockScheduler:
    """获取股票调度器实例"""
    global _stock_scheduler
    if _stock_scheduler is None:
        _stock_scheduler = StockScheduler()
    return _stock_scheduler


async def init_scheduler():
    """初始化并启动调度器"""
    scheduler = get_scheduler()
    scheduler.start()
    await scheduler.init_jobs()
    return scheduler


async def init_stock_scheduler():
    """初始化并启动股票调度器"""
    scheduler = get_stock_scheduler()
    scheduler.start()
    scheduler.init_stock_jobs()
    return scheduler


async def init_all_schedulers():
    """初始化所有调度器"""
    await init_scheduler()
    await init_stock_scheduler()

    # P1: 初始化预警检查任务
    config = get_config()
    alerts_cfg = getattr(config, 'alerts', None)
    if alerts_cfg and getattr(alerts_cfg, 'enabled', False):
        stock_scheduler = get_stock_scheduler()
        stock_scheduler.add_alert_check_job(alerts_cfg.check_interval_seconds)


# ==================== P1: 预警调度 ====================

_alert_manager = None


def get_alert_manager():
    """获取全局预警管理器实例"""
    global _alert_manager
    if _alert_manager is None:
        from .notifier.realtime_alert import AlertManager, create_default_rules
        _alert_manager = AlertManager()
        for rule in create_default_rules():
            _alert_manager.add_rule(rule)
    return _alert_manager


async def run_alert_check_task():
    """定期预警检查任务"""
    try:
        manager = get_alert_manager()

        # 1. 检查股票池价格预警
        import sqlite3
        conn = get_sync_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT stock_code, stock_name, price, change_pct
            FROM stock_pool WHERE status = 'active'
        """)
        stocks = cursor.fetchall()
        conn.close()

        if stocks:
            from .market import get_market_client
            client = get_market_client()
            quotes = client._get_realtime_quotes_sina()

            for code, name, prev_price, prev_change in stocks:
                if code in quotes:
                    quote = quotes[code]
                    current_price = quote.get('price', 0)
                    change_pct = quote.get('change_pct', 0)
                    if current_price > 0:
                        manager.check_price(code, name, current_price, prev_price, change_pct)

        # 2. 清理过期预警
        from .database import cleanup_expired_alerts
        await cleanup_expired_alerts()

        # 3. 清理语义检测器冷却缓存
        manager.semantic_detector.cleanup_cooldowns()

    except Exception as e:
        print(f"预警检查任务失败: {e}")


# ── 港股/美股调度任务 ──────────────────────

async def run_hk_selection_task():
    """港股选股任务 — 基本面深度分析"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 开始执行港股选股任务（基本面分析）")
    print('='*60)

    try:
        from .selector.stock_selector import StockSelector, StockCandidate
        from .market.hk_client import HKMarketClient

        client = HKMarketClient()
        results = client.analyze_and_predict(limit=30)
        print(f"港股基本面分析完成: {len(results)} 只")

        candidates = []
        for stock in results:
            breakdown = stock.get('score_breakdown', {})
            reasons = []
            for dim in ('quality', 'growth', 'valuation'):
                for detail in breakdown.get(dim, {}).get('details', []):
                    reasons.append(detail)
            if stock.get('change_pct', 0) > 3:
                reasons.append(f"强势上涨{stock['change_pct']:+.1f}%")
            elif stock.get('change_pct', 0) > 0:
                reasons.append("正向动量")
            if not reasons:
                reasons.append("综合评分通过筛选")

            candidates.append(StockCandidate(
                code=stock.get('code', ''),
                name=stock.get('name', ''),
                sector='港股',
                concept='基本面优质' if stock.get('prediction_score', 0) > 60 else '港股热点',
                price=stock.get('price', 0),
                change_pct=stock.get('change_pct', 0),
                style_match_score=float(stock.get('prediction_score', 50)),
                selection_source='基本面分析',
                position_weight=0,
                selection_reasons=reasons,
                technical_indicators={
                    'predicted_trend': stock.get('predicted_trend', 'neutral'),
                    'confidence': stock.get('confidence', 0.3),
                    'predicted_change': stock.get('predicted_change', 0),
                    'quality_score': stock.get('quality_score', 50),
                    'growth_score': stock.get('growth_score', 50),
                    'valuation_score': stock.get('valuation_score', 50),
                    'momentum_score': stock.get('momentum_score', 50),
                },
                market='HK',
            ))

        selector = StockSelector(market_type=MarketType.HK)
        selector.save_candidates_to_db(candidates, blogger_name='港股自动')
        print(f"港股选股完成，共 {len(candidates)} 只")
        for c in candidates[:5]:
            print(f"  {c.code} {c.name} 评分={c.style_match_score:.0f} "
                  f"趋势={c.technical_indicators.get('predicted_trend', 'N/A')} "
                  f"原因={c.selection_reasons[:2]}")

    except Exception as e:
        print(f"港股选股任务失败: {e}")
        traceback.print_exc()


async def run_us_selection_task():
    """美股选股任务 — 基本面深度分析"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 开始执行美股选股任务（基本面分析）")
    print('='*60)

    try:
        from .selector.stock_selector import StockSelector, StockCandidate
        from .market.us_client import USMarketClient

        client = USMarketClient()
        results = client.analyze_and_predict(limit=30)
        print(f"美股基本面分析完成: {len(results)} 只")

        candidates = []
        for stock in results:
            breakdown = stock.get('score_breakdown', {})
            reasons = []
            for dim in ('quality', 'growth', 'valuation'):
                for detail in breakdown.get(dim, {}).get('details', []):
                    reasons.append(detail)
            if stock.get('change_pct', 0) > 3:
                reasons.append(f"强势上涨{stock['change_pct']:+.1f}%")
            elif stock.get('change_pct', 0) > 0:
                reasons.append("正向动量")
            if not reasons:
                reasons.append("综合评分通过筛选")

            candidates.append(StockCandidate(
                code=stock.get('code', ''),
                name=stock.get('name', ''),
                sector='美股',
                concept='基本面优质' if stock.get('prediction_score', 0) > 60 else '美股热点',
                price=stock.get('price', 0),
                change_pct=stock.get('change_pct', 0),
                style_match_score=float(stock.get('prediction_score', 50)),
                selection_source='基本面分析',
                position_weight=0,
                selection_reasons=reasons,
                technical_indicators={
                    'predicted_trend': stock.get('predicted_trend', 'neutral'),
                    'confidence': stock.get('confidence', 0.3),
                    'predicted_change': stock.get('predicted_change', 0),
                    'quality_score': stock.get('quality_score', 50),
                    'growth_score': stock.get('growth_score', 50),
                    'valuation_score': stock.get('valuation_score', 50),
                    'momentum_score': stock.get('momentum_score', 50),
                },
                market='US',
            ))

        selector = StockSelector(market_type=MarketType.US)
        selector.save_candidates_to_db(candidates, blogger_name='美股自动')
        print(f"美股选股完成，共 {len(candidates)} 只")
        for c in candidates[:5]:
            print(f"  {c.code} {c.name} 评分={c.style_match_score:.0f} "
                  f"趋势={c.technical_indicators.get('predicted_trend', 'N/A')} "
                  f"原因={c.selection_reasons[:2]}")

    except Exception as e:
        print(f"美股选股任务失败: {e}")
        traceback.print_exc()


async def run_hk_review_task():
    """港股复盘任务 — 验证预测准确性"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 开始执行港股复盘任务")
    print('='*60)

    try:
        from .market.hk_client import HKMarketClient
        client = HKMarketClient()

        # 获取当前涨幅排行用于验证
        current = client.fetch_top_gainers(limit=20)
        print(f"港股今日涨幅前{min(len(current), 5)}名:")
        for s in current[:5]:
            print(f"  {s['code']} {s['name']} {s.get('change_pct', 0):+.1f}%")

        # 获取当前基本面分析结果
        results = client.analyze_and_predict(limit=10)
        if results:
            print(f"\n港股基本面分析TOP5:")
            for s in results[:5]:
                trend = s.get('predicted_trend', 'neutral')
                score = s.get('prediction_score', 50)
                conf = s.get('confidence', 0.3)
                print(f"  {s['code']} {s['name']} 评分={score} 趋势={trend} 置信度={conf:.0%}")
    except Exception as e:
        print(f"港股复盘任务失败: {e}")


async def run_us_review_task():
    """美股复盘任务 — 验证预测准确性"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 开始执行美股复盘任务")
    print('='*60)

    try:
        from .market.us_client import USMarketClient
        client = USMarketClient()

        current = client.fetch_top_gainers(limit=20)
        print(f"美股今日涨幅前{min(len(current), 5)}名:")
        for s in current[:5]:
            print(f"  {s['code']} {s['name']} {s.get('change_pct', 0):+.1f}%")

        results = client.analyze_and_predict(limit=10)
        if results:
            print(f"\n美股基本面分析TOP5:")
            for s in results[:5]:
                trend = s.get('predicted_trend', 'neutral')
                score = s.get('prediction_score', 50)
                conf = s.get('confidence', 0.3)
                print(f"  {s['code']} {s['name']} 评分={score} 趋势={trend} 置信度={conf:.0%}")
    except Exception as e:
        print(f"美股复盘任务失败: {e}")


async def run_quality_review_task():
    """推荐质量回顾任务 — 自动回顾近30天推荐质量，结果写入信号强度判断"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 开始执行推荐质量回顾任务")
    print('='*60)

    try:
        from app.selector.recommendation_feedback import RecommendationFeedback
        fb = RecommendationFeedback()
        report = fb.run_batch_review(days=30)

        health_icons = {'healthy': '✅', 'watch': '⚪', 'degraded': '🟡', 'suspended': '🔴'}
        icon = health_icons.get(report.system_health, '❓')
        print(f"{icon} 系统状态: {report.system_health}")
        print(f"样本: {report.total_reviews}条 | 方向准确率: {report.direction_accuracy:.1%} | 胜率: {report.win_rate:.1%}")
        print(f"平均质量: {report.avg_quality_score:.1f} | 平均偏差: {report.avg_magnitude_error:.1f}%")

        if report.degradation_reasons:
            for r in report.degradation_reasons:
                print(f"  ⚠️  {r}")

        # 同步持仓跟踪
        from app.selector.position_tracker import PositionTracker
        tracker = PositionTracker()
        sync = tracker.sync_from_stock_pool()
        print(f"持仓跟踪同步: 新增{sync['added']}, 更新{sync['updated']}")

        # 更新持仓状态
        updates = tracker.update_all_positions()
        print(f"持仓更新: 价格{updates['price_updated']}只, 状态变更{updates['state_changed']}只")

        print(f"推荐质量回顾任务完成")
    except Exception as e:
        print(f"推荐质量回顾任务失败: {e}")
        traceback.print_exc()
