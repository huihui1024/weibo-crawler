"""
定时任务调度模块 - 仅微博爬虫调度
"""
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore

from .database import get_all_bloggers
from .crawler.weibo import crawl_blogger_task


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
            misfire_grace_time=60
        )

    def remove_blogger_job(self, blogger_id: int):
        """移除博主的定时任务"""
        job_id = f"crawl_blogger_{blogger_id}"
        if self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)

    def update_blogger_job(self, blogger_id: int, interval_minutes: int, enabled: bool):
        """更新博主的定时任务"""
        if enabled:
            self.add_blogger_job(blogger_id, interval_minutes)
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


# 全局调度器实例
_crawler_scheduler: Optional[CrawlerScheduler] = None


def get_scheduler() -> CrawlerScheduler:
    """获取爬虫调度器实例"""
    global _crawler_scheduler
    if _crawler_scheduler is None:
        _crawler_scheduler = CrawlerScheduler()
    return _crawler_scheduler


async def init_scheduler():
    """初始化并启动调度器"""
    scheduler = get_scheduler()
    scheduler.start()
    await scheduler.init_jobs()
    return scheduler
