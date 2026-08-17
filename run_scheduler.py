#!/usr/bin/env python3
"""
启动后台调度服务
运行微博抓取、选股、复盘等定时任务
"""
import sys
import os
import asyncio
import signal
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.scheduler import init_all_schedulers, get_scheduler, get_stock_scheduler
from app.database import load_cookie


async def main():
    """启动所有调度任务"""
    print("=" * 70)
    print(f"启动调度服务 - {datetime.now()}")
    print("=" * 70)

    # 检查Cookie
    cookie = await load_cookie()
    if not cookie:
        print("警告: 未设置微博Cookie，微博抓取任务将无法执行")
        print("请通过Web界面设置Cookie")
    else:
        print(f"Cookie已设置 (长度: {len(cookie)})")

    # 初始化所有调度器
    await init_all_schedulers()

    # 显示任务信息
    crawler_scheduler = get_scheduler()
    stock_scheduler = get_stock_scheduler()

    print("\n微博抓取任务:")
    for job in crawler_scheduler.get_jobs_info():
        print(f"  {job['job_id']}: 下次执行 {job['next_run_time']}")

    print("\n股票相关任务:")
    for job in stock_scheduler.get_jobs_info():
        print(f"  {job['job_id']}: 下次执行 {job['next_run_time']}")

    print("\n调度服务已启动，按 Ctrl+C 停止")
    print("=" * 70)

    # 保持运行
    try:
        while True:
            await asyncio.sleep(3600)  # 每小时打印一次状态
            print(f"\n[{datetime.now()}] 调度服务运行中...")

            # 显示下次任务时间
            for job in crawler_scheduler.get_jobs_info():
                print(f"  {job['job_id']}: 下次执行 {job['next_run_time']}")
    except asyncio.CancelledError:
        print("\n收到停止信号，正在关闭...")
        crawler_scheduler.stop()
        stock_scheduler.stop()
        print("调度服务已停止")


def signal_handler(sig, frame):
    """信号处理"""
    print(f"\n收到信号 {sig}")
    sys.exit(0)


if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行主函数
    asyncio.run(main())