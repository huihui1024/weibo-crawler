#!/usr/bin/env python3
"""
股票价格实时更新服务
每15秒更新一次股票池价格
"""
import sys
import os
import asyncio
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.scheduler import get_stock_scheduler


def signal_handler(sig, frame):
    """处理退出信号"""
    print("\n正在停止价格更新服务...")
    scheduler = get_stock_scheduler()
    scheduler.stop()
    sys.exit(0)


async def main():
    """启动价格更新服务"""
    print("=" * 60)
    print("股票价格实时更新服务")
    print("更新频率: 每15秒")
    print("按 Ctrl+C 停止")
    print("=" * 60)

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 获取调度器并启动
    scheduler = get_stock_scheduler()
    scheduler.start()

    # 只添加价格更新任务
    scheduler.add_stock_price_update_job(interval_seconds=15)

    print(f"\n[{datetime.now()}] 服务已启动，等待下一次更新...")

    # 保持运行
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
