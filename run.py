#!/usr/bin/env python3
"""
微博博主内容抓取应用启动脚本
"""
import uvicorn
from app.config import get_config


def main():
    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False
    )


if __name__ == "__main__":
    main()
