"""
配置管理模块
"""
import os
import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel


class BloggerConfig(BaseModel):
    uid: str
    nickname: str = ""
    interval: int = 60  # 采集间隔(分钟)
    enabled: bool = True


class CrawlerConfig(BaseModel):
    user_agent: str = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
    request_delay: int = 3
    timeout: int = 30
    max_retries: int = 3


class StorageConfig(BaseModel):
    database: str = "data/weibo.db"
    image_dir: str = "data/images"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class MarketConfig(BaseModel):
    enabled_markets: List[str] = ["A", "HK", "US"]
    default_market: str = "A"


class TradingConfig(BaseModel):
    simulated_enabled: bool = True
    commission_rate: float = 0.0003
    slippage: float = 0.001
    initial_capital: float = 1000000


class AppConfig(BaseModel):
    bloggers: List[BloggerConfig] = []
    crawler: CrawlerConfig = CrawlerConfig()
    storage: StorageConfig = StorageConfig()
    server: ServerConfig = ServerConfig()
    market: MarketConfig = MarketConfig()
    trading: TradingConfig = TradingConfig()


_config: Optional[AppConfig] = None


def get_config_path() -> Path:
    """获取配置文件路径"""
    base_dir = Path(__file__).parent.parent
    return base_dir / "config.yaml"


def load_config() -> AppConfig:
    """加载配置文件"""
    global _config
    config_path = get_config_path()

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 确保所有字段都有默认值
        if data.get('bloggers') is None:
            data['bloggers'] = []
        _config = AppConfig(**data)
    else:
        _config = AppConfig()

    return _config


def get_config() -> AppConfig:
    """获取配置实例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def save_config(config: AppConfig) -> None:
    """保存配置到文件"""
    config_path = get_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, allow_unicode=True, default_flow_style=False)
