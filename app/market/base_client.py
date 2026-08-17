"""
市场数据客户端基类
统一A股/港股/美股的数据接口
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


class MarketType(Enum):
    """市场类型"""
    A_SHARE = "A"     # A股
    HK = "HK"         # 港股
    US = "US"         # 美股


class MarketInfo:
    """市场信息常量"""
    MARKET_CONFIG = {
        MarketType.A_SHARE: {
            'name': 'A股',
            'timezone': 'Asia/Shanghai',
            'trading_hours': [(9, 30, 11, 30), (13, 0, 15, 0)],
            'currency': 'CNY',
            'code_pattern': r'^\d{6}$',
            'limit_up_pct': 10.0,   # 涨停幅度（主板）
            'limit_down_pct': -10.0,
        },
        MarketType.HK: {
            'name': '港股',
            'timezone': 'Asia/Hong_Kong',
            'trading_hours': [(9, 30, 12, 0), (13, 0, 16, 0)],
            'currency': 'HKD',
            'code_pattern': r'^\d{5}$',
            'limit_up_pct': None,    # 港股无涨跌停
            'limit_down_pct': None,
        },
        MarketType.US: {
            'name': '美股',
            'timezone': 'America/New_York',
            'trading_hours': [(9, 30, 16, 0)],
            'currency': 'USD',
            'code_pattern': r'^[A-Z]+$',
            'limit_up_pct': None,    # 美股无涨跌停
            'limit_down_pct': None,
        },
    }

    @classmethod
    def get_config(cls, market_type: MarketType) -> Dict:
        return cls.MARKET_CONFIG[market_type]

    @classmethod
    def is_trading_time(cls, market_type: MarketType) -> bool:
        """判断当前是否在该市场的交易时间"""
        config = cls.MARKET_CONFIG[market_type]
        tz_name = config['timezone']
        import pytz
        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)
        hour, minute = now.hour, now.minute
        weekday = now.weekday()

        if weekday >= 5:  # 周六周日不交易
            return False

        for start_h, start_m, end_h, end_m in config['trading_hours']:
            if start_h * 60 + start_m <= hour * 60 + minute <= end_h * 60 + end_m:
                return True
        return False


class MarketClientBase(ABC):
    """市场数据客户端基类"""

    market_type: MarketType = MarketType.A_SHARE

    def __init__(self):
        self._request_interval = 2.0
        self._last_request_time = 0
        self._max_retries = 3
        self._realtime_cache = None
        self._cache_time = None
        self._history_cache = {}

    @abstractmethod
    def get_stock_price(self, code: str) -> Dict[str, Any]:
        """获取单只股票实时价格"""
        pass

    @abstractmethod
    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取实时行情列表"""
        pass

    @abstractmethod
    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq") -> pd.DataFrame:
        """获取历史数据"""
        pass

    @abstractmethod
    def get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览"""
        pass

    @abstractmethod
    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索股票"""
        pass

    @abstractmethod
    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        """获取技术指标"""
        pass

    @abstractmethod
    def get_index_quote(self, code: str) -> Dict[str, Any]:
        """获取指数行情"""
        pass

    def get_market_name(self) -> str:
        """获取市场名称"""
        return MarketInfo.get_config(self.market_type)['name']

    def get_currency(self) -> str:
        """获取货币单位"""
        return MarketInfo.get_config(self.market_type)['currency']

    def _wait_for_rate_limit(self):
        """等待以遵守请求频率限制"""
        import time, random
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed + random.uniform(0.1, 0.5))
        self._last_request_time = time.time()

    def _retry_request(self, func, *args, **kwargs):
        """带重试的请求"""
        import time, random
        last_error = None
        for attempt in range(self._max_retries):
            try:
                self._wait_for_rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                wait_time = (attempt + 1) * 2 + random.uniform(0.5, 1.5)
                print(f"请求失败，{wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self._max_retries}): {e}")
                time.sleep(wait_time)
        raise last_error