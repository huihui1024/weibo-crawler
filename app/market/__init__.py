"""
市场数据模块
统一使用新浪+腾讯双源，不再依赖AkShare/EastMoney不稳定源
实时行情用Sina hq API，历史K线用腾讯
"""
from typing import Dict

from .base_client import MarketClientBase, MarketType, MarketInfo
from .sina_tencent_client import SinaTencentClient, SinaTencentHKClient, SinaTencentUSClient
from .hk_client import HKMarketClient
from .us_client import USMarketClient
from .fundamental_data import FundamentalDataClient

__all__ = [
    'MarketClientBase', 'MarketType', 'MarketInfo',
    'SinaTencentClient', 'SinaTencentHKClient', 'SinaTencentUSClient',
    'HKMarketClient', 'USMarketClient', 'FundamentalDataClient',
    'get_market_client',
]

# 全局客户端缓存
_market_clients: Dict[MarketType, MarketClientBase] = {}


def get_market_client(market_type: MarketType = MarketType.A_SHARE) -> MarketClientBase:
    """获取市场数据客户端实例（工厂函数）

    A股 → SinaTencentClient（新浪实时+腾讯历史，双源稳定）
    港股 → HKMarketClient（基本面分析+预测）
    美股 → USMarketClient（基本面分析+预测）
    """
    global _market_clients
    if market_type not in _market_clients:
        if market_type == MarketType.A_SHARE:
            _market_clients[market_type] = SinaTencentClient()
        elif market_type == MarketType.HK:
            _market_clients[market_type] = HKMarketClient()
        elif market_type == MarketType.US:
            _market_clients[market_type] = USMarketClient()
        else:
            raise ValueError(f"不支持的市场类型: {market_type}")
    return _market_clients[market_type]