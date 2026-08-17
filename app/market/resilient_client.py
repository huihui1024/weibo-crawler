"""
多数据源弹性行情客户端
自动在 akshare / efinance 之间切换，支持：
- 自动故障切换（failover）
- 熔断器（circuit breaker）
- 健康检查（health check）
- 请求重试（retry）
"""
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from .base_client import MarketClientBase, MarketType, MarketInfo


class SourceHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class CircuitBreaker:
    """熔断器：连续失败达到阈值后断开，冷却期后尝试恢复"""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.is_open = False

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True

    def can_execute(self) -> bool:
        with self._lock:
            if not self.is_open:
                return True
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
                return True
            return False

    @property
    def status(self) -> str:
        if not self.is_open:
            return "closed" if self.failure_count == 0 else "degraded"
        return "open"


class DataSourceWrapper:
    """包装单个数据源，带熔断器和健康统计"""

    def __init__(self, name: str, client: MarketClientBase, priority: int = 0):
        self.name = name
        self.client = client
        self.priority = priority  # 数字越小优先级越高
        self.circuit_breaker = CircuitBreaker()
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        self.last_success_time = None
        self.last_error = None

    def execute(self, method_name: str, *args, **kwargs):
        """执行请求，记录成功/失败"""
        if not self.circuit_breaker.can_execute():
            raise Exception(f"数据源 {self.name} 熔断中 (状态: {self.circuit_breaker.status})")

        start_time = time.time()
        self.total_requests += 1
        try:
            method = getattr(self.client, method_name)
            result = method(*args, **kwargs)
            elapsed = time.time() - start_time
            self.circuit_breaker.record_success()
            self.success_requests += 1
            self.last_success_time = datetime.now()
            self.avg_response_time = (self.avg_response_time * (self.success_requests - 1) + elapsed) / self.success_requests
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.failed_requests += 1
            self.last_error = str(e)
            raise

    @property
    def health(self) -> SourceHealth:
        if self.circuit_breaker.is_open:
            return SourceHealth.UNAVAILABLE
        if self.circuit_breaker.failure_count > 0:
            return SourceHealth.DEGRADED
        return SourceHealth.HEALTHY

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.success_requests / self.total_requests


class ResilientMarketClient(MarketClientBase):
    """
    多数据源弹性行情客户端
    按优先级尝试多个数据源，自动切换，带熔断保护
    """
    market_type = MarketType.A_SHARE

    def __init__(self, market_type: MarketType = MarketType.A_SHARE):
        super().__init__()
        self.market_type = market_type
        self._sources: List[DataSourceWrapper] = []
        self._setup_sources()

    def _setup_sources(self):
        """根据市场类型初始化数据源"""
        if self.market_type == MarketType.A_SHARE:
            self._init_a_share_sources()
        elif self.market_type == MarketType.HK:
            self._init_hk_sources()
        elif self.market_type == MarketType.US:
            self._init_us_sources()

    def _init_a_share_sources(self):
        """初始化A股数据源（新浪/腾讯直连优先，efinance备用，akshare最后）"""
        try:
            from .sina_tencent_client import SinaTencentClient
            self._sources.append(DataSourceWrapper("sina_tencent", SinaTencentClient(), priority=0))
        except ImportError:
            pass

        try:
            from .efinance_client import EFinanceAShareClient
            self._sources.append(DataSourceWrapper("efinance", EFinanceAShareClient(), priority=1))
        except ImportError:
            pass

        try:
            from .akshare_client import AkShareClient
            self._sources.append(DataSourceWrapper("akshare", AkShareClient(), priority=2))
        except ImportError:
            pass

    def _init_hk_sources(self):
        try:
            from .sina_tencent_client import SinaTencentHKClient
            self._sources.append(DataSourceWrapper("sina_tencent_hk", SinaTencentHKClient(), priority=0))
        except ImportError:
            pass

        try:
            from .hk_client import HKMarketClient
            self._sources.append(DataSourceWrapper("akshare_hk", HKMarketClient(), priority=1))
        except ImportError:
            pass

    def _init_us_sources(self):
        try:
            from .sina_tencent_client import SinaTencentUSClient
            self._sources.append(DataSourceWrapper("sina_tencent_us", SinaTencentUSClient(), priority=0))
        except ImportError:
            pass

        try:
            from .us_client import USMarketClient
            self._sources.append(DataSourceWrapper("akshare_us", USMarketClient(), priority=1))
        except ImportError:
            pass

    def _execute_with_failover(self, method_name: str, *args, **kwargs):
        """按优先级尝试所有数据源，直到成功"""
        sorted_sources = sorted(self._sources, key=lambda s: s.priority)
        errors = []

        for source in sorted_sources:
            try:
                result = source.execute(method_name, *args, **kwargs)
                return result
            except Exception as e:
                errors.append(f"[{source.name}] {e}")
                continue

        error_detail = "; ".join(errors) if errors else "无可用数据源"
        raise Exception(f"所有数据源均失败: {error_detail}")

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        return self._execute_with_failover("get_stock_price", code)

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._execute_with_failover("get_realtime_quotes", limit)

    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq"):
        return self._execute_with_failover("get_stock_history", code, days, adjust)

    def get_market_overview(self) -> Dict[str, Any]:
        return self._execute_with_failover("get_market_overview")

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        return self._execute_with_failover("search_stock", keyword)

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        return self._execute_with_failover("get_stock_technical_indicators", code)

    def get_index_quote(self, code: str) -> Dict[str, Any]:
        return self._execute_with_failover("get_index_quote", code)

    def get_source_health(self) -> Dict[str, Any]:
        """获取所有数据源的健康状态"""
        result = {}
        for source in self._sources:
            result[source.name] = {
                'health': source.health.value,
                'priority': source.priority,
                'total_requests': source.total_requests,
                'success_requests': source.success_requests,
                'failed_requests': source.failed_requests,
                'success_rate': f"{source.success_rate:.1%}",
                'avg_response_time': f"{source.avg_response_time:.2f}s",
                'circuit_breaker': source.circuit_breaker.status,
                'last_success': source.last_success_time.isoformat() if source.last_success_time else None,
                'last_error': source.last_error,
            }
        return result

    def get_available_sources(self) -> List[str]:
        """获取当前可用的数据源列表"""
        return [s.name for s in self._sources if s.health != SourceHealth.UNAVAILABLE]
