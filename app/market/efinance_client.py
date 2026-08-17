"""
efinance 行情数据客户端
基于 efinance 库获取 A股/港股/美股 实时行情
efinance 使用东方财富直接API，比 akshare 爬取更稳定、更快
"""
import efinance as ef
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .base_client import MarketClientBase, MarketType, MarketInfo


def _safe_float(val) -> float:
    """安全转换为 float，处理 '-'、空值等"""
    if val is None:
        return 0.0
    try:
        if isinstance(val, str):
            val = val.strip().replace(',', '')
            if val in ('-', '', 'N/A', '--'):
                return 0.0
        return float(val)
    except (ValueError, TypeError):
        return 0.0


class EFinanceAShareClient(MarketClientBase):
    """A股 efinance 客户端"""
    market_type = MarketType.A_SHARE

    def __init__(self):
        super().__init__()
        self._request_interval = 1.0
        self._realtime_cache = None
        self._cache_time = None

    def _get_realtime_quotes_ef(self) -> Dict[str, Dict]:
        """获取全市场实时行情并缓存（5分钟）"""
        if self._realtime_cache and self._cache_time:
            if (datetime.now() - self._cache_time).total_seconds() < 300:
                return self._realtime_cache

        try:
            df = ef.stock.get_realtime_quotes()
            if df.empty:
                return {}

            result = {}
            for _, row in df.iterrows():
                code = str(row['股票代码'])
                result[code] = {
                    'code': code,
                    'name': row['股票名称'],
                    'price': _safe_float(row['最新价']),
                    'change_pct': _safe_float(row['涨跌幅']),
                    'change': _safe_float(row['涨跌额']),
                    'volume': _safe_float(row['成交量']),
                    'amount': _safe_float(row['成交额']),
                    'high': _safe_float(row['最高']),
                    'low': _safe_float(row['最低']),
                    'open': _safe_float(row['今开']),
                    'prev_close': _safe_float(row['昨日收盘']),
                    'turnover_rate': _safe_float(row['换手率']),
                    'pe_ratio': _safe_float(row['动态市盈率']),
                    'market_cap': _safe_float(row['总市值']),
                    'update_time': str(row.get('更新时间', '')),
                }

            self._realtime_cache = result
            self._cache_time = datetime.now()
            return result

        except Exception as e:
            print(f"efinance A股行情获取失败: {e}")
            return {}

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        quotes = self._get_realtime_quotes_ef()
        if code in quotes:
            return quotes[code]
        return {'code': code, 'price': 0, 'change_pct': 0}

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        quotes = self._get_realtime_quotes_ef()
        if not quotes:
            return []
        sorted_quotes = sorted(quotes.values(), key=lambda x: x.get('amount', 0), reverse=True)
        return sorted_quotes[:limit]

    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq") -> pd.DataFrame:
        def _fetch():
            df = ef.stock.get_quote_history(code, klt=101, fqt=1 if adjust == "qfq" else 2 if adjust == "hfq" else 0)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
                '成交额': 'amount', '振幅': 'amplitude',
                '涨跌幅': 'change_pct', '涨跌额': 'change',
                '换手率': 'turnover_rate',
            })
            return df.tail(days)
        return self._retry_request(_fetch)

    def get_market_overview(self) -> Dict[str, Any]:
        quotes = self._get_realtime_quotes_ef()
        if not quotes:
            return {}
        up_count = sum(1 for q in quotes.values() if q.get('change_pct', 0) > 0)
        down_count = sum(1 for q in quotes.values() if q.get('change_pct', 0) < 0)
        flat_count = sum(1 for q in quotes.values() if q.get('change_pct', 0) == 0)
        limit_up = sum(1 for q in quotes.values() if q.get('change_pct', 0) >= 9.9)
        limit_down = sum(1 for q in quotes.values() if q.get('change_pct', 0) <= -9.9)
        avg_change = sum(q.get('change_pct', 0) for q in quotes.values()) / len(quotes) if quotes else 0
        return {
            'total_count': len(quotes), 'up_count': up_count, 'down_count': down_count,
            'flat_count': flat_count, 'limit_up': limit_up, 'limit_down': limit_down,
            'avg_change': avg_change,
            'market_strength': 'strong' if avg_change > 1 else 'weak' if avg_change < -1 else 'neutral',
        }

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        quotes = self._get_realtime_quotes_ef()
        if not quotes:
            return []
        results = []
        for q in quotes.values():
            if keyword in q.get('name', '') or keyword in q.get('code', ''):
                results.append(q)
                if len(results) >= 20:
                    break
        return results

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        hist = self.get_stock_history(code, days=60)
        if hist.empty or len(hist) < 20:
            return {}
        close = hist['close'].values
        ma5 = sum(close[-5:]) / 5 if len(close) >= 5 else 0
        ma10 = sum(close[-10:]) / 10 if len(close) >= 10 else 0
        ma20 = sum(close[-20:]) / 20 if len(close) >= 20 else 0
        current_price = close[-1] if len(close) > 0 else 0
        trend = "上涨" if ma5 > ma10 > ma20 else "下跌" if ma5 < ma10 < ma20 else "震荡"
        return {
            'current_price': current_price, 'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
            'trend': trend, 'is_above_ma5': current_price > ma5, 'is_above_ma20': current_price > ma20,
        }

    def get_index_quote(self, code: str) -> Dict[str, Any]:
        try:
            df = ef.stock.get_quote_history(code, klt=101)
            if df.empty:
                return {}
            latest = df.iloc[-1]
            return {
                'code': code, 'name': str(latest.get('股票名称', '')),
                'price': float(latest.get('收盘', 0)),
                'change_pct': float(latest.get('涨跌幅', 0)),
                'change': float(latest.get('涨跌额', 0)),
                'high': float(latest.get('最高', 0)),
                'low': float(latest.get('最低', 0)),
                'open': float(latest.get('开盘', 0)),
                'volume': float(latest.get('成交量', 0)),
                'amount': float(latest.get('成交额', 0)),
            }
        except Exception as e:
            print(f"efinance 获取指数行情失败 {code}: {e}")
            return {}


class EFinanceHKClient(MarketClientBase):
    """港股 efinance 客户端"""
    market_type = MarketType.HK

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        try:
            df = ef.stock.get_quote_history(code, klt=101)
            if df.empty:
                return {'code': code, 'price': 0, 'change_pct': 0}
            latest = df.iloc[-1]
            return {
                'code': code, 'name': str(latest.get('股票名称', '')),
                'price': float(latest.get('收盘', 0)),
                'change_pct': float(latest.get('涨跌幅', 0)),
                'high': float(latest.get('最高', 0)),
                'low': float(latest.get('最低', 0)),
                'open': float(latest.get('开盘', 0)),
                'volume': float(latest.get('成交量', 0)),
                'market': 'HK', 'timestamp': datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"efinance 港股行情失败 {code}: {e}")
            return {'code': code, 'price': 0, 'change_pct': 0}

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        # efinance 港股无全市场快照接口，回退到 akshare
        return []

    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq") -> pd.DataFrame:
        def _fetch():
            df = ef.stock.get_quote_history(code, klt=101)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
                '成交额': 'amount', '涨跌幅': 'change_pct', '换手率': 'turnover_rate',
            })
            return df.tail(days)
        return self._retry_request(_fetch)

    def get_market_overview(self) -> Dict[str, Any]:
        return {'market': 'HK', 'note': 'efinance 港股无市场概览接口'}

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        return []

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        hist = self.get_stock_history(code, days=60)
        if hist.empty or len(hist) < 20:
            return {}
        close = hist['close'].astype(float)
        return {
            'ma5': float(close.rolling(5).mean().iloc[-1]),
            'ma20': float(close.rolling(20).mean().iloc[-1]),
            'rsi_14': self._rsi(close, 14),
        }

    def get_index_quote(self, code: str = "HSI") -> Dict[str, Any]:
        return self.get_market_overview()


class EFinanceUSClient(MarketClientBase):
    """美股 efinance 客户端"""
    market_type = MarketType.US

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        try:
            df = ef.stock.get_quote_history(code, klt=101)
            if df.empty:
                return {'code': code, 'price': 0, 'change_pct': 0}
            latest = df.iloc[-1]
            return {
                'code': code, 'name': str(latest.get('股票名称', '')),
                'price': float(latest.get('收盘', 0)),
                'change_pct': float(latest.get('涨跌幅', 0)),
                'high': float(latest.get('最高', 0)),
                'low': float(latest.get('最低', 0)),
                'open': float(latest.get('开盘', 0)),
                'volume': float(latest.get('成交量', 0)),
                'market': 'US', 'timestamp': datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"efinance 美股行情失败 {code}: {e}")
            return {'code': code, 'price': 0, 'change_pct': 0}

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        return []

    def get_stock_history(self, code: str, days: int = 30, adjust: str = "qfq") -> pd.DataFrame:
        def _fetch():
            df = ef.stock.get_quote_history(code, klt=101)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
                '成交额': 'amount', '涨跌幅': 'change_pct', '换手率': 'turnover_rate',
            })
            return df.tail(days)
        return self._retry_request(_fetch)

    def get_market_overview(self) -> Dict[str, Any]:
        return {'market': 'US', 'note': 'efinance 美股无市场概览接口'}

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        return []

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        hist = self.get_stock_history(code, days=60)
        if hist.empty or len(hist) < 20:
            return {}
        close = hist['close'].astype(float)
        return {
            'ma5': float(close.rolling(5).mean().iloc[-1]),
            'ma20': float(close.rolling(20).mean().iloc[-1]),
            'rsi_14': self._rsi(close, 14),
        }

    def get_index_quote(self, code: str = ".INX") -> Dict[str, Any]:
        return self.get_market_overview()