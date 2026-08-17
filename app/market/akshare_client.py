"""
A股市场数据获取客户端 - 基于akshare
获取实时行情、板块数据、历史数据、资金流向等
"""
import akshare as ak
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache
import time
import random

from .base_client import MarketClientBase, MarketType


class AkShareClient(MarketClientBase):
    """A股市场数据客户端"""
    market_type = MarketType.A_SHARE

    # 板块概念映射
    SECTOR_CONCEPTS = {
        '科技芯片': ['芯片', '半导体', '集成电路', '国产芯片', '光刻胶'],
        '新能源': ['锂电池', '光伏', '储能', '新能源车', '充电桩'],
        '光模块': ['光模块', '光通信', 'CPO', '光纤'],
        '算力': ['算力', 'AI算力', '数据中心', '云计算'],
        '医药医疗': ['创新药', '医疗器械', '中药', '生物疫苗'],
        '消费白酒': ['白酒', '食品饮料', '家电'],
        '军工航天': ['军工', '航天', '商业航天', '无人机'],
        '周期资源': ['煤炭', '有色金属', '钢铁', '化工'],
    }

    # 行业板块代码映射
    INDUSTRY_CODES = {
        '半导体': 'BK0897',
        '光伏': 'BK0478',
        '锂电池': 'BK0574',
        '光模块': 'BK1132',
        'AI算力': 'BK0800',
        '储能': 'BK1004',
        '军工': 'BK0498',
        '新能源汽车': 'BK0493',
    }

    def __init__(self):
        self._request_interval = 2.0  # 请求间隔，避免频率过高（从1秒增加到2秒）
        self._last_request_time = 0
        self._max_retries = 3  # 最大重试次数
        self._realtime_quotes_cache = None  # 新浪数据缓存
        self._cache_time = None  # 缓存时间
        self._history_cache = {}  # 历史数据缓存

    def _wait_for_rate_limit(self):
        """等待以遵守请求频率限制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed + random.uniform(0.1, 0.5))
        self._last_request_time = time.time()

    def _retry_request(self, func, *args, **kwargs):
        """带重试的请求"""
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

    def _get_realtime_quotes_sina(self) -> Dict[str, Dict]:
        """使用新浪数据源获取实时行情（备用方案）"""
        # 检查缓存是否有效（5分钟内）
        if self._realtime_quotes_cache and self._cache_time:
            if (datetime.now() - self._cache_time).total_seconds() < 300:
                return self._realtime_quotes_cache

        try:
            print("使用新浪数据源获取实时行情...")
            df = ak.stock_zh_a_spot()

            if df.empty:
                return {}

            result = {}
            for _, row in df.iterrows():
                # 新浪数据源代码格式: sz300274, sh600519
                raw_code = str(row['代码'])
                # 提取市场前缀和纯代码
                if raw_code.startswith('sz'):
                    market = 'SZ'
                    pure_code = raw_code[2:]
                elif raw_code.startswith('sh'):
                    market = 'SH'
                    pure_code = raw_code[2:]
                elif raw_code.startswith('bj'):
                    market = 'BJ'
                    pure_code = raw_code[2:]
                else:
                    pure_code = raw_code
                    market = 'SZ' if raw_code.startswith('3') or raw_code.startswith('0') else 'SH'

                result[pure_code] = {
                    'code': pure_code,
                    'name': row['名称'],
                    'price': float(row['最新价']) if pd.notna(row['最新价']) else 0,
                    'change_pct': float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
                    'change': float(row['涨跌额']) if pd.notna(row['涨跌额']) else 0,
                    'volume': float(row['成交量']) if pd.notna(row['成交量']) else 0,
                    'amount': float(row['成交额']) if pd.notna(row['成交额']) else 0,
                    'high': float(row['最高']) if pd.notna(row['最高']) else 0,
                    'low': float(row['最低']) if pd.notna(row['最低']) else 0,
                    'open': float(row['今开']) if pd.notna(row['今开']) else 0,
                    'prev_close': float(row['昨收']) if pd.notna(row['昨收']) else 0,
                }

            self._realtime_quotes_cache = result
            self._cache_time = datetime.now()
            print(f"新浪数据源获取成功，共 {len(result)} 只股票")
            return result

        except Exception as e:
            print(f"新浪数据源获取失败: {e}")
            return {}

    def get_stock_price(self, code: str) -> Dict[str, Any]:
        """
        获取单只股票的实时价格（优先使用新浪数据源）

        Args:
            code: 股票代码（如 300274）

        Returns:
            Dict: 包含 price, change_pct 等字段
        """
        # 先尝试新浪数据源
        quotes = self._get_realtime_quotes_sina()
        if code in quotes:
            return quotes[code]

        # 如果新浪数据源没有，尝试东方财富
        try:
            self._wait_for_rate_limit()
            df = ak.stock_zh_a_spot_em()
            if not df.empty:
                row = df[df['代码'] == code]
                if not row.empty:
                    return {
                        'code': code,
                        'price': float(row['最新价'].values[0]) if pd.notna(row['最新价'].values[0]) else 0,
                        'change_pct': float(row['涨跌幅'].values[0]) if pd.notna(row['涨跌幅'].values[0]) else 0,
                    }
        except Exception as e:
            print(f"东方财富数据源获取失败 {code}: {e}")

        return {'code': code, 'price': 0, 'change_pct': 0}

    def get_realtime_quotes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取A股实时行情

        Returns:
            List[Dict]: 股票行情列表，包含代码、名称、价格、涨跌幅等
        """
        try:
            self._wait_for_rate_limit()
            df = ak.stock_zh_a_spot_em()

            if df.empty:
                return []

            result = []
            for _, row in df.head(limit).iterrows():
                result.append({
                    'code': row['代码'],
                    'name': row['名称'],
                    'price': float(row['最新价']) if pd.notna(row['最新价']) else 0,
                    'change_pct': float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
                    'change': float(row['涨跌额']) if pd.notna(row['涨跌额']) else 0,
                    'volume': float(row['成交量']) if pd.notna(row['成交量']) else 0,
                    'amount': float(row['成交额']) if pd.notna(row['成交额']) else 0,
                    'high': float(row['最高']) if pd.notna(row['最高']) else 0,
                    'low': float(row['最低']) if pd.notna(row['最低']) else 0,
                    'open': float(row['今开']) if pd.notna(row['今开']) else 0,
                    'prev_close': float(row['昨收']) if pd.notna(row['昨收']) else 0,
                    'turnover_rate': float(row['换手率']) if pd.notna(row['换手率']) else 0,
                    'pe_ratio': float(row['市盈率-动态']) if pd.notna(row['市盈率-动态']) else 0,
                    'market_cap': float(row['总市值']) if pd.notna(row['总市值']) else 0,
                })

            return result
        except Exception as e:
            print(f"获取实时行情失败: {e}")
            return []

    def get_stock_info(self, code: str) -> Optional[Dict[str, Any]]:
        """
        获取单只股票基本信息

        Args:
            code: 股票代码（如 000001）

        Returns:
            Dict: 股票基本信息
        """
        try:
            self._wait_for_rate_limit()
            df = ak.stock_individual_info_em(symbol=code)

            if df.empty:
                return None

            result = {}
            for _, row in df.iterrows():
                result[row['item']] = row['value']

            return result
        except Exception as e:
            print(f"获取股票信息失败 {code}: {e}")
            return None

    def get_stock_history(self, code: str, days: int = 30,
                          adjust: str = "qfq") -> pd.DataFrame:
        """
        获取股票历史数据

        Args:
            code: 股票代码
            days: 获取天数
            adjust: 复权类型 (qfq-前复权, hfq-后复权, 空-不复权)

        Returns:
            DataFrame: 历史数据，包含日期、开高低收、成交量等
        """
        # 检查缓存（缓存1小时）
        cache_key = f"{code}_{days}_{adjust}"
        if cache_key in self._history_cache:
            cached_data, cached_time = self._history_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < 3600:
                return cached_data

        def _fetch():
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y%m%d')

            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            return df

        try:
            df = self._retry_request(_fetch)

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '换手率': 'turnover_rate'
            })

            result = df.tail(days)
            # 缓存结果
            self._history_cache[cache_key] = (result, datetime.now())
            return result
        except Exception as e:
            print(f"获取历史数据失败 {code}: {e}")
            return pd.DataFrame()

    def get_sector_ranking(self, sector_type: str = "行业") -> List[Dict[str, Any]]:
        """
        获取板块涨跌排行

        Args:
            sector_type: 板块类型 (行业/概念/地域)

        Returns:
            List[Dict]: 板块排行列表
        """
        try:
            self._wait_for_rate_limit()

            if sector_type == "行业":
                df = ak.stock_board_industry_name_em()
            elif sector_type == "概念":
                df = ak.stock_board_concept_name_em()
            else:
                return []

            if df.empty:
                return []

            result = []
            for _, row in df.iterrows():
                result.append({
                    'name': row['板块名称'],
                    'code': row['板块代码'],
                    'change_pct': float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
                    'up_count': int(row['上涨家数']) if pd.notna(row['上涨家数']) else 0,
                    'down_count': int(row['下跌家数']) if pd.notna(row['下跌家数']) else 0,
                    'leading_stock': row['领涨股票'] if pd.notna(row['领涨股票']) else '',
                    'leading_change': float(row['领涨股票-涨跌幅']) if pd.notna(row['领涨股票-涨跌幅']) else 0,
                })

            return result
        except Exception as e:
            print(f"获取板块排行失败: {e}")
            return []

    def get_sector_stocks(self, sector_name: str) -> List[Dict[str, Any]]:
        """
        获取板块成分股

        Args:
            sector_name: 板块名称

        Returns:
            List[Dict]: 成分股列表
        """
        try:
            self._wait_for_rate_limit()
            df = ak.stock_board_industry_cons_em(symbol=sector_name)

            if df.empty:
                return []

            result = []
            for _, row in df.iterrows():
                result.append({
                    'code': row['代码'],
                    'name': row['名称'],
                    'price': float(row['最新价']) if pd.notna(row['最新价']) else 0,
                    'change_pct': float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
                    'market_cap': float(row['总市值']) if pd.notna(row['总市值']) else 0,
                })

            return result
        except Exception as e:
            print(f"获取板块成分股失败 {sector_name}: {e}")
            return []

    def get_concept_stocks(self, concept_name: str) -> List[Dict[str, Any]]:
        """
        获取概念板块成分股

        Args:
            concept_name: 概念名称

        Returns:
            List[Dict]: 成分股列表
        """
        try:
            self._wait_for_rate_limit()
            df = ak.stock_board_concept_cons_em(symbol=concept_name)

            if df.empty:
                return []

            result = []
            for _, row in df.iterrows():
                result.append({
                    'code': row['代码'],
                    'name': row['名称'],
                    'price': float(row['最新价']) if pd.notna(row['最新价']) else 0,
                    'change_pct': float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
                })

            return result
        except Exception as e:
            print(f"获取概念成分股失败 {concept_name}: {e}")
            return []

    def get_money_flow(self, code: str) -> Dict[str, Any]:
        """
        获取个股资金流向

        Args:
            code: 股票代码

        Returns:
            Dict: 资金流向数据
        """
        try:
            self._wait_for_rate_limit()
            df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")

            if df.empty:
                return {}

            # 获取最近一天的数据
            latest = df.iloc[-1]

            return {
                'date': latest['日期'] if pd.notna(latest['日期']) else '',
                'main_net_inflow': float(latest['主力净流入-净额']) if pd.notna(latest['主力净流入-净额']) else 0,
                'main_net_inflow_pct': float(latest['主力净流入-净占比']) if pd.notna(latest['主力净流入-净占比']) else 0,
                'retail_net_inflow': float(latest['小单净流入-净额']) if pd.notna(latest['小单净流入-净额']) else 0,
                'retail_net_inflow_pct': float(latest['小单净流入-净占比']) if pd.notna(latest['小单净流入-净占比']) else 0,
            }
        except Exception as e:
            print(f"获取资金流向失败 {code}: {e}")
            return {}

    def get_top_gainers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取涨幅榜

        Args:
            limit: 返回数量

        Returns:
            List[Dict]: 涨幅榜股票列表
        """
        quotes = self.get_realtime_quotes(limit=500)
        if not quotes:
            return []

        # 按涨跌幅排序
        sorted_quotes = sorted(quotes, key=lambda x: x['change_pct'], reverse=True)
        return sorted_quotes[:limit]

    def get_top_losers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取跌幅榜

        Args:
            limit: 返回数量

        Returns:
            List[Dict]: 跌幅榜股票列表
        """
        quotes = self.get_realtime_quotes(limit=500)
        if not quotes:
            return []

        # 按涨跌幅排序
        sorted_quotes = sorted(quotes, key=lambda x: x['change_pct'])
        return sorted_quotes[:limit]

    def get_hot_concepts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        获取热门概念板块

        Args:
            limit: 返回数量

        Returns:
            List[Dict]: 热门概念列表
        """
        concepts = self.get_sector_ranking(sector_type="概念")
        if not concepts:
            return []

        # 按涨跌幅排序
        sorted_concepts = sorted(concepts, key=lambda x: x['change_pct'], reverse=True)
        return sorted_concepts[:limit]

    def search_stock(self, keyword: str) -> List[Dict[str, Any]]:
        """
        搜索股票

        Args:
            keyword: 搜索关键词（股票名称或代码）

        Returns:
            List[Dict]: 匹配的股票列表
        """
        try:
            self._wait_for_rate_limit()
            df = ak.stock_zh_a_spot_em()

            if df.empty:
                return []

            # 搜索名称或代码
            mask = df['名称'].str.contains(keyword, na=False) | df['代码'].str.contains(keyword, na=False)
            filtered = df[mask]

            result = []
            for _, row in filtered.head(20).iterrows():
                result.append({
                    'code': row['代码'],
                    'name': row['名称'],
                    'price': float(row['最新价']) if pd.notna(row['最新价']) else 0,
                    'change_pct': float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
                })

            return result
        except Exception as e:
            print(f"搜索股票失败 {keyword}: {e}")
            return []

    def get_stock_technical_indicators(self, code: str) -> Dict[str, Any]:
        """
        获取股票技术指标

        Args:
            code: 股票代码

        Returns:
            Dict: 技术指标数据
        """
        try:
            # 获取历史数据
            df = self.get_stock_history(code, days=60)
            if df.empty:
                return {}

            # 计算技术指标
            close_prices = df['close'].values

            # MA均线
            ma5 = sum(close_prices[-5:]) / 5 if len(close_prices) >= 5 else 0
            ma10 = sum(close_prices[-10:]) / 10 if len(close_prices) >= 10 else 0
            ma20 = sum(close_prices[-20:]) / 20 if len(close_prices) >= 20 else 0

            # 当前价格
            current_price = close_prices[-1] if len(close_prices) > 0 else 0

            # 判断趋势
            trend = "上涨" if ma5 > ma10 > ma20 else "下跌" if ma5 < ma10 < ma20 else "震荡"

            # 计算振幅
            high_20 = max(df['high'].tail(20)) if len(df) >= 20 else 0
            low_20 = min(df['low'].tail(20)) if len(df) >= 20 else 0
            amplitude_20 = (high_20 - low_20) / low_20 * 100 if low_20 > 0 else 0

            # 成交量变化
            volume_5_avg = sum(df['volume'].tail(5)) / 5 if len(df) >= 5 else 0
            volume_today = df['volume'].iloc[-1] if len(df) > 0 else 0
            volume_change = (volume_today / volume_5_avg - 1) * 100 if volume_5_avg > 0 else 0

            return {
                'current_price': current_price,
                'ma5': ma5,
                'ma10': ma10,
                'ma20': ma20,
                'trend': trend,
                'amplitude_20d': amplitude_20,
                'volume_change_pct': volume_change,
                'high_20d': high_20,
                'low_20d': low_20,
                'is_above_ma5': current_price > ma5,
                'is_above_ma20': current_price > ma20,
            }
        except Exception as e:
            print(f"获取技术指标失败 {code}: {e}")
            return {}

    def get_index_quote(self, code: str) -> Dict[str, Any]:
        """
        获取指数行情数据

        Args:
            code: 指数代码（如 000001 上证指数, 399001 深证成指, 399006 创业板指）

        Returns:
            Dict: 指数行情数据
        """
        try:
            self._wait_for_rate_limit()
            # 使用新浪数据源获取指数
            df = ak.stock_zh_index_spot_sina()

            if df.empty:
                return {}

            # 查找对应指数
            row = df[df['代码'] == code]
            if row.empty:
                return {}

            return {
                'code': code,
                'name': row['名称'].values[0],
                'price': float(row['最新价'].values[0]) if pd.notna(row['最新价'].values[0]) else 0,
                'change_pct': float(row['涨跌幅'].values[0]) if pd.notna(row['涨跌幅'].values[0]) else 0,
                'change': float(row['涨跌额'].values[0]) if pd.notna(row['涨跌额'].values[0]) else 0,
                'high': float(row['最高'].values[0]) if pd.notna(row['最高'].values[0]) else 0,
                'low': float(row['最低'].values[0]) if pd.notna(row['最低'].values[0]) else 0,
                'open': float(row['今开'].values[0]) if pd.notna(row['今开'].values[0]) else 0,
                'prev_close': float(row['昨收'].values[0]) if pd.notna(row['昨收'].values[0]) else 0,
                'volume': float(row['成交量'].values[0]) if pd.notna(row['成交量'].values[0]) else 0,
                'amount': float(row['成交额'].values[0]) if pd.notna(row['成交额'].values[0]) else 0,
            }
        except Exception as e:
            print(f"获取指数行情失败 {code}: {e}")
            return {}

    def get_market_overview(self) -> Dict[str, Any]:
        """
        获取市场概览（涨跌家数、成交量等）

        Returns:
            Dict: 市场概览数据
        """
        try:
            quotes = self._get_realtime_quotes_sina()
            if not quotes:
                return {}

            up_count = sum(1 for q in quotes.values() if q.get('change_pct', 0) > 0)
            down_count = sum(1 for q in quotes.values() if q.get('change_pct', 0) < 0)
            flat_count = sum(1 for q in quotes.values() if q.get('change_pct', 0) == 0)

            # 涨停跌停统计
            limit_up = sum(1 for q in quotes.values() if q.get('change_pct', 0) >= 9.9)
            limit_down = sum(1 for q in quotes.values() if q.get('change_pct', 0) <= -9.9)

            # 平均涨跌幅
            avg_change = sum(q.get('change_pct', 0) for q in quotes.values()) / len(quotes) if quotes else 0

            return {
                'total_count': len(quotes),
                'up_count': up_count,
                'down_count': down_count,
                'flat_count': flat_count,
                'limit_up': limit_up,
                'limit_down': limit_down,
                'avg_change': avg_change,
                'market_strength': 'strong' if avg_change > 1 else 'weak' if avg_change < -1 else 'neutral'
            }
        except Exception as e:
            print(f"获取市场概览失败: {e}")
            return {}


# 全局实例
_market_client: Optional[AkShareClient] = None


def get_market_client() -> AkShareClient:
    """获取市场数据客户端实例"""
    global _market_client
    if _market_client is None:
        _market_client = AkShareClient()
    return _market_client
