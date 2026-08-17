"""
基本面数据源模块
对接财报数据、北向资金、龙虎榜等基本面数据
支持多数据源、缓存机制、错误处理、数据清洗
"""
import akshare as ak
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import time
import random
import json
import os
import hashlib
import pickle
import math


# ==================== 数据质量常量 ====================

class DataQuality:
    """数据质量等级"""
    HIGH = 1.0      # 高质量：数据完整且有效
    MEDIUM = 0.7    # 中等质量：部分数据有效
    LOW = 0.4       # 低质量：数据可能有问题
    INVALID = 0.0   # 无效数据


class DataValidator:
    """数据校验器"""

    # 合理范围定义
    RANGES = {
        # 财务指标合理范围
        'roe': (-50, 100),           # ROE: -50% 到 100%
        'gross_margin': (-20, 100),  # 毛利率: -20% 到 100%
        'net_margin': (-100, 100),   # 净利率: -100% 到 100%
        'revenue_yoy': (-200, 500),  # 营收同比: -200% 到 500%
        'net_profit_yoy': (-500, 1000),  # 利润同比: -500% 到 1000%
        'asset_liability_ratio': (0, 100),  # 资产负债率: 0% 到 100%
        'current_ratio': (0, 20),    # 流动比率: 0 到 20
        'eps': (-100, 1000),         # 每股收益
        'bvps': (0, 10000),          # 每股净资产
        'pe_ratio': (-1000, 1000),   # 市盈率
        'pb_ratio': (0, 100),        # 市净率

        # 北向资金合理范围
        'hold_ratio': (0, 50),       # 持股比例: 0% 到 50%
        'hold_change_5d': (-100, 200),  # 5日变化: -100% 到 200%
        'hold_change_20d': (-100, 500), # 20日变化
        'net_buy_5d': (-100, 100),   # 5日净买入（亿）

        # 龙虎榜合理范围
        'change_pct': (-30, 30),     # 涨跌幅: -30% 到 30%
        'turnover_rate': (0, 100),    # 换手率: 0% 到 100%
        'net_buy': (-100000, 100000), # 净买入（万）
    }

    # 特殊行业调整
    INDUSTRY_ADJUSTMENTS = {
        '银行': {
            'asset_liability_ratio': (85, 98),  # 银行资产负债率通常很高
            'gross_margin': None,  # 银行毛利率不适用
            'current_ratio': None,  # 银行流动比率不适用
        },
        '保险': {
            'asset_liability_ratio': (80, 98),
        },
        '证券': {
            'asset_liability_ratio': (60, 95),
        },
    }

    @classmethod
    def validate_value(cls, field_name: str, value: float, industry: str = None) -> Tuple[bool, float]:
        """
        校验数值是否在合理范围内

        Args:
            field_name: 字段名
            value: 数值
            industry: 行业（用于特殊行业调整）

        Returns:
            Tuple[bool, float]: (是否有效, 数据质量系数)
        """
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return False, DataQuality.INVALID

        # 检查特殊行业调整
        if industry and industry in cls.INDUSTRY_ADJUSTMENTS:
            industry_ranges = cls.INDUSTRY_ADJUSTMENTS[industry]
            if field_name in industry_ranges:
                special_range = industry_ranges[field_name]
                if special_range is None:
                    # 该指标对该行业不适用
                    return True, DataQuality.MEDIUM
                min_val, max_val = special_range
                if min_val <= value <= max_val:
                    return True, DataQuality.HIGH
                # 在合理范围附近也算有效
                if min_val * 0.8 <= value <= max_val * 1.2:
                    return True, DataQuality.MEDIUM
                return False, DataQuality.LOW

        # 通用范围校验
        if field_name in cls.RANGES:
            min_val, max_val = cls.RANGES[field_name]
            if min_val <= value <= max_val:
                return True, DataQuality.HIGH
            # 在合理范围附近也算有效（给予中等质量）
            if min_val * 0.5 <= value <= max_val * 1.5:
                return True, DataQuality.MEDIUM
            return False, DataQuality.LOW

        # 未定义范围的字段，默认有效
        return True, DataQuality.MEDIUM

    @classmethod
    def clean_value(cls, field_name: str, value: float, default: float = 0) -> Tuple[float, float]:
        """
        清洗数值，返回有效值和质量系数

        Args:
            field_name: 字段名
            value: 原始数值
            default: 无效时的默认值

        Returns:
            Tuple[float, float]: (清洗后的值, 质量系数)
        """
        is_valid, quality = cls.validate_value(field_name, value)
        if is_valid:
            return value, quality
        return default, quality

    @classmethod
    def validate_date(cls, date_str: str, min_days_ago: int = 0, max_days_ago: int = 365) -> Tuple[bool, float]:
        """
        校验日期是否在合理范围内

        Args:
            date_str: 日期字符串
            min_days_ago: 最小天数前
            max_days_ago: 最大天数前

        Returns:
            Tuple[bool, float]: (是否有效, 质量系数)
        """
        try:
            if not date_str:
                return False, DataQuality.INVALID

            date = pd.to_datetime(date_str)
            now = datetime.now()
            days_diff = (now - date).days

            if min_days_ago <= days_diff <= max_days_ago:
                return True, DataQuality.HIGH
            elif days_diff < min_days_ago:
                # 未来日期，无效
                return False, DataQuality.INVALID
            elif days_diff <= max_days_ago * 2:
                # 稍微超出范围，降低质量
                return True, DataQuality.MEDIUM
            else:
                # 太久远的数据
                return True, DataQuality.LOW
        except:
            return False, DataQuality.INVALID


class DataCleaner:
    """数据清洗器"""

    @staticmethod
    def clean_numeric(value: Any, default: float = 0) -> float:
        """清洗数值型数据"""
        if value is None:
            return default
        if isinstance(value, str):
            try:
                value = value.replace(',', '').replace('%', '').strip()
                return float(value) if value else default
            except:
                return default
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return default
            return float(value)
        return default

    @staticmethod
    def clean_percentage(value: Any, default: float = 0) -> float:
        """清洗百分比数据"""
        val = DataCleaner.clean_numeric(value, default)
        # 如果值大于100，可能是百分比形式存储的
        if abs(val) > 100:
            val = val / 100
        return val

    @staticmethod
    def clean_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> pd.DataFrame:
        """清洗DataFrame"""
        if df.empty:
            return df

        # 替换无穷值为NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # 如果有必需列，删除缺失必需列的行
        if required_columns:
            df = df.dropna(subset=required_columns)

        return df

    @staticmethod
    def remove_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 3) -> pd.Series:
        """
        移除异常值

        Args:
            series: 数据序列
            method: 方法 ('iqr' 或 'zscore')
            threshold: 阈值
        """
        if series.empty:
            return series

        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return series.clip(lower, upper)
        elif method == 'zscore':
            mean = series.mean()
            std = series.std()
            if std == 0:
                return series
            z_scores = (series - mean) / std
            return series[abs(z_scores) <= threshold]

        return series


# ==================== 数据类定义 ====================

@dataclass
class FinancialReport:
    """财报数据"""
    code: str
    name: str
    report_date: str
    report_type: str = '季报'

    # 核心财务指标
    revenue: float = 0
    revenue_yoy: float = 0
    net_profit: float = 0
    net_profit_yoy: float = 0
    gross_margin: float = 0
    net_margin: float = 0
    roe: float = 0
    roa: float = 0

    # 每股指标
    eps: float = 0
    bvps: float = 0
    cfps: float = 0

    # 资产负债指标
    total_assets: float = 0
    total_liabilities: float = 0
    asset_liability_ratio: float = 0
    current_ratio: float = 0
    quick_ratio: float = 0

    # 现金流指标
    operating_cash_flow: float = 0
    free_cash_flow: float = 0

    # 估值指标
    pe_ratio: float = 0
    pb_ratio: float = 0
    ps_ratio: float = 0
    total_market_cap: float = 0

    # 成长性指标
    revenue_cagr_3y: float = 0
    profit_cagr_3y: float = 0
    revenue_qoq: float = 0
    profit_qoq: float = 0

    # 行业信息
    industry: str = ''

    # 评分（带权重）
    quality_score: float = 0
    growth_score: float = 0
    valuation_score: float = 0
    total_score: float = 0

    # 数据质量
    data_quality: float = 1.0
    field_quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class NorthBoundFlow:
    """北向资金数据"""
    code: str
    name: str
    trade_date: str

    buy_amount: float = 0
    sell_amount: float = 0
    net_buy: float = 0

    hold_shares: float = 0
    hold_value: float = 0
    hold_ratio: float = 0

    net_buy_5d: float = 0
    net_buy_20d: float = 0
    hold_change_5d: float = 0
    hold_change_20d: float = 0

    continuous_buy_days: int = 0
    continuous_sell_days: int = 0

    flow_score: float = 0

    # 数据质量
    data_quality: float = 1.0
    is_stale: bool = False  # 数据是否过时


@dataclass
class DragonTigerList:
    """龙虎榜数据"""
    code: str
    name: str
    trade_date: str
    reason: str

    close_price: float = 0
    change_pct: float = 0
    turnover_rate: float = 0

    buy_amount: float = 0
    buy_seats: List[Dict] = field(default_factory=list)

    sell_amount: float = 0
    sell_seats: List[Dict] = field(default_factory=list)

    net_buy: float = 0

    institution_buy: float = 0
    institution_sell: float = 0
    hot_money_buy: float = 0
    hot_money_sell: float = 0

    next_1d_change: float = 0
    next_5d_change: float = 0

    dragon_score: float = 0

    # 数据质量
    data_quality: float = 1.0


# ==================== 缓存管理 ====================

class DataCache:
    """数据缓存管理器"""

    def __init__(self, cache_dir: str = None, expire_seconds: int = 3600):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '../../data/cache')
        self.expire_seconds = expire_seconds
        self._memory_cache = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, prefix: str, *args) -> str:
        key_str = f"{prefix}_{'_'.join(str(a) for a in args)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, prefix: str, *args) -> Optional[Any]:
        key = self._get_cache_key(prefix, *args)

        if key in self._memory_cache:
            cached_data, cached_time = self._memory_cache[key]
            if (datetime.now() - cached_time).total_seconds() < self.expire_seconds:
                return cached_data

        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data, cached_time = pickle.load(f)
                if (datetime.now() - cached_time).total_seconds() < self.expire_seconds:
                    self._memory_cache[key] = (cached_data, cached_time)
                    return cached_data
            except:
                pass

        return None

    def set(self, prefix: str, data: Any, *args) -> None:
        key = self._get_cache_key(prefix, *args)
        now = datetime.now()
        self._memory_cache[key] = (data, now)

        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((data, now), f)
        except:
            pass

    def clear(self, prefix: str = None) -> None:
        if prefix:
            keys_to_remove = [k for k in self._memory_cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._memory_cache[k]
        else:
            self._memory_cache.clear()


# ==================== 基本面数据客户端 ====================

class FundamentalDataClient:
    """基本面数据客户端（带数据清洗）"""

    def __init__(self, use_cache: bool = True, cache_expire: int = 3600):
        self._request_interval = 1.5
        self._last_request_time = 0
        self._max_retries = 3
        self._use_cache = use_cache
        self._cache = DataCache(expire_seconds=cache_expire) if use_cache else None
        self._cleaner = DataCleaner()

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed + random.uniform(0.1, 0.3))
        self._last_request_time = time.time()

    def _retry_request(self, func, *args, **kwargs) -> Optional[Any]:
        last_error = None
        for attempt in range(self._max_retries):
            try:
                self._wait_for_rate_limit()
                result = func(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                last_error = e
                wait_time = (attempt + 1) * 2 + random.uniform(0.5, 1.5)
                print(f"请求失败，{wait_time:.1f}秒后重试 (尝试 {attempt + 1}/{self._max_retries}): {e}")
                time.sleep(wait_time)

        if last_error:
            print(f"请求最终失败: {last_error}")
        return None

    def _get_from_cache(self, cache_type: str, key: str) -> Optional[Any]:
        if self._use_cache and self._cache:
            return self._cache.get(cache_type, key)
        return None

    def _set_to_cache(self, cache_type: str, key: str, data: Any) -> None:
        if self._use_cache and self._cache:
            self._cache.set(cache_type, data, key)

    # ==================== 财报数据 ====================

    def get_financial_report(self, code: str) -> Optional[FinancialReport]:
        """获取股票财务报告（带数据清洗）"""
        cached = self._get_from_cache('financial', code)
        if cached:
            return cached

        try:
            stock_name = self._get_stock_name(code)
            industry = self._get_industry(code)

            financial_data = {
                'code': code,
                'name': stock_name,
                'report_date': '',
                'report_type': '季报',
                'industry': industry,
                'field_quality': {},
            }

            # 获取业绩报表数据
            self._fetch_earnings_data(code, financial_data)

            # 获取财务报表数据
            self._fetch_financial_statements(code, financial_data)

            # 获取个股基本信息
            self._fetch_stock_info(code, financial_data)

            # 计算成长性指标
            self._calculate_growth_metrics(code, financial_data)

            # 清洗数据并计算质量
            financial_data = self._clean_financial_data(financial_data, industry)

            report = FinancialReport(
                code=code,
                name=stock_name,
                report_date=financial_data.get('report_date', ''),
                report_type=financial_data.get('report_type', '季报'),
                revenue=financial_data.get('revenue', 0),
                revenue_yoy=financial_data.get('revenue_yoy', 0),
                net_profit=financial_data.get('net_profit', 0),
                net_profit_yoy=financial_data.get('net_profit_yoy', 0),
                gross_margin=financial_data.get('gross_margin', 0),
                net_margin=financial_data.get('net_margin', 0),
                roe=financial_data.get('roe', 0),
                roa=financial_data.get('roa', 0),
                eps=financial_data.get('eps', 0),
                bvps=financial_data.get('bvps', 0),
                cfps=financial_data.get('cfps', 0),
                total_assets=financial_data.get('total_assets', 0),
                total_liabilities=financial_data.get('total_liabilities', 0),
                asset_liability_ratio=financial_data.get('asset_liability_ratio', 0),
                current_ratio=financial_data.get('current_ratio', 0),
                quick_ratio=financial_data.get('quick_ratio', 0),
                operating_cash_flow=financial_data.get('operating_cash_flow', 0),
                free_cash_flow=financial_data.get('free_cash_flow', 0),
                pe_ratio=financial_data.get('pe_ratio', 0),
                pb_ratio=financial_data.get('pb_ratio', 0),
                ps_ratio=financial_data.get('ps_ratio', 0),
                total_market_cap=financial_data.get('total_market_cap', 0),
                revenue_cagr_3y=financial_data.get('revenue_cagr_3y', 0),
                profit_cagr_3y=financial_data.get('profit_cagr_3y', 0),
                revenue_qoq=financial_data.get('revenue_qoq', 0),
                profit_qoq=financial_data.get('profit_qoq', 0),
                industry=industry,
                field_quality=financial_data.get('field_quality', {}),
            )

            # 计算评分（带权重）
            report.quality_score = self._calculate_quality_score(report)
            report.growth_score = self._calculate_growth_score(report)
            report.valuation_score = self._calculate_valuation_score(report)
            report.data_quality = financial_data.get('data_quality', 1.0)

            # 综合评分（根据数据质量调整权重）
            report.total_score = self._calculate_weighted_total_score(report)

            self._set_to_cache('financial', code, report)
            return report

        except Exception as e:
            print(f"获取财报数据失败 {code}: {e}")
            return None

    def _clean_financial_data(self, data: Dict, industry: str = None) -> Dict:
        """清洗财务数据"""
        field_quality = {}

        # 需要清洗的数值字段
        numeric_fields = [
            'revenue', 'revenue_yoy', 'net_profit', 'net_profit_yoy',
            'gross_margin', 'net_margin', 'roe', 'roa',
            'eps', 'bvps', 'cfps',
            'asset_liability_ratio', 'current_ratio', 'quick_ratio',
            'operating_cash_flow', 'pe_ratio', 'pb_ratio',
        ]

        for field in numeric_fields:
            value = data.get(field)
            if value is not None:
                # 清洗数值
                clean_val = DataCleaner.clean_numeric(value, 0)
                # 校验并获取质量系数
                _, quality = DataValidator.validate_value(field, clean_val, industry)
                data[field] = clean_val
                field_quality[field] = quality

        # 计算整体数据质量
        if field_quality:
            data['data_quality'] = sum(field_quality.values()) / len(field_quality)
        else:
            data['data_quality'] = DataQuality.LOW

        data['field_quality'] = field_quality
        return data

    def _get_stock_name(self, code: str) -> str:
        try:
            self._wait_for_rate_limit()
            df = ak.stock_individual_info_em(symbol=code)
            if not df.empty:
                name_row = df[df['item'] == '股票简称']
                if not name_row.empty:
                    return name_row['value'].values[0]
        except:
            pass
        return code

    def _get_industry(self, code: str) -> str:
        try:
            self._wait_for_rate_limit()
            df = ak.stock_individual_info_em(symbol=code)
            if not df.empty:
                industry_row = df[df['item'] == '行业']
                if not industry_row.empty:
                    return str(industry_row['value'].values[0])
        except:
            pass
        return ''

    def _get_equity_from_balance(self, row) -> float:
        """从资产负债表行中提取归母股东权益（兼容多种列名）"""
        equity_keys = [
            '归属于母公司的股东权益合计',
            '归属于母公司股东的权益',
            '归属于母公司股东权益合计',
            '所有者权益合计',
            '所有者权益(或股东权益)合计',
            '股东权益',
            '所有者权益',
        ]
        for key in equity_keys:
            val = row.get(key, None)
            if val is not None and str(val) not in ('nan', 'None', ''):
                cleaned = DataCleaner.clean_numeric(val)
                if cleaned and cleaned > 0:
                    return cleaned
        return 0

    def _find_yoy_match(self, profit_df, report_date: str):
        """在利润表中找到同一报告期的上一年数据
        报告日格式: YYYYMMDD → Q1=0331, H1=0630, Q3=0930, 年度=1231
        """
        if not report_date or len(report_date) < 8:
            return None

        period = report_date[4:]  # e.g. '0331', '0630', '0930', '1231'
        year = int(report_date[:4])
        prev_year_key = f"{year - 1}{period}"

        for i in range(1, len(profit_df)):
            row_date = str(profit_df.iloc[i].get('报告日', ''))
            if row_date == prev_year_key:
                return profit_df.iloc[i]

        return None

    def _fetch_earnings_data(self, code: str, financial_data: Dict) -> None:
        """获取业绩报表数据 — 新浪源优先，东财源备用"""
        # 方案1: 新浪利润表（当前网络可用）
        try:
            self._wait_for_rate_limit()
            profit_df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
            if profit_df is not None and not profit_df.empty and len(profit_df) >= 1:
                latest = profit_df.iloc[0]

                report_date = str(latest.get('报告日', ''))
                financial_data['report_date'] = report_date

                revenue = DataCleaner.clean_numeric(latest.get('营业收入', 0))
                net_profit = DataCleaner.clean_numeric(latest.get('净利润', 0))
                financial_data['revenue'] = revenue / 1e8
                financial_data['net_profit'] = net_profit / 1e8

                # 同比增长: 找同一报告期的上一年数据（Q1→Q1, H1→H1, Q3→Q3, 年度→年度）
                yoy_match = self._find_yoy_match(profit_df, report_date)
                if yoy_match is not None:
                    prev_revenue = DataCleaner.clean_numeric(yoy_match.get('营业收入', 0))
                    prev_profit = DataCleaner.clean_numeric(yoy_match.get('净利润', 0))
                    if prev_revenue and prev_revenue != 0:
                        financial_data['revenue_yoy'] = (revenue - prev_revenue) / abs(prev_revenue) * 100
                    if prev_profit and prev_profit != 0:
                        financial_data['net_profit_yoy'] = (net_profit - prev_profit) / abs(prev_profit) * 100

                # 从利润表计算毛利率
                op_cost = DataCleaner.clean_numeric(latest.get('营业成本', 0))
                if revenue and revenue != 0 and op_cost:
                    financial_data['gross_margin'] = (revenue - op_cost) / revenue * 100
                if revenue and revenue != 0 and net_profit:
                    financial_data['net_margin'] = net_profit / revenue * 100

                return  # 新浪成功，不尝试东财
        except Exception as e:
            print(f"新浪利润表获取失败 {code}: {e}")

        # 方案2: 东财业绩报表（当前网络不可用，作为备用）
        try:
            dates = ['20260331', '20251231', '20250930', '20250630']
            for date in dates:
                try:
                    self._wait_for_rate_limit()
                    df = ak.stock_yjbb_em(date=date)
                    if df.empty:
                        continue

                    stock_data = df[df['股票代码'] == code]
                    if stock_data.empty:
                        continue

                    row = stock_data.iloc[0]
                    financial_data['report_date'] = date
                    financial_data['eps'] = DataCleaner.clean_numeric(row.get('每股收益', 0))
                    financial_data['revenue'] = DataCleaner.clean_numeric(row.get('营业总收入-营业总收入', 0)) / 1e8
                    financial_data['revenue_yoy'] = DataCleaner.clean_numeric(row.get('营业总收入-同比增长', 0))
                    financial_data['net_profit'] = DataCleaner.clean_numeric(row.get('净利润-净利润', 0)) / 1e8
                    financial_data['net_profit_yoy'] = DataCleaner.clean_numeric(row.get('净利润-同比增长', 0))
                    financial_data['bvps'] = DataCleaner.clean_numeric(row.get('每股净资产', 0))
                    financial_data['roe'] = DataCleaner.clean_numeric(row.get('净资产收益率', 0))
                    financial_data['cfps'] = DataCleaner.clean_numeric(row.get('每股经营现金流量', 0))
                    financial_data['gross_margin'] = DataCleaner.clean_numeric(row.get('销售毛利率', 0))
                    break
                except:
                    continue
        except Exception as e:
            print(f"东财业绩报表获取失败 {code}: {e}")

    def _fetch_financial_statements(self, code: str, financial_data: Dict) -> None:
        """获取财务报表数据"""
        try:
            self._wait_for_rate_limit()
            profit_df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
            if not profit_df.empty and len(profit_df) >= 1:
                latest = profit_df.iloc[0]
                if financial_data.get('revenue', 0) == 0:
                    financial_data['revenue'] = DataCleaner.clean_numeric(latest.get('营业收入', 0)) / 1e8
                if financial_data.get('net_profit', 0) == 0:
                    financial_data['net_profit'] = DataCleaner.clean_numeric(latest.get('净利润', 0)) / 1e8

                revenue = financial_data.get('revenue', 0)
                net_profit = financial_data.get('net_profit', 0)
                if revenue > 0:
                    financial_data['net_margin'] = net_profit / revenue * 100

                report_date = str(latest.get('报告日', ''))
                yoy_match = self._find_yoy_match(profit_df, report_date)
                if yoy_match is not None:
                    prev_revenue = DataCleaner.clean_numeric(yoy_match.get('营业收入', 0)) / 1e8
                    prev_profit = DataCleaner.clean_numeric(yoy_match.get('净利润', 0)) / 1e8
                    if prev_revenue > 0 and financial_data.get('revenue_yoy', 0) == 0:
                        financial_data['revenue_yoy'] = (revenue - prev_revenue) / prev_revenue * 100
                    if prev_profit > 0 and financial_data.get('net_profit_yoy', 0) == 0:
                        financial_data['net_profit_yoy'] = (net_profit - prev_profit) / prev_profit * 100
        except Exception as e:
            print(f"获取利润表失败 {code}: {e}")

        try:
            self._wait_for_rate_limit()
            balance_df = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
            if not balance_df.empty:
                latest = balance_df.iloc[0]
                financial_data['total_assets'] = DataCleaner.clean_numeric(latest.get('资产总计', 0)) / 1e8
                financial_data['total_liabilities'] = DataCleaner.clean_numeric(latest.get('负债合计', 0)) / 1e8

                if financial_data['total_assets'] > 0:
                    financial_data['asset_liability_ratio'] = (
                        financial_data['total_liabilities'] / financial_data['total_assets'] * 100
                    )

                current_assets = DataCleaner.clean_numeric(latest.get('流动资产', 0))
                current_liabilities = DataCleaner.clean_numeric(latest.get('流动负债', 0))
                if current_liabilities > 0:
                    financial_data['current_ratio'] = current_assets / current_liabilities
        except Exception as e:
            print(f"获取资产负债表失败 {code}: {e}")

        try:
            self._wait_for_rate_limit()
            cashflow_df = ak.stock_financial_report_sina(stock=code, symbol="现金流量表")
            if not cashflow_df.empty:
                latest = cashflow_df.iloc[0]
                financial_data['operating_cash_flow'] = DataCleaner.clean_numeric(
                    latest.get('经营活动产生的现金流量净额', 0)
                ) / 1e8
        except Exception as e:
            print(f"获取现金流表失败 {code}: {e}")

    def _fetch_stock_info(self, code: str, financial_data: Dict) -> None:
        """获取股票基本信息 — 新浪源优先，东财备用"""
        # 方案1: 从新浪财务数据推算
        try:
            self._wait_for_rate_limit()
            balance_df = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
            if balance_df is not None and not balance_df.empty:
                latest = balance_df.iloc[0]
                total_equity = self._get_equity_from_balance(latest)
                total_assets = DataCleaner.clean_numeric(latest.get('资产总计', 0))
                goodwill = DataCleaner.clean_numeric(latest.get('商誉', 0))

                financial_data['total_assets'] = total_assets / 1e8
                if total_equity > 0:
                    financial_data['bvps'] = financial_data.get('bvps', 0) or total_equity
                    # 计算ROE: 净利润 / 归母股东权益
                    net_profit = financial_data.get('net_profit', 0)
                    if net_profit and total_equity:
                        # 净利润单位是亿元，权益单位是元 → 需要统一
                        roe_val = (net_profit * 1e8) / total_equity * 100
                        if financial_data.get('roe', 0) == 0:
                            financial_data['roe'] = roe_val

                # 商誉占净资产比
                if goodwill and total_equity and total_equity > 0:
                    financial_data['goodwill_ratio'] = goodwill / total_equity * 100
        except Exception as e:
            print(f"新浪资产负债信息获取失败 {code}: {e}")

        # 方案2: 从腾讯行情获取市值
        try:
            from app.market import get_market_client
            client = get_market_client()
            quote = client.get_stock_price(code)
            if quote:
                price = quote.get('price', 0)
                # 从行情数据估算市值（需要总股本，简化处理）
                if price > 0:
                    financial_data['current_price'] = price
        except Exception:
            pass

        # 方案3: 东财个股信息（备用，当前不可用）
        try:
            self._wait_for_rate_limit()
            df = ak.stock_individual_info_em(symbol=code)
            if not df.empty:
                for _, row in df.iterrows():
                    item = row['item']
                    value = row['value']
                    if item == '总市值':
                        financial_data['total_market_cap'] = DataCleaner.clean_numeric(value) / 1e8
        except Exception:
            pass

    def _calculate_growth_metrics(self, code: str, financial_data: Dict) -> None:
        """计算成长性指标 — 新浪利润表多年数据"""
        # 方案1: 从新浪利润表获取多年数据计算CAGR
        try:
            self._wait_for_rate_limit()
            profit_df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
            if profit_df is not None and not profit_df.empty and len(profit_df) >= 4:
                # 按时间排序（最新在前）
                all_data = []
                for i in range(min(len(profit_df), 8)):
                    row = profit_df.iloc[i]
                    rev = DataCleaner.clean_numeric(row.get('营业收入', 0))
                    profit = DataCleaner.clean_numeric(row.get('净利润', 0))
                    date_str = str(row.get('报告日', ''))
                    all_data.append({
                        'date': date_str,
                        'revenue': rev,
                        'net_profit': profit,
                    })

                if len(all_data) >= 4:
                    recent_revenue = all_data[0].get('revenue', 0)
                    old_revenue = all_data[-1].get('revenue', 0)
                    if old_revenue > 0 and recent_revenue > 0:
                        years = len(all_data) / 4
                        financial_data['revenue_cagr_3y'] = (
                            (recent_revenue / old_revenue) ** (1 / years) - 1
                        ) * 100

                    recent_profit = all_data[0].get('net_profit', 0)
                    old_profit = all_data[-1].get('net_profit', 0)
                    if old_profit > 0 and recent_profit > 0:
                        years = len(all_data) / 4
                        financial_data['profit_cagr_3y'] = (
                            (recent_profit / old_profit) ** (1 / years) - 1
                        ) * 100
                return  # 新浪成功
        except Exception as e:
            print(f"新浪成长指标计算失败 {code}: {e}")

        # 方案2: 东财业绩报表（备用）
        try:
            self._wait_for_rate_limit()
            all_data = []
            for date in ['20260331', '20251231', '20240930', '20240630', '20240331', '20231231']:
                try:
                    df = ak.stock_yjbb_em(date=date)
                    stock_data = df[df['股票代码'] == code]
                    if not stock_data.empty:
                        row = stock_data.iloc[0]
                        all_data.append({
                            'date': date,
                            'revenue': DataCleaner.clean_numeric(row.get('营业总收入-营业总收入', 0)),
                            'net_profit': DataCleaner.clean_numeric(row.get('净利润-净利润', 0)),
                        })
                except:
                    continue

            if len(all_data) >= 4:
                recent_revenue = all_data[0].get('revenue', 0)
                old_revenue = all_data[-1].get('revenue', 0)
                if old_revenue > 0 and recent_revenue > 0:
                    years = len(all_data) / 4
                    financial_data['revenue_cagr_3y'] = (
                        (recent_revenue / old_revenue) ** (1 / years) - 1
                    ) * 100

                recent_profit = all_data[0].get('net_profit', 0)
                old_profit = all_data[-1].get('net_profit', 0)
                if old_profit > 0 and recent_profit > 0:
                    years = len(all_data) / 4
                    financial_data['profit_cagr_3y'] = (
                        (recent_profit / old_profit) ** (1 / years) - 1
                    ) * 100
        except Exception as e:
            print(f"东财成长指标计算失败 {code}: {e}")

    def _calculate_quality_score(self, report: FinancialReport) -> float:
        """计算财务质量评分（考虑行业特性）"""
        score = 50
        industry = report.industry

        # 银行、保险、证券行业特殊处理
        if industry in ['银行', '保险', '证券']:
            # 金融行业不考核毛利率、流动比率
            # 重点考核ROE和资产质量
            if report.roe > 15:
                score += 25
            elif report.roe > 10:
                score += 15
            elif report.roe > 5:
                score += 10

            # 资产负债率对金融行业是正常的
            if 85 < report.asset_liability_ratio < 95:
                score += 10  # 正常范围

            # 净利率
            if report.net_margin > 20:
                score += 15
            elif report.net_margin > 10:
                score += 10

            return max(0, min(100, score))

        # 非金融行业正常评分
        # ROE评分
        if report.roe > 25:
            score += 25
        elif report.roe > 20:
            score += 20
        elif report.roe > 15:
            score += 15
        elif report.roe > 10:
            score += 10
        elif report.roe > 5:
            score += 5
        else:
            score -= 10

        # 毛利率评分
        if report.gross_margin > 50:
            score += 15
        elif report.gross_margin > 35:
            score += 10
        elif report.gross_margin > 20:
            score += 5
        elif report.gross_margin < 10 and report.gross_margin != 0:
            score -= 5

        # 净利率评分
        if report.net_margin > 20:
            score += 10
        elif report.net_margin > 10:
            score += 7
        elif report.net_margin > 5:
            score += 3
        elif report.net_margin < 0:
            score -= 10

        # 资产负债率评分
        if 30 < report.asset_liability_ratio < 50:
            score += 10
        elif 20 < report.asset_liability_ratio <= 30 or 50 < report.asset_liability_ratio <= 60:
            score += 5
        elif report.asset_liability_ratio > 80:
            score -= 15
        elif report.asset_liability_ratio > 70:
            score -= 8

        # 流动比率评分
        if report.current_ratio > 2:
            score += 5
        elif report.current_ratio > 1.5:
            score += 3
        elif report.current_ratio < 0.8 and report.current_ratio > 0:
            score -= 5

        # 现金流评分
        if report.operating_cash_flow > 0:
            score += 5
        else:
            score -= 8

        return max(0, min(100, score))

    def _calculate_growth_score(self, report: FinancialReport) -> float:
        """计算成长性评分"""
        score = 50

        # 营收增长评分
        if report.revenue_yoy > 50:
            score += 25
        elif report.revenue_yoy > 30:
            score += 20
        elif report.revenue_yoy > 15:
            score += 15
        elif report.revenue_yoy > 0:
            score += 5
        else:
            score -= 15

        # 利润增长评分
        if report.net_profit_yoy > 50:
            score += 25
        elif report.net_profit_yoy > 30:
            score += 20
        elif report.net_profit_yoy > 15:
            score += 15
        elif report.net_profit_yoy > 0:
            score += 5
        else:
            score -= 15

        # 3年复合增长率加分
        if report.revenue_cagr_3y > 30:
            score += 10
        elif report.revenue_cagr_3y > 15:
            score += 5

        if report.profit_cagr_3y > 30:
            score += 10
        elif report.profit_cagr_3y > 15:
            score += 5

        return max(0, min(100, score))

    def _calculate_valuation_score(self, report: FinancialReport) -> float:
        """计算估值评分"""
        score = 50

        if report.pe_ratio > 0:
            if report.pe_ratio < 15:
                score += 25
            elif report.pe_ratio < 25:
                score += 18
            elif report.pe_ratio < 40:
                score += 10
            elif report.pe_ratio < 60:
                score += 3
            elif report.pe_ratio > 100:
                score -= 15
        else:
            score -= 10

        if report.pb_ratio > 0:
            if report.pb_ratio < 1.5:
                score += 15
            elif report.pb_ratio < 3:
                score += 10
            elif report.pb_ratio < 5:
                score += 5
            elif report.pb_ratio > 10:
                score -= 10

        if report.ps_ratio > 0:
            if report.ps_ratio < 2:
                score += 10
            elif report.ps_ratio < 5:
                score += 5
            elif report.ps_ratio > 15:
                score -= 5

        return max(0, min(100, score))

    def _calculate_weighted_total_score(self, report: FinancialReport) -> float:
        """计算加权综合评分（考虑数据质量）"""
        # 基础权重
        base_weights = {
            'quality': 0.4,
            'growth': 0.35,
            'valuation': 0.25,
        }

        # 根据数据质量调整权重
        quality_factor = report.data_quality

        # 数据质量低时，降低成长性权重（成长数据波动大）
        if quality_factor < DataQuality.MEDIUM:
            adjusted_weights = {
                'quality': base_weights['quality'] * 1.2,
                'growth': base_weights['growth'] * 0.7,
                'valuation': base_weights['valuation'] * 1.1,
            }
        else:
            adjusted_weights = base_weights

        # 归一化权重
        total_weight = sum(adjusted_weights.values())
        for k in adjusted_weights:
            adjusted_weights[k] /= total_weight

        # 计算加权分数
        score = (
            report.quality_score * adjusted_weights['quality'] +
            report.growth_score * adjusted_weights['growth'] +
            report.valuation_score * adjusted_weights['valuation']
        )

        return max(0, min(100, score))

    # ==================== 北向资金数据 ====================

    def get_north_bound_flow(self, code: str) -> Optional[NorthBoundFlow]:
        """获取股票北向资金数据（带数据清洗）"""
        cached = self._get_from_cache('north_bound', code)
        if cached:
            return cached

        try:
            self._wait_for_rate_limit()
            stock_name = self._get_stock_name(code)

            df = ak.stock_hsgt_individual_em(symbol=code)
            if df.empty:
                return None

            # 清洗数据：按日期排序，取最新有效数据
            df = df.sort_values('持股日期', ascending=False)
            df = DataCleaner.clean_dataframe(df)

            # 找到最近的有效数据（不超过30天）
            latest_valid = None
            for idx, row in df.iterrows():
                date_str = str(row.get('持股日期', ''))
                is_valid, quality = DataValidator.validate_date(date_str, 0, 30)
                if is_valid:
                    latest_valid = row
                    break

            if latest_valid is None:
                # 如果没有最近数据，取第一条但标记为过时
                latest_valid = df.iloc[0]
                is_stale = True
            else:
                is_stale = False

            flow = NorthBoundFlow(
                code=code,
                name=stock_name,
                trade_date=str(latest_valid.get('持股日期', '')),
                hold_shares=DataCleaner.clean_numeric(latest_valid.get('持股数量', 0)),
                hold_value=DataCleaner.clean_numeric(latest_valid.get('持股市值', 0)) / 1e8,
                hold_ratio=DataCleaner.clean_numeric(latest_valid.get('持股数量占A股百分比', 0)),
                net_buy=DataCleaner.clean_numeric(latest_valid.get('今日增持资金', 0)) / 1e8,
                is_stale=is_stale,
            )

            # 计算趋势数据（只使用有效数据）
            valid_df = df.head(30)  # 取最近30条
            if len(valid_df) >= 5:
                recent_5 = valid_df.head(5)
                first_hold = recent_5.iloc[-1]['持股数量']
                last_hold = recent_5.iloc[0]['持股数量']

                # 清洗异常变化
                if first_hold > 0:
                    change_5d = (last_hold - first_hold) / first_hold * 100
                    # 校验变化率是否合理
                    _, quality = DataValidator.validate_value('hold_change_5d', change_5d)
                    if quality >= DataQuality.MEDIUM:
                        flow.hold_change_5d = change_5d
                    else:
                        flow.hold_change_5d = 0  # 异常值置零

                flow.net_buy_5d = DataCleaner.clean_numeric(sum(recent_5['今日增持资金'].fillna(0))) / 1e8

                # 计算连续买入天数
                net_buys = recent_5['今日增持资金'].fillna(0).tolist()
                continuous_buy = 0
                for nb in net_buys:
                    if nb > 0:
                        continuous_buy += 1
                    else:
                        break
                flow.continuous_buy_days = continuous_buy

            if len(valid_df) >= 20:
                recent_20 = valid_df.head(20)
                first_hold = recent_20.iloc[-1]['持股数量']
                last_hold = recent_20.iloc[0]['持股数量']
                if first_hold > 0:
                    change_20d = (last_hold - first_hold) / first_hold * 100
                    _, quality = DataValidator.validate_value('hold_change_20d', change_20d)
                    if quality >= DataQuality.MEDIUM:
                        flow.hold_change_20d = change_20d

                flow.net_buy_20d = DataCleaner.clean_numeric(sum(recent_20['今日增持资金'].fillna(0))) / 1e8

            # 计算数据质量
            flow.data_quality = DataQuality.LOW if is_stale else DataQuality.HIGH

            # 计算评分
            flow.flow_score = self._calculate_north_flow_score(flow)

            self._set_to_cache('north_bound', code, flow)
            return flow

        except Exception as e:
            print(f"获取北向资金失败 {code}: {e}")
            return None

    def _calculate_north_flow_score(self, flow: NorthBoundFlow) -> float:
        """计算北向资金评分"""
        # 如果数据过时，降低基础分
        if flow.is_stale:
            return 50  # 返回中性分

        score = 50

        # 持股比例评分
        if flow.hold_ratio > 15:
            score += 25
        elif flow.hold_ratio > 10:
            score += 20
        elif flow.hold_ratio > 5:
            score += 15
        elif flow.hold_ratio > 2:
            score += 10
        elif flow.hold_ratio > 0:
            score += 5

        # 持股变化评分（只考虑合理范围内的变化）
        if abs(flow.hold_change_5d) < 100:  # 排除异常变化
            if flow.hold_change_5d > 10:
                score += 25
            elif flow.hold_change_5d > 5:
                score += 20
            elif flow.hold_change_5d > 2:
                score += 15
            elif flow.hold_change_5d > 0:
                score += 10
            elif flow.hold_change_5d < -5:
                score -= 15
            elif flow.hold_change_5d < -2:
                score -= 10

        # 净买入评分
        if flow.net_buy_5d > 2:
            score += 15
        elif flow.net_buy_5d > 0.5:
            score += 10
        elif flow.net_buy_5d > 0:
            score += 5
        elif flow.net_buy_5d < -2:
            score -= 15
        elif flow.net_buy_5d < -0.5:
            score -= 10

        # 连续买入加分
        if flow.continuous_buy_days >= 5:
            score += 10
        elif flow.continuous_buy_days >= 3:
            score += 5

        return max(0, min(100, score))

    def get_north_bound_top_stocks(self, limit: int = 50) -> List[Dict]:
        """获取北向资金持股TOP股票"""
        try:
            self._wait_for_rate_limit()
            result = []

            df = ak.stock_zh_a_spot_em()
            if df.empty:
                return []

            sample_codes = df['代码'].head(100).tolist()

            for code in sample_codes[:20]:
                try:
                    flow = self.get_north_bound_flow(code)
                    if flow and flow.hold_ratio > 0 and not flow.is_stale:
                        result.append({
                            'code': code,
                            'name': flow.name,
                            'hold_shares': flow.hold_shares,
                            'hold_value': flow.hold_value,
                            'hold_ratio': flow.hold_ratio,
                            'hold_change_5d': flow.hold_change_5d,
                            'flow_score': flow.flow_score,
                            'data_quality': flow.data_quality,
                        })
                except:
                    continue

            result.sort(key=lambda x: x['hold_ratio'], reverse=True)
            return result[:limit]

        except Exception as e:
            print(f"获取北向持股TOP失败: {e}")
            return []

    def get_north_bound_market_flow(self) -> Dict[str, Any]:
        """获取北向资金整体流向"""
        try:
            self._wait_for_rate_limit()
            result = {}

            df_sh = ak.stock_hsgt_hist_em(symbol='沪股通')
            if not df_sh.empty:
                latest_sh = df_sh.iloc[0]
                result['sh_net_buy'] = DataCleaner.clean_numeric(latest_sh.get('当日成交净买额', 0))
                result['sh_total_buy'] = DataCleaner.clean_numeric(latest_sh.get('买入成交额', 0))
                result['sh_total_sell'] = DataCleaner.clean_numeric(latest_sh.get('卖出成交额', 0))

            df_sz = ak.stock_hsgt_hist_em(symbol='深股通')
            if not df_sz.empty:
                latest_sz = df_sz.iloc[0]
                result['sz_net_buy'] = DataCleaner.clean_numeric(latest_sz.get('当日成交净买额', 0))
                result['sz_total_buy'] = DataCleaner.clean_numeric(latest_sz.get('买入成交额', 0))
                result['sz_total_sell'] = DataCleaner.clean_numeric(latest_sz.get('卖出成交额', 0))

            result['total_net_buy'] = result.get('sh_net_buy', 0) + result.get('sz_net_buy', 0)
            result['date'] = str(latest_sh.get('日期', '')) if not df_sh.empty else ''

            return result

        except Exception as e:
            print(f"获取北向资金整体流向失败: {e}")
            return {}

    # ==================== 龙虎榜数据 ====================

    def get_dragon_tiger_list(self, code: str = None, days: int = 30) -> List[DragonTigerList]:
        """获取龙虎榜数据（带数据清洗）"""
        cache_key = f"{code}_{days}" if code else f"all_{days}"
        cached = self._get_from_cache('dragon_tiger', cache_key)
        if cached:
            return cached

        try:
            self._wait_for_rate_limit()

            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

            df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)

            if df.empty:
                return []

            if code:
                df = df[df['代码'] == code]

            result = []
            seen = set()

            for _, row in df.iterrows():
                stock_code = str(row['代码'])
                trade_date = str(row.get('上榜日', ''))

                key = f"{stock_code}_{trade_date}"
                if key in seen:
                    continue
                seen.add(key)

                # 清洗数值
                change_pct = DataCleaner.clean_numeric(row.get('涨跌幅', 0))
                turnover_rate = DataCleaner.clean_numeric(row.get('换手率', 0))
                net_buy = DataCleaner.clean_numeric(row.get('龙虎榜净买额', 0))

                # 校验数据有效性
                _, change_quality = DataValidator.validate_value('change_pct', change_pct)
                _, turnover_quality = DataValidator.validate_value('turnover_rate', turnover_rate)

                # 计算整体数据质量
                data_quality = (change_quality + turnover_quality) / 2

                dragon = DragonTigerList(
                    code=stock_code,
                    name=str(row['名称']),
                    trade_date=trade_date,
                    reason=str(row.get('上榜原因', '')),
                    close_price=DataCleaner.clean_numeric(row.get('收盘价', 0)),
                    change_pct=change_pct,
                    turnover_rate=turnover_rate,
                    buy_amount=DataCleaner.clean_numeric(row.get('龙虎榜买入额', 0)),
                    sell_amount=DataCleaner.clean_numeric(row.get('龙虎榜卖出额', 0)),
                    net_buy=net_buy,
                    next_1d_change=DataCleaner.clean_numeric(row.get('上榜后1日', 0)),
                    next_5d_change=DataCleaner.clean_numeric(row.get('上榜后5日', 0)),
                    data_quality=data_quality,
                )

                dragon.dragon_score = self._calculate_dragon_score(dragon)
                result.append(dragon)

            self._set_to_cache('dragon_tiger', cache_key, result)
            return result

        except Exception as e:
            print(f"获取龙虎榜失败: {e}")
            return []

    def _calculate_dragon_score(self, dragon: DragonTigerList) -> float:
        """计算龙虎榜评分"""
        score = 50

        # 涨跌幅评分
        if 9 < dragon.change_pct < 11:
            score += 25
        elif dragon.change_pct > 7:
            score += 20
        elif dragon.change_pct > 5:
            score += 15
        elif dragon.change_pct > 0:
            score += 10
        elif dragon.change_pct < -5:
            score -= 15

        # 净买入评分
        if dragon.buy_amount > 0:
            buy_ratio = dragon.net_buy / dragon.buy_amount
            if buy_ratio > 0.5:
                score += 20
            elif buy_ratio > 0.3:
                score += 15
            elif buy_ratio > 0:
                score += 10
            else:
                score -= 10

        # 换手率评分
        if 5 < dragon.turnover_rate < 15:
            score += 10
        elif 3 < dragon.turnover_rate <= 5 or 15 < dragon.turnover_rate <= 20:
            score += 5
        elif dragon.turnover_rate > 25:
            score -= 10

        # 上榜原因加分
        reason = dragon.reason
        if '涨停' in reason:
            score += 10
        elif '跌停' in reason:
            score -= 10
        if '机构' in reason:
            score += 5

        return max(0, min(100, score))

    def get_hot_money_stocks(self, days: int = 7) -> List[Dict]:
        """获取近期游资活跃股票"""
        try:
            dragons = self.get_dragon_tiger_list(days=days)
            sorted_dragons = sorted(dragons, key=lambda x: x.net_buy, reverse=True)

            result = []
            for dragon in sorted_dragons[:50]:
                result.append({
                    'code': dragon.code,
                    'name': dragon.name,
                    'trade_date': dragon.trade_date,
                    'change_pct': dragon.change_pct,
                    'net_buy': dragon.net_buy,
                    'buy_amount': dragon.buy_amount,
                    'dragon_score': dragon.dragon_score,
                    'reason': dragon.reason,
                    'data_quality': dragon.data_quality,
                })

            return result

        except Exception as e:
            print(f"获取游资活跃股票失败: {e}")
            return []

    # ==================== 综合基本面数据 ====================

    def get_stock_fundamental(self, code: str) -> Dict[str, Any]:
        """获取股票综合基本面数据"""
        result = {
            'code': code,
            'financial': None,
            'north_bound': None,
            'dragon_tiger': [],
            'fundamental_score': 0,
            'data_completeness': 0,
            'data_quality': 0,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 获取财报数据
        try:
            financial = self.get_financial_report(code)
            if financial:
                result['financial'] = {
                    'report_date': financial.report_date,
                    'revenue': financial.revenue,
                    'revenue_yoy': financial.revenue_yoy,
                    'net_profit': financial.net_profit,
                    'net_profit_yoy': financial.net_profit_yoy,
                    'roe': financial.roe,
                    'gross_margin': financial.gross_margin,
                    'net_margin': financial.net_margin,
                    'asset_liability_ratio': financial.asset_liability_ratio,
                    'current_ratio': financial.current_ratio,
                    'operating_cash_flow': financial.operating_cash_flow,
                    'eps': financial.eps,
                    'bvps': financial.bvps,
                    'pe_ratio': financial.pe_ratio,
                    'pb_ratio': financial.pb_ratio,
                    'industry': financial.industry,
                    'quality_score': financial.quality_score,
                    'growth_score': financial.growth_score,
                    'valuation_score': financial.valuation_score,
                    'total_score': financial.total_score,
                    'data_quality': financial.data_quality,
                    'field_quality': financial.field_quality,
                }
        except Exception as e:
            print(f"获取财报数据失败 {code}: {e}")

        # 获取北向资金
        try:
            north = self.get_north_bound_flow(code)
            if north:
                result['north_bound'] = {
                    'hold_ratio': north.hold_ratio,
                    'hold_value': north.hold_value,
                    'hold_change_5d': north.hold_change_5d,
                    'hold_change_20d': north.hold_change_20d,
                    'net_buy_5d': north.net_buy_5d,
                    'net_buy_20d': north.net_buy_20d,
                    'continuous_buy_days': north.continuous_buy_days,
                    'flow_score': north.flow_score,
                    'data_quality': north.data_quality,
                    'is_stale': north.is_stale,
                }
        except Exception as e:
            print(f"获取北向资金失败 {code}: {e}")

        # 获取龙虎榜
        try:
            dragons = self.get_dragon_tiger_list(code, days=30)
            if dragons:
                result['dragon_tiger'] = [{
                    'trade_date': d.trade_date,
                    'change_pct': d.change_pct,
                    'net_buy': d.net_buy,
                    'reason': d.reason,
                    'dragon_score': d.dragon_score,
                    'next_1d_change': d.next_1d_change,
                    'next_5d_change': d.next_5d_change,
                    'data_quality': d.data_quality,
                } for d in dragons[:5]]
        except Exception as e:
            print(f"获取龙虎榜失败 {code}: {e}")

        # 计算综合基本面评分（带权重）
        result['fundamental_score'] = self._calculate_fundamental_score(result)

        # 计算数据完整性和质量
        result['data_completeness'], result['data_quality'] = self._calculate_data_metrics(result)

        return result

    def get_stock_fundamental_lite(self, code: str) -> Dict[str, Any]:
        """轻量级基本面数据获取 — 仅2次API调用（利润表+资产负债表）
        用于L1排除法和L2估值评分，避免完整获取的5+次API调用
        """
        result = {
            'code': code,
            'financial': None,
            'fundamental_score': 50,
            'data_completeness': 0,
            'data_quality': 0,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        financial_data = {
            'code': code,
            'name': '',
            'report_date': '',
            'industry': '',
            'field_quality': {},
        }

        # 1. 利润表（1次API调用）→ 获取营收、净利润、同比增长
        try:
            self._wait_for_rate_limit()
            profit_df = ak.stock_financial_report_sina(stock=code, symbol="利润表")
            if profit_df is not None and not profit_df.empty and len(profit_df) >= 1:
                latest = profit_df.iloc[0]
                report_date = str(latest.get('报告日', ''))
                financial_data['report_date'] = report_date

                revenue = DataCleaner.clean_numeric(latest.get('营业收入', 0))
                net_profit = DataCleaner.clean_numeric(latest.get('净利润', 0))
                financial_data['revenue'] = revenue / 1e8
                financial_data['net_profit'] = net_profit / 1e8

                # 同比增长
                yoy_match = self._find_yoy_match(profit_df, report_date)
                if yoy_match is not None:
                    prev_revenue = DataCleaner.clean_numeric(yoy_match.get('营业收入', 0))
                    prev_profit = DataCleaner.clean_numeric(yoy_match.get('净利润', 0))
                    if prev_revenue and prev_revenue != 0:
                        financial_data['revenue_yoy'] = (revenue - prev_revenue) / abs(prev_revenue) * 100
                    if prev_profit and prev_profit != 0:
                        financial_data['net_profit_yoy'] = (net_profit - prev_profit) / abs(prev_profit) * 100

                op_cost = DataCleaner.clean_numeric(latest.get('营业成本', 0))
                if revenue and revenue != 0 and op_cost:
                    financial_data['gross_margin'] = (revenue - op_cost) / revenue * 100
                if revenue and revenue != 0 and net_profit:
                    financial_data['net_margin'] = net_profit / revenue * 100

                # 从利润表获取OCF近似值（净利润近似经营现金流）
                financial_data['operating_cash_flow'] = net_profit / 1e8
        except Exception as e:
            print(f"轻量利润表获取失败 {code}: {e}")

        # 2. 资产负债表（1次API调用）→ ROE、商誉占比、权益
        try:
            self._wait_for_rate_limit()
            balance_df = ak.stock_financial_report_sina(stock=code, symbol="资产负债表")
            if balance_df is not None and not balance_df.empty:
                latest = balance_df.iloc[0]
                total_equity = self._get_equity_from_balance(latest)
                total_assets = DataCleaner.clean_numeric(latest.get('资产总计', 0))
                goodwill = DataCleaner.clean_numeric(latest.get('商誉', 0))
                total_liabilities = DataCleaner.clean_numeric(latest.get('负债合计', 0))

                financial_data['total_assets'] = total_assets / 1e8
                financial_data['total_liabilities'] = total_liabilities / 1e8

                if total_assets > 0 and total_liabilities > 0:
                    financial_data['asset_liability_ratio'] = total_liabilities / total_assets * 100

                # ROE
                net_profit = financial_data.get('net_profit', 0)
                if net_profit > 0 and total_equity > 0:
                    financial_data['roe'] = (net_profit * 1e8) / total_equity * 100

                # 商誉占净资产比
                if goodwill and total_equity and total_equity > 0:
                    financial_data['goodwill_ratio'] = goodwill / total_equity * 100

                # bvps存总权益（供估值计算用）
                if total_equity > 0:
                    financial_data['bvps'] = total_equity

                # 从OCF精确值（现金流量表不单独获取，用净利润近似）
                # 如果需要精确OCF，可在完整版获取
        except Exception as e:
            print(f"轻量资产负债表获取失败 {code}: {e}")

        # 清洗数据并计算评分
        financial_data = self._clean_financial_data(financial_data)

        # 计算评分
        financial_data['quality_score'] = self._calculate_quality_score_from_dict(financial_data)
        financial_data['growth_score'] = self._calculate_growth_score_from_dict(financial_data)
        financial_data['valuation_score'] = self._calculate_valuation_score_from_dict(financial_data)
        financial_data['data_quality'] = financial_data.get('data_quality', 1.0)

        result['financial'] = financial_data
        result['fundamental_score'] = self._calculate_fundamental_score(result)
        result['data_completeness'], result['data_quality'] = self._calculate_data_metrics(result)

        return result

    def _calculate_quality_score_from_dict(self, data: Dict) -> float:
        """从字典数据计算质量评分"""
        roe = data.get('roe', 0)
        net_margin = data.get('net_margin', 0)
        asset_liability_ratio = data.get('asset_liability_ratio', 0)
        score = 0
        if roe > 15: score += 40
        elif roe > 10: score += 30
        elif roe > 5: score += 20
        elif roe > 0: score += 10
        if net_margin > 30: score += 30
        elif net_margin > 15: score += 20
        elif net_margin > 5: score += 10
        if asset_liability_ratio < 30: score += 30
        elif asset_liability_ratio < 50: score += 20
        elif asset_liability_ratio < 70: score += 10
        return min(100, score)

    def _calculate_growth_score_from_dict(self, data: Dict) -> float:
        """从字典数据计算成长评分"""
        revenue_yoy = data.get('revenue_yoy', 0)
        net_profit_yoy = data.get('net_profit_yoy', 0)
        score = 0
        if revenue_yoy > 30: score += 50
        elif revenue_yoy > 15: score += 40
        elif revenue_yoy > 5: score += 30
        elif revenue_yoy > 0: score += 20
        elif revenue_yoy > -5: score += 10
        if net_profit_yoy > 30: score += 50
        elif net_profit_yoy > 15: score += 40
        elif net_profit_yoy > 5: score += 30
        elif net_profit_yoy > 0: score += 20
        elif net_profit_yoy > -5: score += 10
        return min(100, score / 2)

    def _calculate_valuation_score_from_dict(self, data: Dict) -> float:
        """从字典数据计算估值评分（简化版）"""
        # 在lite版本中，估值评分暂用默认值50
        # 精确估值需后续调用get_valuation_percentile
        return 40
        """计算综合基本面评分（考虑数据质量权重）"""
        score = 50

        # 动态权重（根据数据质量调整）
        weights = {'financial': 0.5, 'north_bound': 0.3, 'dragon_tiger': 0.2}

        # 获取各部分数据质量
        fin_quality = data.get('financial', {}).get('data_quality', 0) if data.get('financial') else 0
        north_quality = data.get('north_bound', {}).get('data_quality', 0) if data.get('north_bound') else 0

        # 根据数据质量调整权重
        if fin_quality < DataQuality.MEDIUM:
            weights['financial'] *= 0.7
            weights['north_bound'] *= 1.2

        if north_quality < DataQuality.MEDIUM:
            weights['north_bound'] *= 0.5
            weights['financial'] *= 1.1

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for k in weights:
                weights[k] /= total_weight

        # 财务评分
        if data.get('financial'):
            fin = data['financial']
            fin_score = fin.get('total_score', 50)
            quality_factor = fin.get('data_quality', 1.0)
            score += (fin_score - 50) * weights['financial'] * quality_factor

        # 北向资金评分
        if data.get('north_bound'):
            north = data['north_bound']
            north_score = north.get('flow_score', 50)
            quality_factor = north.get('data_quality', 1.0)
            is_stale = north.get('is_stale', False)
            if is_stale:
                quality_factor *= 0.5  # 过时数据降权
            score += (north_score - 50) * weights['north_bound'] * quality_factor

        # 龙虎榜评分
        if data.get('dragon_tiger'):
            dragons = data['dragon_tiger']
            if dragons:
                avg_dragon_score = sum(d['dragon_score'] for d in dragons) / len(dragons)
                avg_quality = sum(d.get('data_quality', 1.0) for d in dragons) / len(dragons)
                score += (avg_dragon_score - 50) * weights['dragon_tiger'] * avg_quality

        return max(0, min(100, score))

    def _calculate_fundamental_score(self, data: Dict) -> float:
        """计算综合基本面评分（考虑数据质量权重）"""
        score = 50

        weights = {'financial': 0.5, 'north_bound': 0.3, 'dragon_tiger': 0.2}

        fin_quality = data.get('financial', {}).get('data_quality', 0) if data.get('financial') else 0
        north_quality = data.get('north_bound', {}).get('data_quality', 0) if data.get('north_bound') else 0

        actual_weights = dict(weights)
        if not data.get('financial'):
            actual_weights['financial'] = 0
        if not data.get('north_bound'):
            actual_weights['north_bound'] = 0
        if not data.get('dragon_tiger'):
            actual_weights['dragon_tiger'] = 0

        total_w = sum(actual_weights.values())
        if total_w == 0:
            return 50

        for k in actual_weights:
            actual_weights[k] /= total_w

        if data.get('financial'):
            fin = data['financial']
            quality_s = fin.get('quality_score', 50)
            growth_s = fin.get('growth_score', 50)
            valuation_s = fin.get('valuation_score', 50)
            fin_score = quality_s * 0.4 + growth_s * 0.35 + valuation_s * 0.25
            score += (fin_score - 50) * actual_weights['financial'] * 2

        if data.get('north_bound'):
            nb = data['north_bound']
            flow_s = nb.get('flow_score', 50)
            score += (flow_s - 50) * actual_weights['north_bound']

        if data.get('dragon_tiger'):
            dragons = data['dragon_tiger']
            if dragons:
                avg_dragon = sum(d.get('dragon_score', 50) for d in dragons[:3]) / min(3, len(dragons))
                score += (avg_dragon - 50) * actual_weights['dragon_tiger']

        return max(0, min(100, score))

    def _calculate_data_metrics(self, data: Dict) -> Tuple[float, float]:
        """计算数据完整性和质量"""
        total_fields = 3
        filled_fields = 0
        quality_scores = []

        if data.get('financial'):
            filled_fields += 1
            quality_scores.append(data['financial'].get('data_quality', 1.0))

        if data.get('north_bound'):
            filled_fields += 1
            quality_scores.append(data['north_bound'].get('data_quality', 1.0))

        if data.get('dragon_tiger'):
            filled_fields += 1
            if data['dragon_tiger']:
                avg_quality = sum(d.get('data_quality', 1.0) for d in data['dragon_tiger']) / len(data['dragon_tiger'])
                quality_scores.append(avg_quality)

        completeness = filled_fields / total_fields
        quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return completeness, quality

    def batch_get_fundamental(self, codes: List[str]) -> Dict[str, Dict]:
        """批量获取股票基本面数据"""
        result = {}
        for code in codes:
            try:
                result[code] = self.get_stock_fundamental(code)
                time.sleep(0.5)
            except Exception as e:
                print(f"获取 {code} 基本面数据失败: {e}")
                result[code] = {'code': code, 'error': str(e)}
        return result

    # ==================== 港股基本面数据 ====================

    def get_hk_financial_report(self, code: str) -> Optional[FinancialReport]:
        """获取港股财务报告（基于akshare港股接口）

        Args:
            code: 5位港股代码，如 00700(腾讯), 09988(阿里)
        """
        cached = self._get_from_cache('hk_financial', code)
        if cached:
            return cached

        try:
            self._wait_for_rate_limit()

            # 获取港股个股信息
            stock_name = ''
            industry = ''
            try:
                df = ak.stock_hk_spot_em()
                row = df[df['代码'] == code]
                if not row.empty:
                    r = row.iloc[0]
                    stock_name = str(r.get('名称', code))
            except Exception as e:
                print(f"获取港股个股信息失败 {code}: {e}")
                stock_name = code

            # 获取港股估值指标（从实时行情中提取）
            financial_data = {
                'code': code,
                'name': stock_name,
                'report_date': '',
                'report_type': '港股季报',
                'industry': industry,
                'field_quality': {},
            }

            # 获取港股财务指标
            self._fetch_hk_financial_data(code, financial_data)

            # 清洗数据
            financial_data = self._clean_financial_data(financial_data, industry)

            report = FinancialReport(
                code=code,
                name=stock_name,
                report_date=financial_data.get('report_date', ''),
                report_type=financial_data.get('report_type', '港股季报'),
                revenue=financial_data.get('revenue', 0),
                revenue_yoy=financial_data.get('revenue_yoy', 0),
                net_profit=financial_data.get('net_profit', 0),
                net_profit_yoy=financial_data.get('net_profit_yoy', 0),
                gross_margin=financial_data.get('gross_margin', 0),
                net_margin=financial_data.get('net_margin', 0),
                roe=financial_data.get('roe', 0),
                roa=financial_data.get('roa', 0),
                eps=financial_data.get('eps', 0),
                bvps=financial_data.get('bvps', 0),
                cfps=financial_data.get('cfps', 0),
                total_assets=financial_data.get('total_assets', 0),
                total_liabilities=financial_data.get('total_liabilities', 0),
                asset_liability_ratio=financial_data.get('asset_liability_ratio', 0),
                current_ratio=financial_data.get('current_ratio', 0),
                quick_ratio=financial_data.get('quick_ratio', 0),
                operating_cash_flow=financial_data.get('operating_cash_flow', 0),
                free_cash_flow=financial_data.get('free_cash_flow', 0),
                pe_ratio=financial_data.get('pe_ratio', 0),
                pb_ratio=financial_data.get('pb_ratio', 0),
                ps_ratio=financial_data.get('ps_ratio', 0),
                total_market_cap=financial_data.get('total_market_cap', 0),
                revenue_cagr_3y=financial_data.get('revenue_cagr_3y', 0),
                profit_cagr_3y=financial_data.get('profit_cagr_3y', 0),
                revenue_qoq=financial_data.get('revenue_qoq', 0),
                profit_qoq=financial_data.get('profit_qoq', 0),
                industry=industry,
                field_quality=financial_data.get('field_quality', {}),
            )

            # 计算评分
            report.quality_score = self._calculate_quality_score(report)
            report.growth_score = self._calculate_growth_score(report)
            report.valuation_score = self._calculate_valuation_score(report)
            report.data_quality = financial_data.get('data_quality', 1.0)
            report.total_score = self._calculate_weighted_total_score(report)

            self._set_to_cache('hk_financial', code, report)
            return report

        except Exception as e:
            print(f"获取港股财报数据失败 {code}: {e}")
            return None

    def _fetch_hk_financial_data(self, code: str, financial_data: Dict) -> None:
        """获取港股财务数据"""
        # 从实时行情获取估值数据
        try:
            self._wait_for_rate_limit()
            df = ak.stock_hk_spot_em()
            row = df[df['代码'] == code]
            if not row.empty:
                r = row.iloc[0]
                financial_data['pe_ratio'] = DataCleaner.clean_numeric(r.get('市盈率-动态', 0))
                financial_data['pb_ratio'] = DataCleaner.clean_numeric(r.get('市净率', 0))
                financial_data['total_market_cap'] = DataCleaner.clean_numeric(r.get('总市值', 0)) / 1e8
                financial_data['name'] = str(r.get('名称', code))
        except Exception as e:
            print(f"获取港股估值数据失败 {code}: {e}")

        # 获取港股业绩报表
        try:
            self._wait_for_rate_limit()
            for date_str in ['20251231', '20250930', '20250630', '20250331', '20241231']:
                try:
                    df = ak.stock_hk_yjbb_em(date=date_str)
                    if df.empty:
                        continue
                    stock_data = df[df['代码'] == code]
                    if stock_data.empty:
                        continue
                    row = stock_data.iloc[0]
                    financial_data['report_date'] = date_str
                    financial_data['revenue'] = DataCleaner.clean_numeric(row.get('营业收入-营业收入', 0)) / 1e8
                    financial_data['revenue_yoy'] = DataCleaner.clean_numeric(row.get('营业收入-同比增长', 0))
                    financial_data['net_profit'] = DataCleaner.clean_numeric(row.get('净利润-净利润', 0)) / 1e8
                    financial_data['net_profit_yoy'] = DataCleaner.clean_numeric(row.get('净利润-同比增长', 0))
                    financial_data['eps'] = DataCleaner.clean_numeric(row.get('每股收益', 0))
                    financial_data['bvps'] = DataCleaner.clean_numeric(row.get('每股净资产', 0))
                    financial_data['roe'] = DataCleaner.clean_numeric(row.get('净资产收益率', 0))
                    financial_data['gross_margin'] = DataCleaner.clean_numeric(row.get('销售毛利率', 0))
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"获取港股业绩报表失败 {code}: {e}")

        # 计算净利率（如果有营收和净利润）
        revenue = financial_data.get('revenue', 0)
        net_profit = financial_data.get('net_profit', 0)
        if revenue > 0 and net_profit != 0:
            financial_data['net_margin'] = net_profit / revenue * 100

        # 获取港股财务报表（资产负债表等）
        try:
            self._wait_for_rate_limit()
            balance_df = ak.stock_hk_financial_report_sina(stock=code, symbol="资产负债表")
            if not balance_df.empty and len(balance_df) >= 1:
                latest = balance_df.iloc[0]
                financial_data['total_assets'] = DataCleaner.clean_numeric(latest.get('资产总计', 0)) / 1e8
                financial_data['total_liabilities'] = DataCleaner.clean_numeric(latest.get('负债合计', 0)) / 1e8
                if financial_data['total_assets'] > 0:
                    financial_data['asset_liability_ratio'] = (
                        financial_data['total_liabilities'] / financial_data['total_assets'] * 100
                    )
                current_assets = DataCleaner.clean_numeric(latest.get('流动资产', 0))
                current_liabilities = DataCleaner.clean_numeric(latest.get('流动负债', 0))
                if current_liabilities > 0:
                    financial_data['current_ratio'] = current_assets / current_liabilities
        except Exception as e:
            print(f"获取港股资产负债表失败 {code}: {e}")

        # 获取现金流量表
        try:
            self._wait_for_rate_limit()
            cashflow_df = ak.stock_hk_financial_report_sina(stock=code, symbol="现金流量表")
            if not cashflow_df.empty:
                latest = cashflow_df.iloc[0]
                financial_data['operating_cash_flow'] = DataCleaner.clean_numeric(
                    latest.get('经营活动产生的现金流量净额', 0)
                ) / 1e8
        except Exception as e:
            print(f"获取港股现金流表失败 {code}: {e}")

    # ==================== 美股基本面数据 ====================

    def get_us_financial_report(self, code: str) -> Optional[FinancialReport]:
        """获取美股财务报告（基于akshare美股接口）

        Args:
            code: 美股代码，如 AAPL, NVDA, TSLA
        """
        cached = self._get_from_cache('us_financial', code)
        if cached:
            return cached

        try:
            self._wait_for_rate_limit()

            # 获取美股个股信息
            stock_name = ''
            try:
                df = ak.stock_us_spot_em()
                row = df[df['代码'] == code]
                if not row.empty:
                    r = row.iloc[0]
                    stock_name = str(r.get('名称', code))
            except Exception as e:
                print(f"获取美股个股信息失败 {code}: {e}")
                stock_name = code

            financial_data = {
                'code': code,
                'name': stock_name,
                'report_date': '',
                'report_type': '美股季报',
                'industry': '',
                'field_quality': {},
            }

            # 获取美股估值指标
            self._fetch_us_financial_data(code, financial_data)

            # 清洗数据
            financial_data = self._clean_financial_data(financial_data, '')

            report = FinancialReport(
                code=code,
                name=stock_name,
                report_date=financial_data.get('report_date', ''),
                report_type=financial_data.get('report_type', '美股季报'),
                revenue=financial_data.get('revenue', 0),
                revenue_yoy=financial_data.get('revenue_yoy', 0),
                net_profit=financial_data.get('net_profit', 0),
                net_profit_yoy=financial_data.get('net_profit_yoy', 0),
                gross_margin=financial_data.get('gross_margin', 0),
                net_margin=financial_data.get('net_margin', 0),
                roe=financial_data.get('roe', 0),
                roa=financial_data.get('roa', 0),
                eps=financial_data.get('eps', 0),
                bvps=financial_data.get('bvps', 0),
                cfps=financial_data.get('cfps', 0),
                total_assets=financial_data.get('total_assets', 0),
                total_liabilities=financial_data.get('total_liabilities', 0),
                asset_liability_ratio=financial_data.get('asset_liability_ratio', 0),
                current_ratio=financial_data.get('current_ratio', 0),
                quick_ratio=financial_data.get('quick_ratio', 0),
                operating_cash_flow=financial_data.get('operating_cash_flow', 0),
                free_cash_flow=financial_data.get('free_cash_flow', 0),
                pe_ratio=financial_data.get('pe_ratio', 0),
                pb_ratio=financial_data.get('pb_ratio', 0),
                ps_ratio=financial_data.get('ps_ratio', 0),
                total_market_cap=financial_data.get('total_market_cap', 0),
                revenue_cagr_3y=financial_data.get('revenue_cagr_3y', 0),
                profit_cagr_3y=financial_data.get('profit_cagr_3y', 0),
                revenue_qoq=financial_data.get('revenue_qoq', 0),
                profit_qoq=financial_data.get('profit_qoq', 0),
                industry=financial_data.get('industry', ''),
                field_quality=financial_data.get('field_quality', {}),
            )

            # 计算评分
            report.quality_score = self._calculate_quality_score(report)
            report.growth_score = self._calculate_growth_score(report)
            report.valuation_score = self._calculate_valuation_score(report)
            report.data_quality = financial_data.get('data_quality', 1.0)
            report.total_score = self._calculate_weighted_total_score(report)

            self._set_to_cache('us_financial', code, report)
            return report

        except Exception as e:
            print(f"获取美股财报数据失败 {code}: {e}")
            return None

    def _fetch_us_financial_data(self, code: str, financial_data: Dict) -> None:
        """获取美股财务数据"""
        # 从实时行情获取估值指标
        try:
            self._wait_for_rate_limit()
            df = ak.stock_us_spot_em()
            row = df[df['代码'] == code]
            if not row.empty:
                r = row.iloc[0]
                financial_data['pe_ratio'] = DataCleaner.clean_numeric(r.get('市盈率-动态', 0))
                financial_data['pb_ratio'] = DataCleaner.clean_numeric(r.get('市净率', 0))
                financial_data['total_market_cap'] = DataCleaner.clean_numeric(r.get('总市值', 0)) / 1e8
                financial_data['name'] = str(r.get('名称', code))
        except Exception as e:
            print(f"获取美股估值数据失败 {code}: {e}")

        # 获取美股业绩报表
        try:
            self._wait_for_rate_limit()
            for date_str in ['20251231', '20250930', '20250630', '20250331', '20241231']:
                try:
                    df = ak.stock_us_yjbb_em(date=date_str)
                    if df.empty:
                        continue
                    stock_data = df[df['代码'] == code]
                    if stock_data.empty:
                        continue
                    row = stock_data.iloc[0]
                    financial_data['report_date'] = date_str
                    financial_data['revenue'] = DataCleaner.clean_numeric(row.get('营业收入-营业收入', 0)) / 1e8
                    financial_data['revenue_yoy'] = DataCleaner.clean_numeric(row.get('营业收入-同比增长', 0))
                    financial_data['net_profit'] = DataCleaner.clean_numeric(row.get('净利润-净利润', 0)) / 1e8
                    financial_data['net_profit_yoy'] = DataCleaner.clean_numeric(row.get('净利润-同比增长', 0))
                    financial_data['eps'] = DataCleaner.clean_numeric(row.get('每股收益', 0))
                    financial_data['bvps'] = DataCleaner.clean_numeric(row.get('每股净资产', 0))
                    financial_data['roe'] = DataCleaner.clean_numeric(row.get('净资产收益率', 0))
                    financial_data['gross_margin'] = DataCleaner.clean_numeric(row.get('销售毛利率', 0))
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"获取美股业绩报表失败 {code}: {e}")

        # 计算净利率
        revenue = financial_data.get('revenue', 0)
        net_profit = financial_data.get('net_profit', 0)
        if revenue > 0 and net_profit != 0:
            financial_data['net_margin'] = net_profit / revenue * 100

    def get_hk_stock_fundamental(self, code: str) -> Dict[str, Any]:
        """获取港股综合基本面数据"""
        result = {
            'code': code,
            'market': 'HK',
            'financial': None,
            'fundamental_score': 0,
            'data_completeness': 0,
            'data_quality': 0,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        try:
            report = self.get_hk_financial_report(code)
            if report:
                result['financial'] = {
                    'report_date': report.report_date,
                    'revenue': report.revenue,
                    'revenue_yoy': report.revenue_yoy,
                    'net_profit': report.net_profit,
                    'net_profit_yoy': report.net_profit_yoy,
                    'roe': report.roe,
                    'gross_margin': report.gross_margin,
                    'net_margin': report.net_margin,
                    'asset_liability_ratio': report.asset_liability_ratio,
                    'current_ratio': report.current_ratio,
                    'operating_cash_flow': report.operating_cash_flow,
                    'eps': report.eps,
                    'bvps': report.bvps,
                    'pe_ratio': report.pe_ratio,
                    'pb_ratio': report.pb_ratio,
                    'industry': report.industry,
                    'quality_score': report.quality_score,
                    'growth_score': report.growth_score,
                    'valuation_score': report.valuation_score,
                    'total_score': report.total_score,
                    'data_quality': report.data_quality,
                }
        except Exception as e:
            print(f"获取港股基本面数据失败 {code}: {e}")

        result['fundamental_score'] = result['financial'].get('total_score', 50) if result.get('financial') else 50
        result['data_completeness'] = 1.0 if result.get('financial') else 0.0
        result['data_quality'] = result['financial'].get('data_quality', 1.0) if result.get('financial') else 0.0

        return result

    def get_us_stock_fundamental(self, code: str) -> Dict[str, Any]:
        """获取美股综合基本面数据"""
        result = {
            'code': code,
            'market': 'US',
            'financial': None,
            'fundamental_score': 0,
            'data_completeness': 0,
            'data_quality': 0,
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        try:
            report = self.get_us_financial_report(code)
            if report:
                result['financial'] = {
                    'report_date': report.report_date,
                    'revenue': report.revenue,
                    'revenue_yoy': report.revenue_yoy,
                    'net_profit': report.net_profit,
                    'net_profit_yoy': report.net_profit_yoy,
                    'roe': report.roe,
                    'gross_margin': report.gross_margin,
                    'net_margin': report.net_margin,
                    'asset_liability_ratio': report.asset_liability_ratio,
                    'current_ratio': report.current_ratio,
                    'operating_cash_flow': report.operating_cash_flow,
                    'eps': report.eps,
                    'bvps': report.bvps,
                    'pe_ratio': report.pe_ratio,
                    'pb_ratio': report.pb_ratio,
                    'industry': report.industry,
                    'quality_score': report.quality_score,
                    'growth_score': report.growth_score,
                    'valuation_score': report.valuation_score,
                    'total_score': report.total_score,
                    'data_quality': report.data_quality,
                }
        except Exception as e:
            print(f"获取美股基本面数据失败 {code}: {e}")

        result['fundamental_score'] = result['financial'].get('total_score', 50) if result.get('financial') else 50
        result['data_completeness'] = 1.0 if result.get('financial') else 0.0
        result['data_quality'] = result['financial'].get('data_quality', 1.0) if result.get('financial') else 0.0

        return result


# ==================== 全局实例 ====================

_fundamental_client: Optional[FundamentalDataClient] = None


def get_fundamental_client(use_cache: bool = True) -> FundamentalDataClient:
    """获取基本面数据客户端实例"""
    global _fundamental_client
    if _fundamental_client is None:
        _fundamental_client = FundamentalDataClient(use_cache=use_cache)
    return _fundamental_client


def clear_fundamental_cache():
    """清除基本面数据缓存"""
    global _fundamental_client
    if _fundamental_client and _fundamental_client._cache:
        _fundamental_client._cache.clear()


# ==================== 因子标准化工具 ====================

def winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """去极值：将超出分位数的值截断到边界"""
    if series.empty or series.std() == 0:
        return series
    q_low = series.quantile(lower)
    q_high = series.quantile(upper)
    return series.clip(q_low, q_high)


def zscore_normalize(series: pd.Series) -> pd.Series:
    """Z-score标准化"""
    if series.empty:
        return series
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series([0.0] * len(series), index=series.index)
    result = (series - mean) / std
    result = result.replace([np.inf, -np.inf], 0).fillna(0)
    return result


def standardize_factors(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """对指定列做 winsorize + zscore 标准化"""
    for col in columns:
        if col in df.columns and df[col].notna().any():
            df[col] = winsorize(df[col].fillna(50))
            df[col] = zscore_normalize(df[col])
    return df


def get_valuation_percentile(stock_code: str, fd_client: FundamentalDataClient = None) -> Tuple[float, float]:
    """计算PE/PB估值百分位

    方案1: 从腾讯行情获取市值 + 新浪财报获取净利润/权益 → 计算PE/PB
    方案2: 默认50

    百分位越低 = 越被低估。
    Returns:
        Tuple[float, float]: (pe_percentile, pb_percentile)，范围 [0, 100]
    """
    pe_pct, pb_pct = 50.0, 50.0

    try:
        # 从腾讯行情获取总市值（亿元）
        market_cap = 0.0
        try:
            prefix = 'sh' if stock_code.startswith(('6', '9')) else 'sz'
            url = f"http://qt.gtimg.cn/q={prefix}{stock_code}"
            import requests
            r = requests.get(url, timeout=5)
            text = r.text.strip()
            if '~' in text:
                fields = text.split('~')
                if len(fields) > 45:
                    market_cap = float(fields[44])  # 总市值(亿元)
        except Exception:
            pass

        if market_cap <= 0:
            return pe_pct, pb_pct

        # 获取财报数据
        if fd_client is None:
            fd_client = get_fundamental_client()

        financial = fd_client.get_stock_fundamental(stock_code)
        if not financial or not financial.get('financial'):
            return pe_pct, pb_pct

        fin = financial['financial']
        net_profit = fin.get('net_profit', 0)  # 亿元
        total_equity_raw = fin.get('bvps', 0)  # 实际是归母权益总额(元)

        # 计算PE: 总市值 / 年化净利润
        report_date = fin.get('report_date', '')
        annualize_factor = 1
        if len(report_date) >= 8:
            period = report_date[4:]
            if period == '0331':
                annualize_factor = 4
            elif period == '0630':
                annualize_factor = 2
            elif period == '0930':
                annualize_factor = 4 / 3

        if net_profit > 0 and annualize_factor > 0:
            annualized_profit = net_profit * annualize_factor
            pe = market_cap / annualized_profit
            if pe < 10:
                pe_pct = 10.0
            elif pe < 20:
                pe_pct = 30.0
            elif pe < 40:
                pe_pct = 50.0
            elif pe < 80:
                pe_pct = 70.0
            else:
                pe_pct = 90.0

        # 计算PB: 总市值 / 归母权益
        if total_equity_raw > 0:
            market_cap_yuan = market_cap * 1e8  # 转回元
            pb = market_cap_yuan / total_equity_raw
            if pb < 1:
                pb_pct = 10.0
            elif pb < 2:
                pb_pct = 30.0
            elif pb < 4:
                pb_pct = 50.0
            elif pb < 8:
                pb_pct = 70.0
            else:
                pb_pct = 90.0

        return pe_pct, pb_pct
    except Exception:
        pass

    # 方案2: akshare stock_a_lg_indicator（可能已废弃）
    try:
        import akshare as ak
        if hasattr(ak, 'stock_a_lg_indicator'):
            df = ak.stock_a_lg_indicator(symbol=stock_code)
            if df is not None and not df.empty:
                # 取最近3年数据
                cutoff = datetime.now() - timedelta(days=365 * 3)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'] >= cutoff]

                if not df.empty and len(df) >= 30:
                    pe_col = None
                    pb_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'pe' in col_lower and pe_col is None:
                            pe_col = col
                        if 'pb' in col_lower and pb_col is None:
                            pb_col = col

                    latest = df.iloc[-1]
                    pe_pct = 50.0
                    pb_pct = 50.0

                    if pe_col and pe_col in df.columns:
                        current_pe = DataCleaner.clean_numeric(latest.get(pe_col, 0))
                        if current_pe > 0:
                            pe_series = df[pe_col].apply(DataCleaner.clean_numeric)
                            pe_series = pe_series[pe_series > 0]
                            if len(pe_series) >= 30:
                                pe_pct = float((pe_series <= current_pe).sum() / len(pe_series) * 100)

                    if pb_col and pb_col in df.columns:
                        current_pb = DataCleaner.clean_numeric(latest.get(pb_col, 0))
                        if current_pb > 0:
                            pb_series = df[pb_col].apply(DataCleaner.clean_numeric)
                            pb_series = pb_series[pb_series > 0]
                            if len(pb_series) >= 30:
                                pb_pct = float((pb_series <= current_pb).sum() / len(pb_series) * 100)

                    return round(pe_pct, 1), round(pb_pct, 1)
    except Exception:
        pass

    return 50.0, 50.0


# ==================== 测试入口 ====================

if __name__ == "__main__":
    client = get_fundamental_client()

    print("=== 测试财报数据（带数据清洗）===")
    report = client.get_financial_report('300308')
    if report:
        print(f"股票: {report.name}")
        print(f"行业: {report.industry}")
        print(f"营收: {report.revenue:.2f}亿, 同比: {report.revenue_yoy:.1f}%")
        print(f"净利润: {report.net_profit:.2f}亿, 同比: {report.net_profit_yoy:.1f}%")
        print(f"ROE: {report.roe:.2f}%")
        print(f"毛利率: {report.gross_margin:.2f}%")
        print(f"数据质量: {report.data_quality:.2f}")
        print(f"质量评分: {report.quality_score:.1f}")
        print(f"成长评分: {report.growth_score:.1f}")
        print(f"综合评分: {report.total_score:.1f}")

    print("\n=== 测试银行股（特殊行业处理）===")
    report = client.get_financial_report('000001')
    if report:
        print(f"股票: {report.name}")
        print(f"行业: {report.industry}")
        print(f"资产负债率: {report.asset_liability_ratio:.2f}%")
        print(f"毛利率: {report.gross_margin:.2f}%")
        print(f"质量评分: {report.quality_score:.1f}")

    print("\n=== 测试北向资金（带数据清洗）===")
    flow = client.get_north_bound_flow('600519')
    if flow:
        print(f"交易日期: {flow.trade_date}")
        print(f"持股比例: {flow.hold_ratio:.2f}%")
        print(f"5日变化: {flow.hold_change_5d:.2f}%")
        print(f"数据是否过时: {flow.is_stale}")
        print(f"数据质量: {flow.data_quality:.2f}")
        print(f"评分: {flow.flow_score:.1f}")
