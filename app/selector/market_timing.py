"""大盘择时模块
先定能不能做，再选做哪些股 — 择时优先于选股

基于沪深300指数判断大盘环境，分为五档：
- 全面做多：均线多头+放量，满仓操作
- 谨慎做多：趋势偏多但有疑虑，仓位受限
- 震荡：方向不明，减量操作
- 谨慎做空：趋势偏空，极低仓位
- 全面做空：均线空头+放量下跌，禁止建仓

择时结果直接影响：
1. 股票池数量（做空时减半）
2. 总仓位上限（做空时≤15%）
3. 选股标准（做空时只保留防御标的）
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import date, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_sync_db


# 择时等级
TIMING_FULL_LONG = 'full_long'        # 全面做多
TIMING_CAUTIOUS_LONG = 'cautious_long'  # 谨慎做多
TIMING_NEUTRAL = 'neutral'              # 震荡
TIMING_CAUTIOUS_SHORT = 'cautious_short'  # 谨慎做空
TIMING_FULL_SHORT = 'full_short'        # 全面做空

# 各等级对应的风控参数
TIMING_CONFIG = {
    TIMING_FULL_LONG: {
        'label': '全面做多',
        'pool_ratio': 1.0,           # 股票池不变
        'max_position': 0.85,        # 总仓位上限85%
        'single_position': 0.05,     # 单票5%
        'allow_hot_stocks': True,    # 允许纯热点题材
        'min_composite_score': 0.2,  # 最低综合分
    },
    TIMING_CAUTIOUS_LONG: {
        'label': '谨慎做多',
        'pool_ratio': 0.8,
        'max_position': 0.60,
        'single_position': 0.04,
        'allow_hot_stocks': True,
        'min_composite_score': 0.3,
    },
    TIMING_NEUTRAL: {
        'label': '震荡',
        'pool_ratio': 0.6,
        'max_position': 0.40,
        'single_position': 0.03,
        'allow_hot_stocks': True,
        'min_composite_score': 0.3,
    },
    TIMING_CAUTIOUS_SHORT: {
        'label': '谨慎做空',
        'pool_ratio': 0.4,
        'max_position': 0.15,
        'single_position': 0.02,
        'allow_hot_stocks': False,   # 禁止纯热点
        'min_composite_score': 0.5,
    },
    TIMING_FULL_SHORT: {
        'label': '全面做空',
        'pool_ratio': 0.2,
        'max_position': 0.10,
        'single_position': 0.02,
        'allow_hot_stocks': False,
        'min_composite_score': 0.8,  # 极高门槛
    },
}


class MarketTiming:
    """大盘择时模块"""

    # 沪深300指数代码
    CSI300_CODE = 'sh000300'

    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'data/weibo.db'

    def get_timing(self) -> Dict:
        """获取当前大盘择时判断

        Returns:
            Dict with timing level, config, and reasoning
        """
        # 获取沪深300数据
        index_data = self._get_index_data()

        if index_data is None or index_data.empty or len(index_data) < 20:
            # 数据不足，默认震荡
            return {
                'timing': TIMING_NEUTRAL,
                'config': TIMING_CONFIG[TIMING_NEUTRAL],
                'reason': '指数数据不足，默认震荡',
                'ma5': 0, 'ma20': 0, 'trend': 'unknown',
            }

        close = index_data['close']
        volume = index_data.get('volume', pd.Series([0] * len(close)))

        # 计算均线
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()

        current_close = close.iloc[-1]
        current_ma5 = ma5.iloc[-1]
        current_ma20 = ma20.iloc[-1]

        # 1. 趋势判断
        if current_ma5 > current_ma20 and current_close > current_ma5:
            trend = 'up'
        elif current_ma5 < current_ma20 and current_close < current_ma5:
            trend = 'down'
        else:
            trend = 'mixed'

        # 2. 均线斜率（趋势强度）
        if len(ma5) >= 5:
            ma5_slope = (ma5.iloc[-1] - ma5.iloc[-5]) / ma5.iloc[-5] * 100
        else:
            ma5_slope = 0

        # 3. 成交量变化
        if len(volume) >= 5 and volume.iloc[-5:].mean() > 0:
            vol_ratio = volume.iloc[-1] / volume.iloc[-5:].mean()
        else:
            vol_ratio = 1.0

        # 4. 近期涨跌幅
        if len(close) >= 5:
            recent_5d_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
        else:
            recent_5d_change = 0

        if len(close) >= 20:
            recent_20d_change = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
        else:
            recent_20d_change = 0

        # 5. 综合判定
        timing = self._judge_timing(
            trend, ma5_slope, vol_ratio,
            recent_5d_change, recent_20d_change
        )

        config = TIMING_CONFIG[timing]

        # 生成判断依据
        reasons = []
        if trend == 'up':
            reasons.append(f'MA5({current_ma5:.0f})>MA20({current_ma20:.0f})，多头排列')
        elif trend == 'down':
            reasons.append(f'MA5({current_ma5:.0f})<MA20({current_ma20:.0f})，空头排列')
        else:
            reasons.append(f'MA5({current_ma5:.0f})与MA20({current_ma20:.0f})交叉，方向不明')

        if abs(ma5_slope) > 1:
            reasons.append(f'5日斜率{ma5_slope:+.1f}%')

        if vol_ratio > 1.3:
            reasons.append(f'放量({vol_ratio:.1f}倍)')
        elif vol_ratio < 0.7:
            reasons.append(f'缩量({vol_ratio:.1f}倍)')

        reasons.append(f'近5日{recent_5d_change:+.1f}%/近20日{recent_20d_change:+.1f}%')

        return {
            'timing': timing,
            'config': config,
            'reason': '；'.join(reasons),
            'ma5': round(current_ma5, 2),
            'ma20': round(current_ma20, 2),
            'trend': trend,
            'ma5_slope': round(ma5_slope, 2),
            'vol_ratio': round(vol_ratio, 2),
            'recent_5d_change': round(recent_5d_change, 2),
            'recent_20d_change': round(recent_20d_change, 2),
        }

    def _judge_timing(self, trend: str, ma5_slope: float,
                      vol_ratio: float, change_5d: float,
                      change_20d: float) -> str:
        """综合判定择时等级"""
        # 评分制：正分偏多，负分偏空
        score = 0

        # 趋势
        if trend == 'up':
            score += 2
        elif trend == 'down':
            score -= 2

        # 均线斜率
        if ma5_slope > 2:
            score += 1
        elif ma5_slope > 0:
            score += 0.5
        elif ma5_slope < -2:
            score -= 1
        elif ma5_slope < 0:
            score -= 0.5

        # 量能配合
        if trend == 'up' and vol_ratio > 1.2:
            score += 1  # 上涨放量=确认
        elif trend == 'up' and vol_ratio < 0.8:
            score -= 0.5  # 上涨缩量=虚涨
        elif trend == 'down' and vol_ratio > 1.3:
            score -= 1  # 下跌放量=恐慌
        elif trend == 'down' and vol_ratio < 0.7:
            score += 0.5  # 下跌缩量=卖盘枯竭

        # 近期涨跌幅
        if change_5d > 3:
            score += 1
        elif change_5d > 0:
            score += 0.3
        elif change_5d < -3:
            score -= 1
        elif change_5d < 0:
            score -= 0.3

        if change_20d > 8:
            score += 1
        elif change_20d < -8:
            score -= 1

        # 评分→等级映射
        if score >= 4:
            return TIMING_FULL_LONG
        elif score >= 2:
            return TIMING_CAUTIOUS_LONG
        elif score >= -1:
            return TIMING_NEUTRAL
        elif score >= -3:
            return TIMING_CAUTIOUS_SHORT
        else:
            return TIMING_FULL_SHORT

    def _get_index_data(self) -> Optional[pd.DataFrame]:
        """获取沪深300指数数据"""
        # 1. 从市场客户端获取
        try:
            from app.market import get_market_client
            client = get_market_client()
            df = client.get_stock_history('000300', days=30)
            if df is not None and not df.empty and len(df) >= 20:
                return df
        except Exception:
            pass

        # 2. 从AkShare获取
        try:
            import akshare as ak
            df = ak.stock_zh_index_daily(symbol='sh000300')
            if df is not None and not df.empty and len(df) >= 20:
                return df.tail(30)
        except Exception:
            pass

        return None

    def should_trade(self) -> bool:
        """是否应该交易（谨慎做空及以上允许有限交易）"""
        timing = self.get_timing()
        return timing['timing'] in (
            TIMING_FULL_LONG, TIMING_CAUTIOUS_LONG, TIMING_NEUTRAL
        )

    def get_max_position(self) -> float:
        """获取当前最大仓位"""
        timing = self.get_timing()
        return timing['config']['max_position']

    def apply_timing_to_pool(self, candidates: list, max_stocks: int) -> list:
        """根据择时结果过滤股票池

        Args:
            candidates: 候选股票列表
            max_stocks: 原始最大股票数

        Returns:
            过滤后的股票列表
        """
        timing = self.get_timing()
        config = timing['config']

        # 调整股票池数量
        adjusted_max = max(3, int(max_stocks * config['pool_ratio']))

        # 谨慎做空/全面做空时，禁止纯热点题材
        if not config['allow_hot_stocks']:
            filtered = [s for s in candidates
                       if s.get('selection_source') != '市场热点'
                       or s.get('hot_freshness', 0) >= 80]  # 只保留极新鲜热点
            if len(filtered) > 0:
                candidates = filtered

        # 提高准入门槛
        min_score = config['min_composite_score']
        if min_score > 0.3:
            scored = [s for s in candidates if s.get('composite_score', 0) >= min_score]
            if len(scored) > 0:
                candidates = scored

        return candidates[:adjusted_max]
