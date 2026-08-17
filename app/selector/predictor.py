"""
股票预测器
基于技术指标和市场数据，预测次日走势、涨幅、持股周期
改进版本：增加市场情绪因子、板块热度、波动性指标
预测次日走势，次日复盘验证
"""
from typing import Dict, Optional, Any, List
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
import json
import sqlite3
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_sync_db

import pandas as pd
import numpy as np

from app.market import get_market_client


# 市场情绪阈值配置
MARKET_SENTIMENT_THRESHOLDS = {
    'strong_bull': 2.0,    # 强势上涨阈值
    'bull': 1.0,           # 上涨阈值
    'bear': -1.0,          # 下跌阈值
    'strong_bear': -2.0,   # 强势下跌阈值
}

# 板块热度权重
SECTOR_HEAT_WEIGHT = 0.3

# 波动性指标权重
VOLATILITY_WEIGHT = 0.4


@dataclass
class DailyPrediction:
    """每日预测 - 预测次日走势"""
    code: str
    name: str
    prediction_date: date  # 预测目标日期（次日）
    prediction_time: datetime  # 预测生成时间

    # 当前状态
    current_price: float  # 当前股票价格

    # 当日预测
    expected_open_change: float  # 预期开盘涨跌（相对于昨收）
    expected_amplitude: float  # 预期振幅
    expected_high: float  # 预期最高价
    expected_low: float  # 预期最低价
    expected_close_change: float  # 预期收盘涨跌
    expected_trend: str  # 预期走势描述
    expected_strength: str  # 预期强度

    # 持股周期预测
    expected_hold_days: int  # 建议持有天数
    expected_hold_return: float  # 预期持股收益
    expected_buy_point: str  # 建议买入点位
    expected_sell_point: str  # 建议卖出点位
    expected_sell_price: float = 0.0  # 卖点价格

    # 分析依据
    technical_basis: Dict[str, Any] = field(default_factory=dict)
    market_basis: Dict[str, Any] = field(default_factory=dict)

    # 选股原因
    selection_reason: str = ""

    # 风险提示
    risk_warning: str = ""

    # 实际结果（复盘时填充）
    actual_change: float = 0.0  # 当天实际涨幅

    # ML模型相关字段
    prediction_source: str = "rule_based"  # ml_model / rule_based
    confidence: float = 0.0  # ML置信度 0-1
    confidence_interval: str = ""  # 如 "[+0.5%, +3.2%]"


class Predictor:
    """股票预测器 - 改进版（集成ML模型 + 复盘反馈闭环）"""

    def __init__(self):
        self.market_client = get_market_client()
        self._market_sentiment = None
        self._hot_sectors = None
        self._cache_time = None
        self._ml_model = None  # ML模型实例（延迟加载）
        self._feedback_adj = None  # 复盘反馈调整结果（延迟加载）

    def _get_ml_model(self):
        """延迟加载ML模型"""
        if self._ml_model is None:
            from .ml_predictor import get_ml_model
            self._ml_model = get_ml_model()
        return self._ml_model

    def _get_feedback_adjustment(self) -> float:
        """获取复盘反馈的sentiment_factor调整系数

        Returns:
            float: sentiment_factor调整系数（1.0=不变，<1=降低，>1=提高）
        """
        if self._feedback_adj is not None:
            return self._feedback_adj

        try:
            from app.reviewer.feedback_adjuster import FeedbackAdjuster
            adjuster = FeedbackAdjuster()
            result = adjuster.analyze_feedback(days=5)
            if result.total_samples > 0:
                self._feedback_adj = result.sentiment_factor_adj
            else:
                self._feedback_adj = 1.0
        except Exception as e:
            print(f"获取复盘反馈调整失败: {e}")
            self._feedback_adj = 1.0

        return self._feedback_adj

    def _get_timing_offset(self) -> float:
        """获取大盘择时偏移量

        根据择时等级返回预测修正值，解决系统性偏多+0.58%的问题。
        熊市时自动下调预测值，避免在下跌市中持续亏损。

        Returns:
            float: 偏移百分比（负值=下调预测，正值=上调）
        """
        try:
            from app.selector.market_timing import MarketTiming
            mt = MarketTiming()
            result = mt.get_timing()
            timing = result.get('timing', 'neutral')

            offsets = {
                'full_long': 0.2,
                'cautious_long': 0.1,
                'neutral': 0.0,
                'cautious_short': -0.5,
                'full_short': -1.0,
            }
            offset = offsets.get(timing, 0.0)
            if offset != 0.0:
                print(f"  择时偏移: {timing} → 预测修正{offset:+.1f}%")
            return offset
        except Exception as e:
            print(f"获取择时偏移失败: {e}")
            return 0.0

    def _get_market_sentiment(self) -> Dict[str, Any]:
        """获取市场整体情绪（大盘影响）"""
        # 缓存5分钟
        if self._market_sentiment and self._cache_time:
            if (datetime.now() - self._cache_time).total_seconds() < 300:
                return self._market_sentiment

        try:
            # 获取上证指数
            sh_index = self.market_client.get_stock_price('000001')
            # 获取深证成指
            sz_index = self.market_client.get_stock_price('399001')
            # 获取创业板指
            cyb_index = self.market_client.get_stock_price('399006')

            sh_change = sh_index.get('change_pct', 0) if sh_index else 0
            sz_change = sz_index.get('change_pct', 0) if sz_index else 0
            cyb_change = cyb_index.get('change_pct', 0) if cyb_index else 0

            # 计算综合市场情绪（加权平均）
            avg_change = (sh_change * 0.4 + sz_change * 0.3 + cyb_change * 0.3)

            # 判断市场情绪
            if avg_change > MARKET_SENTIMENT_THRESHOLDS['strong_bull']:
                sentiment = 'strong_bull'
                sentiment_factor = 1.3  # 强势上涨，个股预期上调
            elif avg_change > MARKET_SENTIMENT_THRESHOLDS['bull']:
                sentiment = 'bull'
                sentiment_factor = 1.15
            elif avg_change < MARKET_SENTIMENT_THRESHOLDS['strong_bear']:
                sentiment = 'strong_bear'
                sentiment_factor = 0.7  # 强势下跌，个股预期下调
            elif avg_change < MARKET_SENTIMENT_THRESHOLDS['bear']:
                sentiment = 'bear'
                sentiment_factor = 0.85
            else:
                sentiment = 'neutral'
                sentiment_factor = 1.0

            self._market_sentiment = {
                'sentiment': sentiment,
                'factor': sentiment_factor,
                'sh_change': sh_change,
                'sz_change': sz_change,
                'cyb_change': cyb_change,
                'avg_change': avg_change
            }
            self._cache_time = datetime.now()
            return self._market_sentiment

        except Exception as e:
            print(f"获取市场情绪失败: {e}")
            return {'sentiment': 'neutral', 'factor': 1.0, 'avg_change': 0}

    def _get_hot_sectors(self) -> List[str]:
        """获取热门板块列表"""
        if self._hot_sectors and self._cache_time:
            if (datetime.now() - self._cache_time).total_seconds() < 300:
                return self._hot_sectors

        try:
            hot_concepts = self.market_client.get_hot_concepts(limit=10)
            self._hot_sectors = [c['name'] for c in hot_concepts if c.get('change_pct', 0) > 0]
            return self._hot_sectors
        except Exception as e:
            print(f"获取热门板块失败: {e}")
            return []

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR（平均真实波幅）"""
        if len(df) < period:
            return 0

        high = df['high'].tail(period).values
        low = df['low'].tail(period).values
        close = df['close'].tail(period + 1).values[:-1]  # 前一日收盘

        tr_list = []
        for i in range(period):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i]),
                abs(low[i] - close[i])
            )
            tr_list.append(tr)

        return sum(tr_list) / period if tr_list else 0

    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算波动性指标"""
        if len(df) < 20:
            return {'volatility': 0, 'atr': 0, 'atr_pct': 0}

        # 计算收益率标准差（年化波动率）
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0

        # 计算ATR
        atr = self._calculate_atr(df)
        current_price = df['close'].iloc[-1]
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        # 计算历史振幅统计
        if 'amplitude' in df.columns:
            avg_amplitude = df['amplitude'].tail(20).mean()
            max_amplitude = df['amplitude'].tail(20).max()
            min_amplitude = df['amplitude'].tail(20).min()
        else:
            # 手动计算振幅
            amplitudes = (df['high'] - df['low']) / df['close'].shift(1) * 100
            amplitudes = amplitudes.dropna()
            avg_amplitude = amplitudes.tail(20).mean() if len(amplitudes) > 0 else 3.0
            max_amplitude = amplitudes.tail(20).max() if len(amplitudes) > 0 else 5.0
            min_amplitude = amplitudes.tail(20).min() if len(amplitudes) > 0 else 2.0

        return {
            'volatility': volatility,
            'atr': atr,
            'atr_pct': atr_pct,
            'avg_amplitude': avg_amplitude,
            'max_amplitude': max_amplitude,
            'min_amplitude': min_amplitude
        }

    def predict(self, code: str, name: str = None,
                blogger_cycle: str = "波段（1-4周）") -> Optional[DailyPrediction]:
        """
        对股票进行预测（优先使用ML模型，降级到规则引擎）

        Args:
            code: 股票代码
            name: 股票名称（可选）
            blogger_cycle: 博主交易周期

        Returns:
            DailyPrediction: 预测结果
        """
        try:
            # 先获取实时数据
            sina_data = self.market_client.get_stock_price(code)
            if not sina_data or sina_data.get('price', 0) <= 0:
                print(f"股票 {code} 无法获取实时数据")
                return None

            if not name:
                name = sina_data.get('name', code)

            price = sina_data.get('price', 0)

            # === 尝试使用ML模型预测 ===
            ml_result = self._predict_with_ml(code, name, price, blogger_cycle)
            if ml_result is not None:
                return ml_result

            # === 降级：使用规则引擎 ===
            df = self.market_client.get_stock_history(code, days=60)

            if not df.empty and len(df) >= 20:
                tech = self.market_client.get_stock_technical_indicators(code)
                try:
                    flow = self.market_client.get_money_flow(code)
                except:
                    flow = {}
                return self._analyze_and_predict(code, name, df, tech, flow, blogger_cycle)
            else:
                print(f"股票 {code} 历史数据不足({len(df)}条)，使用实时数据预测")
                return self._simple_predict(code, name, sina_data, blogger_cycle)

        except Exception as e:
            print(f"预测股票 {code} 失败: {e}")
            return None

    def _predict_with_ml(self, code: str, name: str, current_price: float,
                         blogger_cycle: str) -> Optional[DailyPrediction]:
        """使用ML模型进行预测，返回None则降级到规则引擎"""
        try:
            model = self._get_ml_model()
            if not model.is_trained:
                return None

            from .data_collector import fetch_kline_data
            klines = fetch_kline_data(code, days=60)
            if len(klines) < 25:
                return None

            ml_result = model.predict(klines, code, name or code)
            if ml_result is None:
                return None

            now = datetime.now()
            target_date = date.today() + timedelta(days=1)

            # ML涨跌幅 → 中文趋势描述
            change = ml_result.predicted_change_pct
            if ml_result.predicted_trend == 'up':
                trend_desc = "震荡上涨" if change < 2 else "强势上涨"
                strength = "中强" if change < 2 else "强"
            elif ml_result.predicted_trend == 'down':
                trend_desc = "震荡下跌" if change > -2 else "弱势下跌"
                strength = "中弱" if change > -2 else "弱"
            else:
                trend_desc = "横盘震荡"
                strength = "中"

            # 持股周期
            if '超短线' in blogger_cycle:
                hold_days = 2
            elif '短线' in blogger_cycle:
                hold_days = 5
            elif '波段' in blogger_cycle:
                hold_days = 15
            elif '中长线' in blogger_cycle:
                hold_days = 60
            else:
                hold_days = 10

            ci = ml_result.confidence_interval
            interval_str = f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]"

            return DailyPrediction(
                code=code,
                name=name or code,
                prediction_date=target_date,
                prediction_time=now,
                current_price=current_price,
                expected_open_change=round(change * 0.3, 2),
                expected_amplitude=round(abs(change) * 1.5, 2) if change else 3.0,
                expected_high=round(current_price * (1 + abs(change) * 0.01), 2),
                expected_low=round(current_price * (1 - abs(change) * 0.01), 2),
                expected_close_change=round(change, 2),
                expected_trend=trend_desc,
                expected_strength=strength,
                expected_hold_days=hold_days,
                expected_hold_return=round(change * hold_days / 5, 2),
                expected_buy_point=f"参考现价 {current_price:.2f} 元附近",
                expected_sell_point=f"目标涨幅 {abs(change) * 2:.1f}%",
                expected_sell_price=round(current_price * (1 + abs(change) * 2 / 100), 2),
                technical_basis={'source': 'ml_model', 'predicted_change': change,
                                 'direction_prob': ml_result.predicted_direction_prob,
                                 'feature_importance': ml_result.feature_importance},
                market_basis={},
                selection_reason=self._build_ml_reason(ml_result),
                risk_warning=f"ML模型置信度{ml_result.confidence:.0%}，置信区间{interval_str}",
                prediction_source="ml_model",
                confidence=ml_result.confidence,
                confidence_interval=interval_str,
            )
        except Exception as e:
            print(f"ML预测失败({code}): {e}")
            return None

    def _build_ml_reason(self, ml_result) -> str:
        """生成ML预测原因说明"""
        parts = [f"ML模型预测{ml_result.predicted_trend}"]
        parts.append(f"预测涨跌幅{ml_result.predicted_change_pct:+.2f}%")
        parts.append(f"方向概率{ml_result.predicted_direction_prob:.0%}")

        top_features = sorted(
            ml_result.feature_importance.items(),
            key=lambda x: -x[1]
        )[:3] if ml_result.feature_importance else []
        if top_features:
            names = ', '.join(f"{n}({v:.3f})" for n, v in top_features)
            parts.append(f"关键因子: {names}")

        return "；".join(parts)

    def _simple_predict(self, code: str, name: str,
                        sina_data: Dict, blogger_cycle: str) -> DailyPrediction:
        """简化预测（当历史数据不可用时）- 预测次日走势"""
        now = datetime.now()
        # 预测目标日期为次日
        target_date = date.today() + timedelta(days=1)

        price = sina_data.get('price', 0)
        change_pct = sina_data.get('change_pct', 0)

        # 根据当日涨跌判断趋势
        if change_pct > 3:
            trend = "强势上涨"
            strength = "强"
            expected_change = change_pct * 0.5  # 预期继续上涨
        elif change_pct > 1:
            trend = "震荡上涨"
            strength = "中强"
            expected_change = change_pct * 0.3
        elif change_pct < -3:
            trend = "弱势下跌"
            strength = "弱"
            expected_change = change_pct * 0.5
        elif change_pct < -1:
            trend = "震荡下跌"
            strength = "中弱"
            expected_change = change_pct * 0.3
        else:
            trend = "横盘震荡"
            strength = "中"
            expected_change = change_pct * 0.2

        # 根据博主周期调整持股天数
        if '超短线' in blogger_cycle:
            hold_days = 2
        elif '短线' in blogger_cycle:
            hold_days = 5
        elif '波段' in blogger_cycle:
            hold_days = 15
        elif '中长线' in blogger_cycle:
            hold_days = 60
        else:
            hold_days = 10

        return DailyPrediction(
            code=code,
            name=name,
            prediction_date=target_date,  # 预测次日
            prediction_time=now,
            current_price=price,
            expected_open_change=change_pct * 0.3,
            expected_amplitude=abs(change_pct) * 1.5 if change_pct else 3.0,
            expected_high=price * (1 + abs(change_pct) * 0.01),
            expected_low=price * (1 - abs(change_pct) * 0.01),
            expected_close_change=round(expected_change, 2),
            expected_trend=trend,
            expected_strength=strength,
            expected_hold_days=hold_days,
            expected_hold_return=round(expected_change * hold_days / 5, 2),
            expected_buy_point=f"参考现价 {price:.2f} 元附近",
            expected_sell_point=f"目标涨幅 {abs(expected_change) * 2:.1f}%",
            expected_sell_price=round(price * (1 + abs(expected_change) * 2 / 100), 2),
            technical_basis={'source': 'sina', 'price': price, 'change_pct': change_pct},
            market_basis={},
            selection_reason="基于实时行情数据预测",
            risk_warning="历史数据不足，预测仅供参考。请关注系统实盘胜率，单笔仓位≤5%，严格止损"
        )

    def _analyze_and_predict(self, code: str, name: str,
                             df: pd.DataFrame, tech: Dict,
                             flow: Dict, blogger_cycle: str) -> DailyPrediction:
        """分析并生成预测 - 预测次日走势"""

        now = datetime.now()
        # 预测目标日期为次日
        target_date = date.today() + timedelta(days=1)

        # 获取最近数据
        latest = df.iloc[-1]
        prev_close = latest['close']
        current_price = tech.get('current_price', prev_close)

        # 获取市场情绪
        market_sentiment = self._get_market_sentiment()
        sentiment_factor = market_sentiment.get('factor', 1.0)

        # 复盘反馈闭环：调整sentiment_factor
        feedback_adj = self._get_feedback_adjustment()
        sentiment_factor *= feedback_adj

        # 大盘择时偏移：根据择时等级修正预测，避免系统性偏多
        timing_offset = self._get_timing_offset()

        # 获取热门板块
        hot_sectors = self._get_hot_sectors()

        # 计算波动性指标
        volatility_data = self._calculate_volatility(df)

        # 分析趋势（加入市场情绪因子+择时偏移）
        trend_analysis = self._analyze_trend(df, tech, sentiment_factor, timing_offset)

        # 分析振幅（使用波动性指标）
        amplitude_analysis = self._analyze_amplitude(df, tech, volatility_data)

        # 分析买卖点（加入资金流向）
        buy_sell_analysis = self._analyze_buy_sell_points(df, tech, flow)

        # 分析持股周期
        hold_analysis = self._analyze_hold_period(df, tech, blogger_cycle, timing_offset)

        # 计算卖点价格（考虑波动性）
        sell_price = round(current_price * (1 + hold_analysis['return'] / 100), 2)

        # 生成预测
        prediction = DailyPrediction(
            code=code,
            name=name,
            prediction_date=target_date,  # 预测次日
            prediction_time=now,
            current_price=current_price,
            expected_open_change=trend_analysis['open_change'],
            expected_amplitude=amplitude_analysis['amplitude'],
            expected_high=prev_close * (1 + amplitude_analysis['high_pct'] / 100),
            expected_low=prev_close * (1 + amplitude_analysis['low_pct'] / 100),
            expected_close_change=trend_analysis['close_change'],
            expected_trend=trend_analysis['trend_desc'],
            expected_strength=trend_analysis['strength'],
            expected_hold_days=hold_analysis['days'],
            expected_hold_return=hold_analysis['return'],
            expected_buy_point=buy_sell_analysis['buy_point'],
            expected_sell_point=buy_sell_analysis['sell_point'],
            expected_sell_price=sell_price,
            technical_basis={**tech, **volatility_data},
            market_basis={'flow': flow, 'sentiment': market_sentiment, 'hot_sectors': hot_sectors},
            selection_reason=self._generate_reason(tech, trend_analysis),
            risk_warning=self._generate_risk(tech, amplitude_analysis, volatility_data)
        )

        return prediction

    def _analyze_trend(self, df: pd.DataFrame, tech: Dict,
                        sentiment_factor: float = 1.0, timing_offset: float = 0.0) -> Dict:
        """分析趋势 - 改进版，加入市场情绪因子+近期动量+择时偏移"""
        close = df['close'].iloc[-1]
        ma5 = tech.get('ma5', close)
        ma10 = tech.get('ma10', close)
        ma20 = tech.get('ma20', close)

        # 近3日动量：用实际涨跌幅修正纯均线判断
        recent_change = 0
        if len(df) >= 4:
            recent_change = (df['close'].iloc[-1] / df['close'].iloc[-4] - 1) * 100

        # 判断趋势方向 — 修正正负不对称：看多+2.0/+1.0 vs 看空-1.5/-0.8
        # 改为对称：强趋势±2.0，中趋势±1.0，弱趋势±0.5
        if close > ma5 > ma10 > ma20:
            trend_desc = "强势上涨"
            strength = "强"
            close_change = 2.0
            open_change = 0.8
        elif close > ma5 and close > ma20:
            trend_desc = "震荡上涨"
            strength = "中强"
            close_change = 1.0
            open_change = 0.3
        elif close < ma5 < ma10 < ma20:
            trend_desc = "弱势下跌"
            strength = "弱"
            close_change = -2.0
            open_change = -0.8
        elif close < ma5:
            trend_desc = "震荡下跌"
            strength = "中弱"
            close_change = -1.0
            open_change = -0.3
        else:
            trend_desc = "横盘震荡"
            strength = "中"
            close_change = 0.0
            open_change = 0.0

        # 根据成交量调整
        volume_change = tech.get('volume_change_pct', 0)
        if volume_change > 50:  # 放量
            if close_change > 0:
                close_change *= 1.3
                trend_desc = "放量" + trend_desc
            else:
                close_change *= 0.8

        # 【改进】应用市场情绪因子
        close_change = close_change * sentiment_factor
        open_change = open_change * sentiment_factor

        # 【改进】应用择时偏移：熊市下调，牛市上调
        close_change += timing_offset
        open_change += timing_offset * 0.5

        # 【改进】根据资金流向调整
        main_inflow = tech.get('main_net_inflow_pct', 0)
        if main_inflow > 3:
            close_change *= 1.1
        elif main_inflow < -3:
            close_change *= 0.9

        return {
            'trend_desc': trend_desc,
            'strength': strength,
            'close_change': round(close_change, 2),
            'open_change': round(open_change, 2)
        }

    def _analyze_amplitude(self, df: pd.DataFrame, tech: Dict,
                            volatility_data: Dict = None) -> Dict:
        """分析振幅 - 改进版，使用波动性指标"""
        # 【改进】优先使用波动性指标
        if volatility_data:
            atr_pct = volatility_data.get('atr_pct', 0)
            avg_amplitude = volatility_data.get('avg_amplitude', 3.0)
            max_amplitude = volatility_data.get('max_amplitude', 5.0)
            min_amplitude = volatility_data.get('min_amplitude', 2.0)
        else:
            # 计算历史振幅
            if 'amplitude' in df.columns:
                avg_amplitude = df['amplitude'].tail(10).mean()
                max_amplitude = df['amplitude'].tail(20).max()
                min_amplitude = df['amplitude'].tail(20).min()
            else:
                amplitudes = (df['high'] - df['low']) / df['close'].shift(1) * 100
                amplitudes = amplitudes.dropna()
                avg_amplitude = amplitudes.tail(10).mean() if len(amplitudes) > 0 else 3.0
                max_amplitude = amplitudes.tail(20).max() if len(amplitudes) > 0 else 5.0
                min_amplitude = amplitudes.tail(20).min() if len(amplitudes) > 0 else 2.0
            atr_pct = avg_amplitude

        # 【改进】综合计算预期振幅
        # 使用ATR和历史振幅的加权平均
        if atr_pct > 0:
            base_amplitude = (atr_pct * VOLATILITY_WEIGHT + avg_amplitude * (1 - VOLATILITY_WEIGHT))
        else:
            base_amplitude = avg_amplitude

        # 根据趋势调整
        trend = tech.get('trend', '震荡')
        if trend == '上涨':
            amplitude = base_amplitude * 1.15  # 上涨时振幅略大
        elif trend == '下跌':
            amplitude = base_amplitude * 1.1   # 下跌时振幅也略大
        else:
            amplitude = base_amplitude

        # 【改进】根据成交量变化调整
        volume_change = tech.get('volume_change_pct', 0)
        if volume_change > 50:  # 放量时振幅增大
            amplitude *= 1.2
        elif volume_change < -30:  # 缩量时振幅减小
            amplitude *= 0.9

        # 限制振幅范围（避免极端预测，但给足缓冲）
        amplitude = max(min_amplitude * 0.9, min(amplitude, max_amplitude * 1.5))

        # 计算高低点
        high_pct = amplitude / 2
        low_pct = -amplitude / 2

        return {
            'amplitude': round(amplitude, 2),
            'high_pct': round(high_pct, 2),
            'low_pct': round(low_pct, 2)
        }

    def _analyze_buy_sell_points(self, df: pd.DataFrame,
                                  tech: Dict, flow: Dict) -> Dict:
        """分析买卖点"""
        close = df['close'].iloc[-1]
        ma5 = tech.get('ma5', close)
        ma20 = tech.get('ma20', close)
        low_20 = tech.get('low_20d', close)

        # 买入点建议
        if close > ma5:
            buy_point = f"回调至MA5附近({ma5:.2f})可考虑低吸"
        elif close < ma20:
            buy_point = f"企稳后介入，关注{low_20:.2f}支撑"
        else:
            buy_point = f"突破MA5({ma5:.2f})确认后买入"

        # 资金流向影响
        main_inflow = flow.get('main_net_inflow_pct', 0)
        if main_inflow > 5:
            buy_point += "，主力资金流入积极"
        elif main_inflow < -5:
            buy_point += "，注意主力资金流出"

        # 卖出点建议
        high_20 = tech.get('high_20d', close)
        if close > high_20 * 0.95:
            sell_point = f"冲高至{high_20:.2f}附近可减仓"
        else:
            sell_point = f"涨幅达5-10%可分批止盈"

        return {
            'buy_point': buy_point,
            'sell_point': sell_point
        }

    def _analyze_hold_period(self, df: pd.DataFrame,
                              tech: Dict, blogger_cycle: str,
                              timing_offset: float = 0.0) -> Dict:
        """分析持股周期"""
        # 根据博主周期调整
        if '超短线' in blogger_cycle:
            days = 2
        elif '短线' in blogger_cycle:
            days = 5
        elif '波段' in blogger_cycle:
            days = 15
        elif '中长线' in blogger_cycle:
            days = 60
        else:
            days = 10

        # 根据趋势调整预期收益 — 修正正负不对称(原bull=+0.5%, bear=-0.3%)
        trend = tech.get('trend', '震荡')
        if trend == '上涨':
            daily_return = 0.5
        elif trend == '下跌':
            daily_return = -0.5
        else:
            daily_return = 0.1

        # 应用择时偏移
        daily_return += timing_offset * 0.3

        total_return = days * daily_return

        return {
            'days': days,
            'return': round(total_return, 2)
        }

    def _generate_reason(self, tech: Dict, trend: Dict) -> str:
        """生成选股原因"""
        reasons = []

        if tech.get('is_above_ma5'):
            reasons.append("股价站上5日均线")
        if tech.get('is_above_ma20'):
            reasons.append("股价站上20日均线")
        if tech.get('trend') == '上涨':
            reasons.append("上升趋势明确")
        if tech.get('volume_change_pct', 0) > 30:
            reasons.append("成交量放大")

        if reasons:
            return "技术面：" + "；".join(reasons)
        return "符合筛选条件"

    def _generate_risk(self, tech: Dict, amplitude: Dict,
                        volatility_data: Dict = None) -> str:
        """生成风险提示 — 强制包含64%胜率回撤风险"""
        risks = []

        # 始终包含的核心风险提示
        risks.append("历史胜率约64%，单次判断失败概率36%，请严格止损")
        risks.append("64%胜率下连续3次失败概率约4.7%，建议单笔仓位≤总资金5%")

        # 动态胜率
        try:
            from app.selector.stock_selector import get_dynamic_win_rate
            dyn = get_dynamic_win_rate()
            wr = dyn['win_rate'] or dyn['backtest_win_rate']
            signal = dyn['signal_strength']
            if dyn['total_samples'] > 0:
                risks.append(f"系统近期胜率{wr}%，非确定性预测")
            else:
                risks.append(f"系统回测胜率{wr}%（无实盘验证），非确定性预测")
            if signal == 'suspended':
                risks.append("⚠️ 荐股信号已暂停，请勿跟投")
            elif signal == 'degraded':
                risks.append("⚠️ 荐股信号降级，建议减仓观望")
        except Exception:
            pass

        if amplitude['amplitude'] > 5:
            risks.append("波动较大，注意仓位控制")
        if not tech.get('is_above_ma20'):
            risks.append("股价低于20日均线，趋势偏弱")
        if tech.get('trend') == '下跌':
            risks.append("当前下跌趋势，谨慎介入")

        # 【改进】增加波动性风险提示
        if volatility_data:
            volatility = volatility_data.get('volatility', 0)
            if volatility > 40:  # 年化波动率超过40%
                risks.append("个股波动性较高，风险较大")
            atr_pct = volatility_data.get('atr_pct', 0)
            if atr_pct > 5:
                risks.append(f"ATR指标显示近期波动较大({atr_pct:.1f}%)")

        if risks:
            return "风险提示：" + "；".join(risks)
        return "暂无明显风险"

    def batch_predict(self, codes: List[str],
                      blogger_cycle: str = "波段（1-4周）",
                      use_factor_model: bool = True) -> List[DailyPrediction]:
        """批量预测 - 改进版，支持多因子评分"""
        predictions = []

        # 如果启用多因子模型，先进行评分筛选
        if use_factor_model:
            try:
                from .multi_factor_model import MultiFactorModel, prepare_stock_data_for_scoring

                model = MultiFactorModel()
                stocks_data = []

                for code in codes:
                    data = prepare_stock_data_for_scoring(code, self.market_client)
                    name = data.get('name', code)
                    stocks_data.append({'code': code, 'name': name, 'data': data})

                # 评分并排序
                scores = model.score_stocks(stocks_data)

                # 只预测评分较高的股票
                top_scores = model.get_top_stocks(scores, top_n=min(len(codes), 10))

                for score in top_scores:
                    pred = self.predict(score.code, blogger_cycle=blogger_cycle)
                    if pred:
                        # 将因子评分添加到预测中
                        pred.technical_basis['factor_score'] = score.total_score
                        pred.technical_basis['factor_scores'] = {
                            fs.factor_name: fs.normalized_score for fs in score.factor_scores
                        }
                        pred.selection_reason = f"多因子评分{score.total_score:.1f}分({score.recommendation})，{pred.selection_reason}"
                        predictions.append(pred)

                print(f"多因子筛选：从{len(codes)}只股票中选出{len(predictions)}只进行预测")

            except Exception as e:
                print(f"多因子评分失败，使用传统预测: {e}")
                for code in codes:
                    pred = self.predict(code, blogger_cycle=blogger_cycle)
                    if pred:
                        predictions.append(pred)
        else:
            for code in codes:
                pred = self.predict(code, blogger_cycle=blogger_cycle)
                if pred:
                    predictions.append(pred)

        return predictions

    def save_predictions_to_db(self, predictions: List[DailyPrediction],
                                blogger_name: str = None,
                                db_path: str = None):
        """保存预测到数据库

        Args:
            predictions: 预测列表
            blogger_name: 博主名称
            db_path: 数据库路径
        """
        conn = get_sync_db(db_path)
        cursor = conn.cursor()

        # 清空该博主对次日的旧预测（如果有指定博主）
        target_date = date.today() + timedelta(days=1)
        if blogger_name:
            cursor.execute("DELETE FROM predictions WHERE prediction_date = ? AND blogger_name = ?", (target_date, blogger_name))
        else:
            cursor.execute("DELETE FROM predictions WHERE prediction_date = ?", (target_date,))

        # 确保新列存在
        for col, default in [('confidence', '0.0'), ('confidence_interval', 'NULL'), ('prediction_source', "'rule_based'")]:
            try:
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col} TEXT DEFAULT {default}")
            except sqlite3.OperationalError:
                pass

        # 插入新预测
        for p in predictions:
            cursor.execute("""
                INSERT INTO predictions
                (stock_code, stock_name, prediction_date, prediction_time,
                 current_price, expected_open_change, expected_amplitude, expected_high, expected_low,
                 expected_close_change, expected_trend, expected_strength,
                 expected_hold_days, expected_hold_return, expected_buy_point, expected_sell_point, expected_sell_price,
                 selection_reason, risk_warning, technical_basis, market_basis, blogger_name,
                 confidence, confidence_interval, prediction_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.code, p.name, p.prediction_date, p.prediction_time,
                p.current_price,
                p.expected_open_change, p.expected_amplitude, p.expected_high, p.expected_low,
                p.expected_close_change, p.expected_trend, p.expected_strength,
                p.expected_hold_days, p.expected_hold_return, p.expected_buy_point, p.expected_sell_point, p.expected_sell_price,
                p.selection_reason, p.risk_warning,
                json.dumps(p.technical_basis, ensure_ascii=False, default=str),
                json.dumps(p.market_basis, ensure_ascii=False, default=str),
                blogger_name,
                getattr(p, 'confidence', 0.0),
                getattr(p, 'confidence_interval', ''),
                getattr(p, 'prediction_source', 'rule_based')
            ))

        conn.commit()
        conn.close()
        print(f"已保存 {len(predictions)} 条预测到数据库")


def generate_prediction_report(predictions: List[DailyPrediction]) -> str:
    """生成预测报告"""
    target_date = date.today() + timedelta(days=1)
    report = f"""
# 📈 每日股票预测报告

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
> 预测目标日期：{target_date}（次日）

---

## 预测股票列表

"""

    for i, p in enumerate(predictions, 1):
        report += f"""
### {i}. {p.code} {p.name}

**当日预测**
- 预期走势：{p.expected_trend}（强度：{p.expected_strength}）
- 预期振幅：{p.expected_amplitude:.2f}%
- 预期涨跌：{p.expected_close_change:+.2f}%
- 预期最高：{p.expected_high:.2f}元
- 预期最低：{p.expected_low:.2f}元

**持股周期**
- 建议持有：{p.expected_hold_days}天
- 预期收益：{p.expected_hold_return:+.2f}%

**操作建议**
- 买入点位：{p.expected_buy_point}
- 卖出点位：{p.expected_sell_point}

**选股原因**
> {p.selection_reason}

**风险提示**
> {p.risk_warning}

---

"""

    report += """
## 📝 分析说明

本预测基于技术指标分析，仅供参考：
- 预测结果不构成投资建议
- 请结合自身判断做出决策
- 投资有风险，入市需谨慎
"""

    return report


if __name__ == "__main__":
    # 测试预测器
    predictor = Predictor()

    # 预测几只热门股票
    test_codes = ['300750', '300308', '600519']  # 宁德时代、中际旭创、贵州茅台

    predictions = predictor.batch_predict(test_codes)

    print(generate_prediction_report(predictions))