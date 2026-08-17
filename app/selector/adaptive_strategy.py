"""自适应策略框架
根据市场状态(bull/bear/oscillation)动态切换策略参数：
- 止损止盈阈值
- 持股周期
- 仓位风格
- 因子权重偏移
- 选股筛选条件
- 全市场临时止损触发
- 最长持仓强制离场
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StrategyProfile:
    """策略配置"""
    name: str
    regime: str  # bull / bear / oscillation

    # 止损止盈
    stop_loss_pct: float       # 止损百分比(负数)
    take_profit_pct: float     # 止盈百分比(正数)
    trailing_stop_pct: float   # 移动止损百分比(正数)

    # 持股周期
    min_hold_days: int
    max_hold_days: int
    default_hold_days: int

    # 仓位风格
    max_total_position: float  # 最大总仓位(%)
    single_position: float     # 单票仓位(%)
    max_positions: int         # 最大持仓数

    # 因子权重偏移(叠加在regime_weights上)
    weight_adjustment: Dict[str, float] = field(default_factory=dict)

    # 筛选条件偏移
    fundamental_min_score: float = 60.0   # 基本面最低分
    valuation_max_percentile: float = 80.0  # 估值最大百分位
    min_roe: float = 8.0                  # 最低ROE
    min_revenue_growth: float = 0.0       # 最低营收增速

    # 描述
    description: str = ""


# ── 三套策略配置 ──

BULL_STRATEGY = StrategyProfile(
    name='牛市进攻策略',
    regime='bull',
    stop_loss_pct=-8.0,
    take_profit_pct=15.0,
    trailing_stop_pct=5.0,
    min_hold_days=3,
    max_hold_days=20,
    default_hold_days=10,
    max_total_position=85.0,
    single_position=5.0,
    max_positions=15,
    weight_adjustment={
        'momentum': 0.05,      # 牛市加动量
        'trend': 0.03,         # 加趋势
        'valuation': -0.03,    # 估值可放宽
    },
    fundamental_min_score=50.0,
    valuation_max_percentile=90.0,
    min_roe=5.0,
    min_revenue_growth=-5.0,
    description='牛市追涨：放宽止损到-8%，提高止盈到+15%，允许追高，放宽基本面门槛',
)

BEAR_STRATEGY = StrategyProfile(
    name='熊市防御策略',
    regime='bear',
    stop_loss_pct=-3.0,
    take_profit_pct=8.0,
    trailing_stop_pct=2.0,
    min_hold_days=1,
    max_hold_days=10,
    default_hold_days=5,
    max_total_position=30.0,
    single_position=3.0,
    max_positions=8,
    weight_adjustment={
        'fundamental': 0.05,   # 熊市重基本面
        'valuation': 0.05,     # 重估值安全
        'momentum': -0.05,     # 减动量(追涨危险)
        'sentiment': -0.03,    # 减情绪(恐慌中情绪不可靠)
    },
    fundamental_min_score=70.0,
    valuation_max_percentile=50.0,
    min_roe=15.0,
    min_revenue_growth=5.0,
    description='熊市防守：收紧止损到-3%，只选ROE>15%+营收增长>5%的防御股，低仓位',
)

OSCILLATION_STRATEGY = StrategyProfile(
    name='震荡市均值回归策略',
    regime='oscillation',
    stop_loss_pct=-5.0,
    take_profit_pct=10.0,
    trailing_stop_pct=3.0,
    min_hold_days=2,
    max_hold_days=15,
    default_hold_days=7,
    max_total_position=60.0,
    single_position=4.0,
    max_positions=12,
    weight_adjustment={
        'capital_flow': 0.03,  # 震荡市重资金流向
        'cognitive_gap': 0.02, # 认知差机会
        'sentiment': -0.02,    # 情绪偏弱
    },
    fundamental_min_score=60.0,
    valuation_max_percentile=70.0,
    min_roe=8.0,
    min_revenue_growth=0.0,
    description='震荡市快进快出：持股周期缩短，关注均值回归和事件驱动，中仓位',
)

STRATEGY_MAP = {
    'bull': BULL_STRATEGY,
    'bear': BEAR_STRATEGY,
    'oscillation': OSCILLATION_STRATEGY,
}


class AdaptiveStrategyManager:
    """自适应策略管理器"""

    def __init__(self):
        self._current_regime = None
        self._current_profile = None
        self._regime_history: List[Dict] = []

    def get_strategy(self, regime: str) -> StrategyProfile:
        """获取当前市场状态对应的策略配置"""
        profile = STRATEGY_MAP.get(regime, OSCILLATION_STRATEGY)
        self._current_regime = regime
        self._current_profile = profile
        return profile

    def apply_weight_adjustment(self, base_weights: Dict[str, float],
                                 regime: str) -> Dict[str, float]:
        """将策略的权重偏移叠加到基础因子权重上

        Args:
            base_weights: _get_regime_weights返回的基础权重
            regime: 市场状态

        Returns:
            调整后的权重(归一化到总和=1)
        """
        profile = self.get_strategy(regime)
        adjusted = dict(base_weights)

        for factor, delta in profile.weight_adjustment.items():
            if factor in adjusted:
                adjusted[factor] = max(0, adjusted[factor] + delta)

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def apply_fundamental_filter(self, candidates: List[Dict],
                                  regime: str) -> List[Dict]:
        """根据策略调整基本面筛选门槛

        Args:
            candidates: 候选股票列表
            regime: 市场状态

        Returns:
            筛选后的候选列表
        """
        profile = self.get_strategy(regime)
        filtered = []

        for s in candidates:
            fund_score = s.get('fundamental_score', 0)
            pe_pct = s.get('pe_percentile', 100)
            roe = s.get('roe', 0)
            rev_growth = s.get('revenue_growth', 0)

            # 基本面评分门槛
            if fund_score < profile.fundamental_min_score:
                continue

            # 估值百分位门槛
            if pe_pct > profile.valuation_max_percentile:
                continue

            # ROE门槛(如果有数据)
            if roe > 0 and roe < profile.min_roe:
                continue

            # 营收增速门槛(如果有数据)
            if rev_growth != 0 and rev_growth < profile.min_revenue_growth:
                continue

            filtered.append(s)

        # 如果筛选后为空，放宽条件保留原候选(避免空仓)
        if not filtered and candidates:
            return candidates

        return filtered

    def calculate_atr_stop_loss(self, price: float, atr: float,
                                 regime: str) -> Tuple[float, float]:
        """基于ATR计算动态止损止盈

        Args:
            price: 当前价格
            atr: 真实波幅(ATR)
            regime: 市场状态

        Returns:
            (stop_loss_price, take_profit_price)
        """
        profile = self.get_strategy(regime)

        if atr <= 0 or price <= 0:
            # ATR无效时用固定百分比
            stop_loss = price * (1 + profile.stop_loss_pct / 100)
            take_profit = price * (1 + profile.take_profit_pct / 100)
            return round(stop_loss, 2), round(take_profit, 2)

        # ATR倍数根据策略调整
        if regime == 'bull':
            sl_atr_mult = 2.0   # 牛市：2倍ATR止损(宽)
            tp_atr_mult = 3.0   # 3倍ATR止盈
        elif regime == 'bear':
            sl_atr_mult = 1.5   # 熊市：1.5倍ATR止损(紧)
            tp_atr_mult = 2.0   # 2倍ATR止盈
        else:
            sl_atr_mult = 2.0   # 震荡：2倍ATR
            tp_atr_mult = 2.5

        stop_loss = price - sl_atr_mult * atr
        take_profit = price + tp_atr_mult * atr

        # 钳位：不超过策略的固定百分比上下限
        max_sl = price * (1 + profile.stop_loss_pct / 100)
        max_tp = price * (1 + profile.take_profit_pct / 100)
        stop_loss = max(stop_loss, max_sl)  # 止损价取较高者(更保守)
        take_profit = min(take_profit, max_tp)  # 止盈价取较低者(更保守)

        return round(max(stop_loss, 0), 2), round(take_profit, 2)

    def calculate_trailing_stop(self, entry_price: float, highest_price: float,
                                 regime: str) -> float:
        """计算移动止损价

        Args:
            entry_price: 买入价
            highest_price: 持仓期间最高价
            regime: 市场状态

        Returns:
            移动止损价
        """
        profile = self.get_strategy(regime)
        trailing_price = highest_price * (1 - profile.trailing_stop_pct / 100)

        # 不低于买入价的固定止损
        fixed_stop = entry_price * (1 + profile.stop_loss_pct / 100)
        return round(max(trailing_price, fixed_stop), 2)

    def get_position_guidance(self, regime: str, signal_strength: str = 'normal') -> Dict:
        """获取仓位指引

        Args:
            regime: 市场状态
            signal_strength: 信号强度(normal/degraded/suspended)

        Returns:
            Dict with position guidance
        """
        profile = self.get_strategy(regime)
        max_pos = profile.max_total_position

        # 信号强度折扣
        if signal_strength == 'suspended':
            max_pos *= 0.3
        elif signal_strength == 'degraded':
            max_pos *= 0.6

        return {
            'max_total_position': round(max_pos, 1),
            'single_position': profile.single_position,
            'max_positions': profile.max_positions,
            'strategy_name': profile.name,
            'stop_loss_pct': profile.stop_loss_pct,
            'take_profit_pct': profile.take_profit_pct,
            'trailing_stop_pct': profile.trailing_stop_pct,
            'default_hold_days': profile.default_hold_days,
            'description': profile.description,
        }

    # 全市场临时止损阈值
    MARKET_CIRCUIT_BREAKER_PCT = -3.0  # 单日跌幅超-3%触发全市场减仓

    # 择时状态下的单票仓位上限
    TIMING_SINGLE_POSITION_MAP = {
        'full_long': 0.05,
        'cautious_long': 0.04,
        'neutral': 0.03,
        'cautious_short': 0.02,
        'full_short': 0.02,
    }

    # 最长持仓强制离场天数
    MAX_HOLD_DAYS_HARD_LIMIT = 30  # 超过30天无条件清仓
    MAX_HOLD_DAYS_REGIME = {
        'bull': 20,
        'bear': 10,
        'oscillation': 15,
    }

    def check_market_circuit_breaker(self, index_change_pct: float) -> Dict:
        """全市场临时止损检查

        当沪深300单日跌幅超-3%时，触发全市场减仓指令

        Args:
            index_change_pct: 沪深300当日涨跌幅

        Returns:
            Dict with breaker status and action
        """
        triggered = index_change_pct <= self.MARKET_CIRCUIT_BREAKER_PCT
        return {
            'triggered': triggered,
            'index_change': index_change_pct,
            'threshold': self.MARKET_CIRCUIT_BREAKER_PCT,
            'action': 'reduce_all_positions_to_half' if triggered else 'none',
            'message': f"⚠️ 沪深300单日{index_change_pct:+.2f}%，触发全市场临时止损！建议所有持仓减半" if triggered else "市场正常",
        }

    def check_max_hold_days(self, hold_days: int, regime: str) -> Dict:
        """最长持仓强制离场检查

        Args:
            hold_days: 已持有天数
            regime: 当前市场状态

        Returns:
            Dict with hold check result
        """
        regime_limit = self.MAX_HOLD_DAYS_REGIME.get(regime, 15)
        hard_limit = self.MAX_HOLD_DAYS_HARD_LIMIT

        if hold_days >= hard_limit:
            return {
                'force_exit': True,
                'reason': f'持仓{hold_days}天超硬限制{hard_limit}天，无条件清仓',
                'urgency': 'immediate',
            }
        elif hold_days >= regime_limit:
            return {
                'force_exit': True,
                'reason': f'持仓{hold_days}天超{regime}限制{regime_limit}天，建议清仓',
                'urgency': 'high',
            }
        elif hold_days >= regime_limit - 3:
            return {
                'force_exit': False,
                'reason': f'持仓{hold_days}天接近{regime}限制{regime_limit}天，请准备离场',
                'urgency': 'warning',
            }
        return {
            'force_exit': False,
            'reason': '',
            'urgency': 'none',
        }

    def get_timing_single_position(self, timing_level: str) -> float:
        """根据择时等级获取单票仓位上限

        Args:
            timing_level: 择时等级(full_long/cautious_long/neutral/cautious_short/full_short)

        Returns:
            单票仓位上限百分比
        """
        return self.TIMING_SINGLE_POSITION_MAP.get(timing_level, 0.03) * 100
        """获取当前策略状态摘要"""
        if not self._current_profile:
            return {'status': '未初始化'}

        return {
            'regime': self._current_regime,
            'strategy': self._current_profile.name,
            'description': self._current_profile.description,
            'stop_loss': f"{self._current_profile.stop_loss_pct}%",
            'take_profit': f"+{self._current_profile.take_profit_pct}%",
            'max_position': f"{self._current_profile.max_total_position}%",
            'hold_days': f"{self._current_profile.min_hold_days}-{self._current_profile.max_hold_days}天",
        }
