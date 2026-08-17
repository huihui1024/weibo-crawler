"""交易决策引擎
从"选股"到"荐股"的关键跨越：回答"现在该买什么、买多少、什么时候卖、错了怎么办"

核心模块：
1. 买入时机判定：技术面+资金面确认，时机评分>70才发信号
2. 动态调仓：板块轮动检测 + 信号衰减 + 组合再平衡
3. 卖出决策：信号反转 + 机会成本 + 时间衰减
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_sync_db
from app.market import get_market_client


# ── 数据结构 ──

@dataclass
class BuySignal:
    """买入信号"""
    code: str
    name: str
    signal_type: str           # "breakout"/"pullback"/"capital_inflow"/"momentum"
    timing_score: float        # 时机评分 0-100
    entry_price: float         # 建议买入价
    stop_loss_price: float     # 止损价
    take_profit_price: float   # 止盈价
    position_pct: float        # 建议仓位
    reasons: List[str] = field(default_factory=list)
    confidence: str = "medium"  # "high"/"medium"/"low"


@dataclass
class SellSignal:
    """卖出信号"""
    code: str
    name: str
    signal_type: str           # "stop_loss"/"signal_reversal"/"opportunity_cost"/"time_decay"/"take_profit"
    urgency: str               # "immediate"/"normal"/"watch"
    suggested_price: float     # 建议卖出价
    reasons: List[str] = field(default_factory=list)
    hold_days: int = 0
    profit_pct: float = 0.0


@dataclass
class RebalanceAction:
    """调仓动作"""
    code: str
    name: str
    action: str                # "increase"/"decrease"/"add"/"remove"/"hold"
    target_position_pct: float
    current_position_pct: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class DecisionReport:
    """决策报告"""
    date: str
    regime: str
    strategy_name: str
    buy_signals: List[BuySignal] = field(default_factory=list)
    sell_signals: List[SellSignal] = field(default_factory=list)
    rebalance_actions: List[RebalanceAction] = field(default_factory=list)
    portfolio_actions: List[str] = field(default_factory=list)  # 组合级操作建议
    risk_warnings: List[str] = field(default_factory=list)


class TradingDecisionEngine:
    """交易决策引擎"""

    # 买入时机评分阈值
    TIMING_SCORE_THRESHOLD = 70

    # 信号衰减检测
    SIGNAL_DECAY_DAYS = 3       # 连续3天评分下降
    SIGNAL_DECAY_THRESHOLD = 10  # 评分下降超过10分

    # 时间衰减
    TIME_DECAY_MULTIPLIER = 2.0  # 持仓超过预期周期2倍

    # 板块轮动
    SECTOR_MOMENTUM_DAYS = 5    # 5日动量
    SECTOR_ROTATION_THRESHOLD = -2.0  # 板块动量<-2%触发减仓

    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'data/weibo.db'
        self.market_client = get_market_client()

    # ── 1. 买入时机判定 ──

    def evaluate_buy_timing(self, code: str, name: str = '',
                            tech: Dict = None, flow: Dict = None,
                            regime: str = 'oscillation') -> Optional[BuySignal]:
        """评估买入时机

        三维确认：技术面 + 资金面 + 因子面
        时机评分 = 技术面得分×0.4 + 资金面得分×0.35 + 因子面得分×0.25
        >70分发出买入信号
        """
        if tech is None:
            try:
                tech = self.market_client.get_stock_technical_indicators(code)
            except Exception:
                tech = {}

        if flow is None:
            try:
                flow = self.market_client.get_stock_capital_flow(code)
            except Exception:
                flow = {}

        # 技术面得分(0-100)
        tech_score = self._score_technical_entry(tech, regime)

        # 资金面得分(0-100)
        flow_score = self._score_capital_entry(flow)

        # 因子面得分(0-100) — 从股票池获取
        factor_score = self._score_factor_entry(code)

        # 综合时机评分
        timing_score = tech_score * 0.4 + flow_score * 0.35 + factor_score * 0.25

        # 确定信号类型
        signal_type = self._determine_entry_signal(tech, flow, tech_score, flow_score)

        # 生成买入理由
        reasons = []
        if tech_score >= 70:
            reasons.append(f"技术面确认({tech_score:.0f}分)")
        elif tech_score >= 50:
            reasons.append(f"技术面中性({tech_score:.0f}分)")
        else:
            reasons.append(f"技术面偏弱({tech_score:.0f}分)")

        if flow_score >= 70:
            reasons.append(f"资金面强劲({flow_score:.0f}分)")
        elif flow_score >= 50:
            reasons.append(f"资金面中性({flow_score:.0f}分)")

        if factor_score >= 70:
            reasons.append(f"因子面优秀({factor_score:.0f}分)")

        # 置信度
        if timing_score >= 80:
            confidence = "high"
        elif timing_score >= 70:
            confidence = "medium"
        else:
            confidence = "low"

        # 获取当前价格和ATR
        price = tech.get('current_price', 0)
        if price <= 0:
            try:
                df = self.market_client.get_stock_history(code, days=1)
                if not df.empty:
                    price = df.iloc[-1]['close']
            except Exception:
                price = 0

        if price <= 0:
            return None

        atr = tech.get('atr', tech.get('amplitude_20d', 20) * price / 100)

        # 自适应止损止盈
        from app.selector.adaptive_strategy import AdaptiveStrategyManager
        asm = AdaptiveStrategyManager()
        sl_price, tp_price = asm.calculate_atr_stop_loss(price, atr, regime)

        # 仓位：基于时机评分
        position_pct = self._calculate_entry_position(timing_score, regime)

        return BuySignal(
            code=code,
            name=name or code,
            signal_type=signal_type,
            timing_score=round(timing_score, 1),
            entry_price=price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            position_pct=position_pct,
            reasons=reasons,
            confidence=confidence,
        )

    def _score_technical_entry(self, tech: Dict, regime: str) -> float:
        """技术面买入时机评分"""
        score = 50  # 基础分

        # 趋势确认(±15)
        trend = tech.get('trend', '')
        if trend == '上涨':
            score += 15
        elif trend == '下跌':
            score -= 15

        # MA位置(±10)
        if tech.get('is_above_ma20'):
            score += 10
            if tech.get('is_above_ma5'):
                score += 5  # MA5>MA20多头发散
        else:
            score -= 10

        # 突破信号(+15)
        if tech.get('near_resistance_break'):  # 接近/突破阻力位
            score += 15

        # 放量确认(+10)
        vol_change = tech.get('volume_change_pct', 0) or 0
        if vol_change > 30:  # 成交量放大30%+
            score += 10
        elif vol_change > 10:
            score += 5

        # 回踩支撑(+10)
        if tech.get('near_support') and trend != '下跌':
            score += 10  # 回踩支撑企稳

        # RSI适中(±5)
        rsi = tech.get('rsi', 50) or 50
        if 40 <= rsi <= 60:
            score += 5  # RSI适中，不超买不超卖
        elif rsi > 80:
            score -= 10  # 超买风险
        elif rsi < 20:
            score += 5  # 超卖反弹机会

        # 牛市放宽技术面要求
        if regime == 'bull':
            if trend == '上涨':
                score += 5  # 牛市追涨加分

        return max(0, min(100, score))

    def _score_capital_entry(self, flow: Dict) -> float:
        """资金面买入时机评分"""
        score = 50

        # 主力净流入(±15)
        net_inflow = flow.get('net_inflow', 0) or 0
        if net_inflow > 0:
            score += min(15, int(net_inflow / 1e7))  # 每1000万+1分，上限15
        elif net_inflow < 0:
            score -= min(15, int(abs(net_inflow) / 1e7))

        # 北向资金(+10)
        north_flow = flow.get('north_flow', 0) or 0
        if north_flow > 0:
            score += 10

        # 连续净流入(+10)
        consecutive_inflow = flow.get('consecutive_inflow_days', 0) or 0
        if consecutive_inflow >= 3:
            score += 10
        elif consecutive_inflow >= 2:
            score += 5

        # 大单占比(+5)
        big_order_ratio = flow.get('big_order_ratio', 0.5) or 0.5
        if big_order_ratio > 0.6:
            score += 5

        return max(0, min(100, score))

    def _score_factor_entry(self, code: str) -> float:
        """因子面评分 — 从stock_pool获取综合评分"""
        try:
            conn = get_sync_db(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT style_match_score, fundamental_score, capital_flow_score
                FROM stock_pool WHERE stock_code = ? AND status = 'active'
                LIMIT 1
            """, (code,))
            row = cursor.fetchone()
            conn.close()
            if row:
                style = row[0] or 50
                fund = row[1] or 50
                flow = row[2] or 50
                return (style * 0.4 + fund * 0.35 + flow * 0.25)
        except Exception:
            pass
        return 50

    def _determine_entry_signal(self, tech: Dict, flow: Dict,
                                 tech_score: float, flow_score: float) -> str:
        """确定买入信号类型"""
        # 突破信号：技术面强+价格接近阻力位
        if tech_score >= 70 and tech.get('near_resistance_break'):
            return "breakout"

        # 回踩信号：MA上方+接近支撑位
        if tech.get('is_above_ma20') and tech.get('near_support'):
            return "pullback"

        # 资金流入信号：资金面强
        if flow_score >= 70:
            return "capital_inflow"

        # 动量信号：趋势向上+量能配合
        if tech.get('trend') == '上涨' and (tech.get('volume_change_pct', 0) or 0) > 10:
            return "momentum"

        return "general"

    def _calculate_entry_position(self, timing_score: float, regime: str) -> float:
        """基于时机评分计算建议仓位"""
        from app.selector.adaptive_strategy import AdaptiveStrategyManager
        asm = AdaptiveStrategyManager()
        guidance = asm.get_position_guidance(regime)
        max_single = guidance['single_position']

        if timing_score >= 85:
            return max_single           # 满配
        elif timing_score >= 75:
            return max_single * 0.7     # 七成配
        elif timing_score >= 70:
            return max_single * 0.5     # 半配
        else:
            return 0                    # 不买入

    # ── 2. 卖出决策 ──

    def evaluate_sell_timing(self, code: str, name: str = '',
                              hold_days: int = 0, profit_pct: float = 0,
                              entry_price: float = 0, current_price: float = 0,
                              highest_price: float = 0,
                              expected_hold_days: int = 10,
                              regime: str = 'oscillation') -> Optional[SellSignal]:
        """评估卖出时机

        四类卖出信号：
        1. 止损止盈 — 固定规则
        2. 信号反转 — 基本面/技术面信号反转
        3. 机会成本 — 出现更好标的
        4. 时间衰减 — 横盘过久
        """
        from app.selector.adaptive_strategy import AdaptiveStrategyManager
        asm = AdaptiveStrategyManager()
        profile = asm.get_strategy(regime)

        reasons = []
        signal_type = None
        urgency = "normal"

        # 1. 止损止盈检查
        if profit_pct <= profile.stop_loss_pct:
            signal_type = "stop_loss"
            urgency = "immediate"
            reasons.append(f"触发止损({profit_pct:+.1f}%<{profile.stop_loss_pct}%)")
        elif profit_pct >= profile.take_profit_pct:
            signal_type = "take_profit"
            urgency = "normal"
            reasons.append(f"触发止盈({profit_pct:+.1f}%>{profile.take_profit_pct}%)")

        # 移动止损：从最高点回落超过trailing_stop_pct
        if signal_type is None and highest_price > entry_price and current_price > 0:
            from_peak = (highest_price - current_price) / highest_price * 100
            if from_peak >= profile.trailing_stop_pct and profit_pct > 0:
                signal_type = "take_profit"
                urgency = "normal"
                reasons.append(f"移动止损(从峰值回落{from_peak:.1f}%>{profile.trailing_stop_pct}%)")

        # 2. 信号反转检查
        if signal_type is None:
            reversal = self._check_signal_reversal(code)
            if reversal:
                signal_type = "signal_reversal"
                urgency = "normal"
                reasons.append(reversal)

        # 3. 时间衰减检查
        if signal_type is None and hold_days > expected_hold_days * self.TIME_DECAY_MULTIPLIER:
            # 持仓超过预期周期2倍且涨幅不大
            if abs(profit_pct) < 3:  # 涨跌幅<3%
                signal_type = "time_decay"
                urgency = "watch"
                reasons.append(f"持仓{hold_days}天超过预期{expected_hold_days}天的{self.TIME_DECAY_MULTIPLIER}倍，且涨幅仅{profit_pct:+.1f}%")

        # 4. 机会成本检查(简化：通过stock_pool排名)
        if signal_type is None and hold_days > expected_hold_days:
            opportunity = self._check_opportunity_cost(code)
            if opportunity:
                signal_type = "opportunity_cost"
                urgency = "watch"
                reasons.append(opportunity)

        if signal_type is None:
            return None

        return SellSignal(
            code=code,
            name=name or code,
            signal_type=signal_type,
            urgency=urgency,
            suggested_price=current_price if current_price > 0 else entry_price,
            reasons=reasons,
            hold_days=hold_days,
            profit_pct=profit_pct,
        )

    def _check_signal_reversal(self, code: str) -> Optional[str]:
        """检查信号反转"""
        try:
            tech = self.market_client.get_stock_technical_indicators(code)

            # 技术面反转：趋势从上涨转为下跌
            if tech.get('trend') == '下跌' and not tech.get('is_above_ma20'):
                return "技术面反转：跌破MA20且趋势转跌"

            # 放量下跌
            vol_change = tech.get('volume_change_pct', 0) or 0
            if tech.get('trend') == '下跌' and vol_change > 30:
                return "技术面反转：放量下跌"

            # 资金面反转：主力连续净流出
            try:
                flow = self.market_client.get_stock_capital_flow(code)
                net_inflow = flow.get('net_inflow', 0) or 0
                consecutive_outflow = flow.get('consecutive_outflow_days', 0) or 0
                if consecutive_outflow >= 3 and net_inflow < 0:
                    return f"资金面反转：主力连续{consecutive_outflow}天净流出"
            except Exception:
                pass

        except Exception:
            pass

        return None

    def _check_opportunity_cost(self, code: str) -> Optional[str]:
        """检查机会成本：是否有更好的标的"""
        try:
            conn = get_sync_db(self.db_path)
            cursor = conn.cursor()

            # 获取当前股票评分
            cursor.execute("""
                SELECT style_match_score FROM stock_pool
                WHERE stock_code = ? AND status = 'active'
            """, (code,))
            current = cursor.fetchone()

            if not current:
                conn.close()
                return None

            current_score = current[0] or 50

            # 检查是否有评分更高的股票不在持仓中
            cursor.execute("""
                SELECT stock_code, stock_name, style_match_score
                FROM stock_pool WHERE status = 'active'
                AND style_match_score > ?
                ORDER BY style_match_score DESC LIMIT 3
            """, (current_score + 10,))

            better = cursor.fetchall()
            conn.close()

            if better:
                names = [f"{b[1]}({b[2]:.0f}分)" for b in better]
                return f"存在更优标的: {', '.join(names)}"

        except Exception:
            pass

        return None

    # ── 3. 动态调仓 ──

    def evaluate_rebalance(self, positions: List[Dict],
                            regime: str = 'oscillation') -> List[RebalanceAction]:
        """评估组合调仓需求

        三类调仓触发：
        1. 板块轮动：持仓板块动量衰减→减仓
        2. 信号衰减：持仓股因子评分连续下降→减仓
        3. 组合再平衡：偏离目标权重过大
        """
        actions = []

        if not positions:
            return actions

        # 1. 板块轮动检测
        sector_momentum = self._calculate_sector_momentum(positions)
        for pos in positions:
            sector = pos.get('sector', '')
            momentum = sector_momentum.get(sector, 0)
            if momentum < self.SECTOR_ROTATION_THRESHOLD:
                actions.append(RebalanceAction(
                    code=pos['code'],
                    name=pos.get('name', ''),
                    action='decrease',
                    target_position_pct=max(0, pos.get('position_pct', 0) * 0.5),
                    current_position_pct=pos.get('position_pct', 0),
                    reasons=[f"板块轮动: {sector}近{self.SECTOR_MOMENTUM_DAYS}日动量{momentum:+.1f}%"]
                ))

        # 2. 信号衰减检测
        for pos in positions:
            decay = self._check_signal_decay(pos['code'])
            if decay:
                actions.append(RebalanceAction(
                    code=pos['code'],
                    name=pos.get('name', ''),
                    action='decrease',
                    target_position_pct=max(0, pos.get('position_pct', 0) * 0.5),
                    current_position_pct=pos.get('position_pct', 0),
                    reasons=[decay]
                ))

        # 3. 组合再平衡
        rebalance_actions = self._check_portfolio_rebalance(positions, regime)
        actions.extend(rebalance_actions)

        # 去重：同一股票取最激进的action
        deduped = {}
        for a in actions:
            if a.code not in deduped:
                deduped[a.code] = a
            else:
                existing = deduped[a.code]
                # remove > decrease > hold > increase > add
                action_priority = {'remove': 5, 'decrease': 4, 'hold': 3, 'increase': 2, 'add': 1}
                if action_priority.get(a.action, 0) > action_priority.get(existing.action, 0):
                    deduped[a.code] = a
                # 合并理由
                else:
                    existing.reasons.extend(a.reasons)

        return list(deduped.values())

    def _calculate_sector_momentum(self, positions: List[Dict]) -> Dict[str, float]:
        """计算持仓板块近N日动量"""
        sectors = set(p.get('sector', '') for p in positions if p.get('sector'))
        momentum = {}

        for sector in sectors:
            try:
                # 用板块指数近5日涨跌幅近似
                # 简化：用持仓股的平均涨跌幅
                sector_stocks = [p for p in positions if p.get('sector') == sector]
                changes = []
                for s in sector_stocks:
                    change = s.get('change_pct', 0) or 0
                    changes.append(change)
                momentum[sector] = np.mean(changes) if changes else 0
            except Exception:
                momentum[sector] = 0

        return momentum

    def _check_signal_decay(self, code: str) -> Optional[str]:
        """检查持仓股信号衰减"""
        try:
            conn = get_sync_db(self.db_path)
            cursor = conn.cursor()

            # 从daily_reviews中获取最近评分趋势
            cursor.execute("""
                SELECT review_date, prediction_score
                FROM review_results
                WHERE stock_code = ?
                ORDER BY review_date DESC LIMIT 5
            """, (code,))
            rows = cursor.fetchall()
            conn.close()

            if len(rows) < 3:
                return None

            # 检查评分是否连续下降
            scores = [r[1] for r in rows if r[1] is not None]
            if len(scores) >= 3:
                declining = all(scores[i] < scores[i-1] for i in range(1, min(3, len(scores))))
                total_drop = scores[0] - scores[min(2, len(scores)-1)] if len(scores) >= 3 else 0

                if declining and total_drop > self.SIGNAL_DECAY_THRESHOLD:
                    return f"信号衰减: 评分连续{min(3, len(scores))}天下降{total_drop:.0f}分"

        except Exception:
            pass

        return None

    def _check_portfolio_rebalance(self, positions: List[Dict],
                                     regime: str) -> List[RebalanceAction]:
        """组合再平衡检查"""
        actions = []

        if not positions:
            return actions

        total_pos = sum(p.get('position_pct', 0) for p in positions)
        if total_pos <= 0:
            return actions

        # 检查单票偏离
        for pos in positions:
            current = pos.get('position_pct', 0)
            weight = current / total_pos if total_pos > 0 else 0

            # 单票权重>30% → 减仓到20%
            if weight > 0.3:
                target = total_pos * 0.2
                actions.append(RebalanceAction(
                    code=pos['code'],
                    name=pos.get('name', ''),
                    action='decrease',
                    target_position_pct=round(target, 1),
                    current_position_pct=current,
                    reasons=[f"权重过于集中({weight*100:.0f}%>30%)，目标降至20%"]
                ))

        return actions

    # ── 4. 完整决策报告 ──

    def generate_decision_report(self, candidate_codes: List[str] = None,
                                  held_positions: List[Dict] = None,
                                  regime: str = None) -> DecisionReport:
        """生成完整交易决策报告

        Args:
            candidate_codes: 候选股票代码列表
            held_positions: 当前持仓列表 [{'code', 'name', 'sector', 'position_pct', ...}]
            regime: 市场状态

        Returns:
            DecisionReport
        """
        if regime is None:
            try:
                regime = self.market_client.detect_market_regime()
            except Exception:
                regime = 'oscillation'

        # 如果没有传入held_positions，从持仓跟踪系统获取
        if held_positions is None:
            held_positions = self._get_tracked_held_positions()

        from app.selector.adaptive_strategy import AdaptiveStrategyManager
        asm = AdaptiveStrategyManager()
        guidance = asm.get_position_guidance(regime)

        report = DecisionReport(
            date=datetime.now().strftime('%Y-%m-%d %H:%M'),
            regime=regime,
            strategy_name=guidance['strategy_name'],
        )

        # ── 买入信号 ──
        if candidate_codes:
            for code in candidate_codes:
                try:
                    signal = self.evaluate_buy_timing(code, regime=regime)
                    if signal and signal.timing_score >= self.TIMING_SCORE_THRESHOLD:
                        report.buy_signals.append(signal)
                except Exception as e:
                    print(f"评估 {code} 买入时机失败: {e}")

        # ── 卖出信号 ──
        if held_positions:
            for pos in held_positions:
                try:
                    signal = self.evaluate_sell_timing(
                        code=pos['code'],
                        name=pos.get('name', ''),
                        hold_days=pos.get('hold_days', 0),
                        profit_pct=pos.get('profit_pct', 0),
                        entry_price=pos.get('entry_price', pos.get('avg_cost', 0)),
                        current_price=pos.get('current_price', 0),
                        highest_price=pos.get('highest_price', pos.get('current_price', 0)),
                        expected_hold_days=pos.get('expected_hold_days', guidance['default_hold_days']),
                        regime=regime,
                    )
                    if signal:
                        report.sell_signals.append(signal)
                except Exception as e:
                    print(f"评估 {pos.get('code', '')} 卖出时机失败: {e}")

        # ── 调仓建议 ──
        if held_positions:
            rebalance = self.evaluate_rebalance(held_positions, regime)
            report.rebalance_actions = rebalance

        # ── 组合级风险警告 ──
        if held_positions:
            total_pos = sum(p.get('position_pct', 0) for p in held_positions)
            if total_pos > guidance['max_total_position']:
                report.risk_warnings.append(
                    f"总仓位{total_pos:.1f}%超过策略上限{guidance['max_total_position']}%"
                )

            sectors = set(p.get('sector', '') for p in held_positions if p.get('sector'))
            if len(sectors) < 3:
                report.risk_warnings.append(f"仅覆盖{len(sectors)}个行业，集中度风险偏高")

        # ── 组合操作建议 ──
        if report.buy_signals:
            top_buys = sorted(report.buy_signals, key=lambda x: -x.timing_score)[:3]
            for s in top_buys:
                report.portfolio_actions.append(
                    f"买入 {s.code} {s.name} [{s.signal_type}] 时机{ s.timing_score:.0f}分 "
                    f"入场{s.entry_price:.2f} 止损{s.stop_loss_price:.2f} 仓位{s.position_pct:.1f}%"
                )

        if report.sell_signals:
            immediate = [s for s in report.sell_signals if s.urgency == 'immediate']
            if immediate:
                for s in immediate:
                    report.portfolio_actions.append(
                        f"⚠️ 立即卖出 {s.code} {s.name} [{s.signal_type}] {s.reasons[0]}"
                    )
            for s in report.sell_signals:
                if s.urgency != 'immediate':
                    report.portfolio_actions.append(
                        f"考虑卖出 {s.code} {s.name} [{s.signal_type}] {s.reasons[0]}"
                    )

        for a in report.rebalance_actions:
            report.portfolio_actions.append(
                f"调仓 {a.code} {a.name} {a.action} {a.current_position_pct:.1f}%→{a.target_position_pct:.1f}% {a.reasons[0]}"
            )

        return report

    def _get_tracked_held_positions(self) -> List[Dict]:
        """从持仓跟踪系统获取当前持有中的持仓"""
        try:
            from app.selector.position_tracker import PositionTracker, STATE_BOUGHT, STATE_HOLDING
            tracker = PositionTracker()
            positions = tracker.get_active_positions()
            held = []
            for p in positions:
                if p.state in (STATE_BOUGHT, STATE_HOLDING) and p.entry_price > 0:
                    held.append({
                        'code': p.code,
                        'name': p.name,
                        'sector': p.sector,
                        'position_pct': p.position_pct,
                        'hold_days': p.hold_days,
                        'profit_pct': p.profit_pct,
                        'entry_price': p.entry_price,
                        'current_price': p.current_price,
                        'highest_price': p.highest_price,
                        'expected_hold_days': p.expected_hold_days,
                    })
            return held
        except Exception:
            return []

    def save_decision_report(self, report: DecisionReport, blogger_name: str = None):
        """保存决策报告到数据库"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                blogger_name TEXT DEFAULT '',
                regime TEXT,
                strategy_name TEXT,
                buy_signals_json TEXT DEFAULT '[]',
                sell_signals_json TEXT DEFAULT '[]',
                rebalance_actions_json TEXT DEFAULT '[]',
                portfolio_actions_json TEXT DEFAULT '[]',
                risk_warnings_json TEXT DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        def _serialize_signals(signals):
            return json.dumps([{
                'code': s.code, 'name': s.name,
                'signal_type': s.signal_type,
                'timing_score': getattr(s, 'timing_score', 0),
                'urgency': getattr(s, 'urgency', ''),
                'entry_price': getattr(s, 'entry_price', 0),
                'stop_loss_price': getattr(s, 'stop_loss_price', 0),
                'take_profit_price': getattr(s, 'take_profit_price', 0),
                'position_pct': getattr(s, 'position_pct', 0),
                'reasons': s.reasons,
                'confidence': getattr(s, 'confidence', ''),
                'hold_days': getattr(s, 'hold_days', 0),
                'profit_pct': getattr(s, 'profit_pct', 0),
            } for s in signals], ensure_ascii=False)

        cursor.execute("""
            INSERT INTO decision_reports
            (report_date, blogger_name, regime, strategy_name,
             buy_signals_json, sell_signals_json, rebalance_actions_json,
             portfolio_actions_json, risk_warnings_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report.date, blogger_name or '', report.regime, report.strategy_name,
            _serialize_signals(report.buy_signals),
            _serialize_signals(report.sell_signals),
            json.dumps([{'code': a.code, 'name': a.name, 'action': a.action,
                         'target': a.target_position_pct, 'current': a.current_position_pct,
                         'reasons': a.reasons} for a in report.rebalance_actions], ensure_ascii=False),
            json.dumps(report.portfolio_actions, ensure_ascii=False),
            json.dumps(report.risk_warnings, ensure_ascii=False),
        ))

        conn.commit()
        conn.close()


def get_latest_decision_report(blogger_name: str = None, db_path: str = None) -> Optional[Dict]:
    """获取最新决策报告"""
    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decision_reports'")
        if not cursor.fetchone():
            conn.close()
            return None

        if blogger_name:
            cursor.execute("""
                SELECT report_date, regime, strategy_name,
                       buy_signals_json, sell_signals_json, rebalance_actions_json,
                       portfolio_actions_json, risk_warnings_json
                FROM decision_reports WHERE blogger_name = ?
                ORDER BY created_at DESC LIMIT 1
            """, (blogger_name,))
        else:
            cursor.execute("""
                SELECT report_date, regime, strategy_name,
                       buy_signals_json, sell_signals_json, rebalance_actions_json,
                       portfolio_actions_json, risk_warnings_json
                FROM decision_reports ORDER BY created_at DESC LIMIT 1
            """)

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        return {
            'report_date': row[0],
            'regime': row[1],
            'strategy_name': row[2],
            'buy_signals': json.loads(row[3]) if row[3] else [],
            'sell_signals': json.loads(row[4]) if row[4] else [],
            'rebalance_actions': json.loads(row[5]) if row[5] else [],
            'portfolio_actions': json.loads(row[6]) if row[6] else [],
            'risk_warnings': json.loads(row[7]) if row[7] else [],
        }
    finally:
        conn.close()
