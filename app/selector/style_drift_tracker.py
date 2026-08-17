"""
风格漂移与调仓时序追踪模块
核心投研能力：追踪博主/策略的风格变化和调仓时机
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from app.database import get_sync_db


@dataclass
class PositionSnapshot:
    """持仓快照"""
    date: date
    positions: Dict[str, float]  # code -> weight%
    sectors: Dict[str, float]  # sector -> weight%
    total_value: float = 0
    cash_ratio: float = 0


@dataclass
class StyleMetrics:
    """风格指标"""
    date: date
    # 规模风格
    large_cap_ratio: float = 0  # 大盘股权重
    mid_cap_ratio: float = 0    # 中盘股权重
    small_cap_ratio: float = 0  # 小盘股权重

    # 价值/成长风格
    value_ratio: float = 0      # 价值股权重
    growth_ratio: float = 0     # 成长股权重
    blend_ratio: float = 0      # 混合型权重

    # 行业风格
    tech_ratio: float = 0       # 科技股权重
    consume_ratio: float = 0    # 消费股权重
    finance_ratio: float = 0    # 金融股权重
    cycle_ratio: float = 0      # 周期股权重
    defense_ratio: float = 0    # 防御股权重

    # 交易风格
    turnover_rate: float = 0    # 换手率
    hold_period: float = 0      # 平均持仓周期(天)
    concentration: float = 0    # 持仓集中度(Top10权重)


@dataclass
class StyleDriftEvent:
    """风格漂移事件"""
    date: date
    drift_type: str  # sector_shift, cap_change, style_change, concentration_change
    from_value: str
    to_value: str
    magnitude: float  # 漂移幅度
    description: str


@dataclass
class PositionAdjustment:
    """调仓记录"""
    date: date
    stock_code: str
    stock_name: str
    action: str  # buy, sell, add, reduce
    weight_change: float  # 权重变化
    price: float = 0
    reason: str = ""


@dataclass
class RebalanceSignal:
    """调仓信号"""
    date: date
    signal_type: str  # trend_follow, mean_revert, risk_control, opportunity
    suggested_actions: List[PositionAdjustment]
    confidence: float
    reason: str


class StyleDriftTracker:
    """风格漂移追踪器"""

    # 股票规模分类（按市值）
    LARGE_CAP_THRESHOLD = 500  # 500亿以上为大盘
    MID_CAP_THRESHOLD = 100    # 100-500亿为中盘

    # 行业分类映射
    SECTOR_CLASSIFICATION = {
        '科技': ['半导体', '集成电路', '国产芯片', '光模块', 'CPO', '算力', 'AI', '人工智能', '软件', '电子'],
        '消费': ['白酒', '食品饮料', '家电', '医药', '医疗', '零售'],
        '金融': ['证券', '券商', '银行', '保险'],
        '周期': ['煤炭', '有色金属', '钢铁', '化工', '建材', '地产'],
        '防御': ['公用事业', '电力', '高速公路', '港口'],
        '新能源': ['锂电池', '光伏', '储能', '新能源车'],
        '军工': ['军工', '航天', '航空'],
    }

    # 价值/成长判断指标
    VALUE_PE_THRESHOLD = 15    # PE<15视为价值股
    GROWTH_PE_THRESHOLD = 30   # PE>30视为成长股
    GROWTH_ROE_THRESHOLD = 15  # ROE>15%视为成长股

    def __init__(self, db_path: str = 'data/weibo.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 持仓历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blogger_name TEXT NOT NULL,
                snapshot_date DATE NOT NULL,
                positions TEXT,
                sectors TEXT,
                total_value REAL DEFAULT 0,
                cash_ratio REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(blogger_name, snapshot_date)
            )
        """)

        # 风格指标历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS style_metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blogger_name TEXT NOT NULL,
                metric_date DATE NOT NULL,
                large_cap_ratio REAL DEFAULT 0,
                mid_cap_ratio REAL DEFAULT 0,
                small_cap_ratio REAL DEFAULT 0,
                value_ratio REAL DEFAULT 0,
                growth_ratio REAL DEFAULT 0,
                blend_ratio REAL DEFAULT 0,
                tech_ratio REAL DEFAULT 0,
                consume_ratio REAL DEFAULT 0,
                finance_ratio REAL DEFAULT 0,
                cycle_ratio REAL DEFAULT 0,
                defense_ratio REAL DEFAULT 0,
                turnover_rate REAL DEFAULT 0,
                hold_period REAL DEFAULT 0,
                concentration REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(blogger_name, metric_date)
            )
        """)

        # 风格漂移事件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS style_drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blogger_name TEXT NOT NULL,
                event_date DATE NOT NULL,
                drift_type TEXT NOT NULL,
                from_value TEXT,
                to_value TEXT,
                magnitude REAL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 调仓记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blogger_name TEXT NOT NULL,
                adjust_date DATE NOT NULL,
                stock_code TEXT NOT NULL,
                stock_name TEXT,
                action TEXT NOT NULL,
                weight_change REAL,
                price REAL DEFAULT 0,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def save_position_snapshot(self, blogger_name: str, snapshot: PositionSnapshot):
        """保存持仓快照"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO position_history
            (blogger_name, snapshot_date, positions, sectors, total_value, cash_ratio)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            blogger_name,
            snapshot.date,
            json.dumps(snapshot.positions, ensure_ascii=False),
            json.dumps(snapshot.sectors, ensure_ascii=False),
            snapshot.total_value,
            snapshot.cash_ratio
        ))

        conn.commit()
        conn.close()

    def get_position_history(self, blogger_name: str, days: int = 90) -> List[PositionSnapshot]:
        """获取持仓历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT snapshot_date, positions, sectors, total_value, cash_ratio
            FROM position_history
            WHERE blogger_name = ?
            AND snapshot_date >= date('now', ?)
            ORDER BY snapshot_date ASC
        """, (blogger_name, f'-{days} days'))

        snapshots = []
        for row in cursor.fetchall():
            snapshots.append(PositionSnapshot(
                date=date.fromisoformat(row[0]),
                positions=json.loads(row[1]) if row[1] else {},
                sectors=json.loads(row[2]) if row[2] else {},
                total_value=row[3],
                cash_ratio=row[4]
            ))

        conn.close()
        return snapshots

    def calculate_style_metrics(self, snapshot: PositionSnapshot,
                                 stock_info: Dict[str, Dict] = None) -> StyleMetrics:
        """
        计算风格指标

        Args:
            snapshot: 持仓快照
            stock_info: 股票信息字典 {code: {market_cap, pe, roe, sector, ...}}
        """
        metrics = StyleMetrics(date=snapshot.date)

        if not snapshot.positions or not stock_info:
            return metrics

        total_weight = sum(snapshot.positions.values())
        if total_weight == 0:
            return metrics

        # 计算规模风格
        for code, weight in snapshot.positions.items():
            info = stock_info.get(code, {})
            market_cap = info.get('market_cap', 0)  # 市值(亿)

            weight_ratio = weight / total_weight
            if market_cap >= self.LARGE_CAP_THRESHOLD:
                metrics.large_cap_ratio += weight_ratio
            elif market_cap >= self.MID_CAP_THRESHOLD:
                metrics.mid_cap_ratio += weight_ratio
            else:
                metrics.small_cap_ratio += weight_ratio

        # 计算价值/成长风格
        for code, weight in snapshot.positions.items():
            info = stock_info.get(code, {})
            pe = info.get('pe', 0)
            roe = info.get('roe', 0)

            weight_ratio = weight / total_weight
            if pe > 0:
                if pe < self.VALUE_PE_THRESHOLD:
                    metrics.value_ratio += weight_ratio
                elif pe > self.GROWTH_PE_THRESHOLD or roe > self.GROWTH_ROE_THRESHOLD:
                    metrics.growth_ratio += weight_ratio
                else:
                    metrics.blend_ratio += weight_ratio
            else:
                metrics.blend_ratio += weight_ratio

        # 计算行业风格
        for code, weight in snapshot.positions.items():
            info = stock_info.get(code, {})
            sector = info.get('sector', '')

            weight_ratio = weight / total_weight
            classified = False

            for style_sector, keywords in self.SECTOR_CLASSIFICATION.items():
                if any(kw in sector for kw in keywords):
                    if style_sector == '科技':
                        metrics.tech_ratio += weight_ratio
                    elif style_sector == '消费':
                        metrics.consume_ratio += weight_ratio
                    elif style_sector == '金融':
                        metrics.finance_ratio += weight_ratio
                    elif style_sector == '周期':
                        metrics.cycle_ratio += weight_ratio
                    elif style_sector == '防御':
                        metrics.defense_ratio += weight_ratio
                    classified = True
                    break

            if not classified and sector:
                # 未分类的行业，根据关键词判断
                if any(kw in sector for kw in ['新能源', '锂电', '光伏', '储能']):
                    metrics.tech_ratio += weight_ratio  # 新能源归入科技
                elif any(kw in sector for kw in ['军工', '航天']):
                    metrics.tech_ratio += weight_ratio  # 军工归入科技

        # 计算持仓集中度
        sorted_weights = sorted(snapshot.positions.values(), reverse=True)
        top10_weight = sum(sorted_weights[:10])
        metrics.concentration = top10_weight / total_weight if total_weight > 0 else 0

        return metrics

    def save_style_metrics(self, blogger_name: str, metrics: StyleMetrics):
        """保存风格指标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO style_metrics_history
            (blogger_name, metric_date, large_cap_ratio, mid_cap_ratio, small_cap_ratio,
             value_ratio, growth_ratio, blend_ratio, tech_ratio, consume_ratio,
             finance_ratio, cycle_ratio, defense_ratio, turnover_rate, hold_period, concentration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            blogger_name, metrics.date,
            metrics.large_cap_ratio, metrics.mid_cap_ratio, metrics.small_cap_ratio,
            metrics.value_ratio, metrics.growth_ratio, metrics.blend_ratio,
            metrics.tech_ratio, metrics.consume_ratio, metrics.finance_ratio,
            metrics.cycle_ratio, metrics.defense_ratio,
            metrics.turnover_rate, metrics.hold_period, metrics.concentration
        ))

        conn.commit()
        conn.close()

    def get_style_history(self, blogger_name: str, days: int = 90) -> List[StyleMetrics]:
        """获取风格历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT metric_date, large_cap_ratio, mid_cap_ratio, small_cap_ratio,
                   value_ratio, growth_ratio, blend_ratio, tech_ratio, consume_ratio,
                   finance_ratio, cycle_ratio, defense_ratio, turnover_rate, hold_period, concentration
            FROM style_metrics_history
            WHERE blogger_name = ?
            AND metric_date >= date('now', ?)
            ORDER BY metric_date ASC
        """, (blogger_name, f'-{days} days'))

        metrics_list = []
        for row in cursor.fetchall():
            metrics_list.append(StyleMetrics(
                date=date.fromisoformat(row[0]),
                large_cap_ratio=row[1], mid_cap_ratio=row[2], small_cap_ratio=row[3],
                value_ratio=row[4], growth_ratio=row[5], blend_ratio=row[6],
                tech_ratio=row[7], consume_ratio=row[8], finance_ratio=row[9],
                cycle_ratio=row[10], defense_ratio=row[11],
                turnover_rate=row[12], hold_period=row[13], concentration=row[14]
            ))

        conn.close()
        return metrics_list

    def detect_style_drift(self, blogger_name: str,
                           lookback_days: int = 30) -> List[StyleDriftEvent]:
        """
        检测风格漂移

        Args:
            blogger_name: 博主名称
            lookback_days: 回看天数

        Returns:
            风格漂移事件列表
        """
        drift_events = []

        # 获取历史风格指标
        metrics_history = self.get_style_history(blogger_name, days=lookback_days + 30)

        if len(metrics_history) < 2:
            return drift_events

        # 比较最近两期
        current = metrics_history[-1]
        previous = metrics_history[-2]

        # 检测规模风格漂移
        if abs(current.large_cap_ratio - previous.large_cap_ratio) > 0.15:
            drift_events.append(StyleDriftEvent(
                date=current.date,
                drift_type='cap_change',
                from_value=f"大盘{previous.large_cap_ratio:.0%}",
                to_value=f"大盘{current.large_cap_ratio:.0%}",
                magnitude=abs(current.large_cap_ratio - previous.large_cap_ratio),
                description=f"规模风格漂移：大盘股权重从{previous.large_cap_ratio:.0%}变为{current.large_cap_ratio:.0%}"
            ))

        # 检测价值/成长风格漂移
        if abs(current.growth_ratio - previous.growth_ratio) > 0.15:
            drift_events.append(StyleDriftEvent(
                date=current.date,
                drift_type='style_change',
                from_value=f"成长{previous.growth_ratio:.0%}",
                to_value=f"成长{current.growth_ratio:.0%}",
                magnitude=abs(current.growth_ratio - previous.growth_ratio),
                description=f"投资风格漂移：成长股权重从{previous.growth_ratio:.0%}变为{current.growth_ratio:.0%}"
            ))

        # 检测行业风格漂移
        sector_changes = [
            ('科技', current.tech_ratio, previous.tech_ratio),
            ('消费', current.consume_ratio, previous.consume_ratio),
            ('金融', current.finance_ratio, previous.finance_ratio),
            ('周期', current.cycle_ratio, previous.cycle_ratio),
        ]

        for sector_name, curr_ratio, prev_ratio in sector_changes:
            if abs(curr_ratio - prev_ratio) > 0.15:
                drift_events.append(StyleDriftEvent(
                    date=current.date,
                    drift_type='sector_shift',
                    from_value=f"{sector_name}{prev_ratio:.0%}",
                    to_value=f"{sector_name}{curr_ratio:.0%}",
                    magnitude=abs(curr_ratio - prev_ratio),
                    description=f"行业配置漂移：{sector_name}权重从{prev_ratio:.0%}变为{curr_ratio:.0%}"
                ))

        # 检测集中度变化
        if abs(current.concentration - previous.concentration) > 0.15:
            drift_events.append(StyleDriftEvent(
                date=current.date,
                drift_type='concentration_change',
                from_value=f"集中度{previous.concentration:.0%}",
                to_value=f"集中度{current.concentration:.0%}",
                magnitude=abs(current.concentration - previous.concentration),
                description=f"持仓集中度变化：从{previous.concentration:.0%}变为{current.concentration:.0%}"
            ))

        # 保存漂移事件
        for event in drift_events:
            self._save_drift_event(blogger_name, event)

        return drift_events

    def _save_drift_event(self, blogger_name: str, event: StyleDriftEvent):
        """保存风格漂移事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO style_drift_events
            (blogger_name, event_date, drift_type, from_value, to_value, magnitude, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            blogger_name, event.date, event.drift_type,
            event.from_value, event.to_value, event.magnitude, event.description
        ))

        conn.commit()
        conn.close()

    def get_drift_events(self, blogger_name: str, days: int = 90) -> List[StyleDriftEvent]:
        """获取风格漂移历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT event_date, drift_type, from_value, to_value, magnitude, description
            FROM style_drift_events
            WHERE blogger_name = ?
            AND event_date >= date('now', ?)
            ORDER BY event_date DESC
        """, (blogger_name, f'-{days} days'))

        events = []
        for row in cursor.fetchall():
            events.append(StyleDriftEvent(
                date=date.fromisoformat(row[0]),
                drift_type=row[1],
                from_value=row[2],
                to_value=row[3],
                magnitude=row[4],
                description=row[5]
            ))

        conn.close()
        return events

    def track_position_adjustments(self, blogger_name: str,
                                     current_snapshot: PositionSnapshot,
                                     previous_snapshot: PositionSnapshot) -> List[PositionAdjustment]:
        """
        追踪调仓变化

        Args:
            blogger_name: 博主名称
            current_snapshot: 当前持仓快照
            previous_snapshot: 上一期持仓快照

        Returns:
            调仓记录列表
        """
        adjustments = []

        current_positions = current_snapshot.positions
        previous_positions = previous_snapshot.positions

        all_codes = set(current_positions.keys()) | set(previous_positions.keys())

        for code in all_codes:
            current_weight = current_positions.get(code, 0)
            previous_weight = previous_positions.get(code, 0)
            weight_change = current_weight - previous_weight

            if abs(weight_change) < 0.5:  # 忽略微小的权重变化
                continue

            # 判断操作类型
            if previous_weight == 0 and current_weight > 0:
                action = 'buy'
                reason = '新建仓位'
            elif current_weight == 0 and previous_weight > 0:
                action = 'sell'
                reason = '清仓'
            elif weight_change > 0:
                action = 'add'
                reason = '加仓'
            else:
                action = 'reduce'
                reason = '减仓'

            adjustment = PositionAdjustment(
                date=current_snapshot.date,
                stock_code=code,
                stock_name='',  # 需要后续补充
                action=action,
                weight_change=abs(weight_change),
                reason=reason
            )
            adjustments.append(adjustment)

        # 保存调仓记录
        for adj in adjustments:
            self._save_position_adjustment(blogger_name, adj)

        return adjustments

    def _save_position_adjustment(self, blogger_name: str, adjustment: PositionAdjustment):
        """保存调仓记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO position_adjustments
            (blogger_name, adjust_date, stock_code, stock_name, action, weight_change, price, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            blogger_name, adjustment.date, adjustment.stock_code,
            adjustment.stock_name, adjustment.action,
            adjustment.weight_change, adjustment.price, adjustment.reason
        ))

        conn.commit()
        conn.close()

    def get_adjustment_history(self, blogger_name: str, days: int = 30) -> List[PositionAdjustment]:
        """获取调仓历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT adjust_date, stock_code, stock_name, action, weight_change, price, reason
            FROM position_adjustments
            WHERE blogger_name = ?
            AND adjust_date >= date('now', ?)
            ORDER BY adjust_date DESC
        """, (blogger_name, f'-{days} days'))

        adjustments = []
        for row in cursor.fetchall():
            adjustments.append(PositionAdjustment(
                date=date.fromisoformat(row[0]),
                stock_code=row[1],
                stock_name=row[2] or '',
                action=row[3],
                weight_change=row[4],
                price=row[5],
                reason=row[6] or ''
            ))

        conn.close()
        return adjustments

    def calculate_turnover_rate(self, blogger_name: str, days: int = 30) -> float:
        """
        计算换手率

        Args:
            blogger_name: 博主名称
            days: 计算周期

        Returns:
            换手率
        """
        adjustments = self.get_adjustment_history(blogger_name, days=days)

        if not adjustments:
            return 0

        # 计算买入和卖出的总权重变化
        buy_weight = sum(a.weight_change for a in adjustments if a.action in ['buy', 'add'])
        sell_weight = sum(a.weight_change for a in adjustments if a.action in ['sell', 'reduce'])

        # 换手率 = (买入 + 卖出) / 2 / 平均持仓
        turnover = (buy_weight + sell_weight) / 2

        return turnover


class RebalanceSignalGenerator:
    """调仓信号生成器"""

    def __init__(self, tracker: StyleDriftTracker):
        self.tracker = tracker

    def generate_signals(self, blogger_name: str,
                          current_positions: Dict[str, float],
                          market_data: Dict[str, Dict] = None) -> List[RebalanceSignal]:
        """
        生成调仓信号

        Args:
            blogger_name: 博主名称
            current_positions: 当前持仓 {code: weight}
            market_data: 市场数据 {code: {price, change_pct, trend, ...}}

        Returns:
            调仓信号列表
        """
        signals = []

        # 1. 检测风格漂移信号
        drift_events = self.tracker.detect_style_drift(blogger_name)
        if drift_events:
            # 根据漂移生成调仓建议
            for event in drift_events:
                if event.drift_type == 'sector_shift':
                    signal = self._generate_sector_rebalance_signal(
                        event, current_positions, market_data
                    )
                    if signal:
                        signals.append(signal)

        # 2. 检测趋势跟随信号
        trend_signal = self._generate_trend_follow_signal(current_positions, market_data)
        if trend_signal:
            signals.append(trend_signal)

        # 3. 检测均值回归信号
        revert_signal = self._generate_mean_revert_signal(blogger_name, current_positions, market_data)
        if revert_signal:
            signals.append(revert_signal)

        # 4. 检测风险控制信号
        risk_signal = self._generate_risk_control_signal(current_positions, market_data)
        if risk_signal:
            signals.append(risk_signal)

        return signals

    def _generate_sector_rebalance_signal(self, drift_event: StyleDriftEvent,
                                           current_positions: Dict[str, float],
                                           market_data: Dict[str, Dict]) -> Optional[RebalanceSignal]:
        """生成行业再平衡信号"""
        # 如果行业配置发生大幅漂移，建议关注
        if drift_event.magnitude > 0.2:
            return RebalanceSignal(
                date=drift_event.date,
                signal_type='sector_rebalance',
                suggested_actions=[],
                confidence=0.7,
                reason=f"行业配置发生显著漂移：{drift_event.description}，建议评估是否需要调整"
            )
        return None

    def _generate_trend_follow_signal(self, current_positions: Dict[str, float],
                                       market_data: Dict[str, Dict]) -> Optional[RebalanceSignal]:
        """生成趋势跟随信号"""
        if not market_data:
            return None

        actions = []

        for code, weight in current_positions.items():
            data = market_data.get(code, {})
            trend = data.get('trend', 'neutral')
            change_pct = data.get('change_pct', 0)

            # 上升趋势且涨幅不大，建议加仓
            if trend == 'up' and 0 < change_pct < 5:
                actions.append(PositionAdjustment(
                    date=date.today(),
                    stock_code=code,
                    stock_name=data.get('name', ''),
                    action='add',
                    weight_change=min(weight * 0.2, 5),  # 加仓不超过5%
                    reason='趋势向上，建议加仓'
                ))

            # 下跌趋势，建议减仓
            elif trend == 'down' and change_pct < -3:
                actions.append(PositionAdjustment(
                    date=date.today(),
                    stock_code=code,
                    stock_name=data.get('name', ''),
                    action='reduce',
                    weight_change=min(weight * 0.3, 5),  # 减仓不超过5%
                    reason='趋势向下，建议减仓'
                ))

        if actions:
            return RebalanceSignal(
                date=date.today(),
                signal_type='trend_follow',
                suggested_actions=actions,
                confidence=0.6,
                reason='根据趋势信号生成调仓建议'
            )
        return None

    def _generate_mean_revert_signal(self, blogger_name: str,
                                      current_positions: Dict[str, float],
                                      market_data: Dict[str, Dict]) -> Optional[RebalanceSignal]:
        """生成均值回归信号"""
        # 获取历史持仓
        history = self.tracker.get_position_history(blogger_name, days=60)

        if len(history) < 5:
            return None

        # 计算历史平均持仓
        avg_positions = defaultdict(float)
        for snapshot in history:
            for code, weight in snapshot.positions.items():
                avg_positions[code] += weight

        for code in avg_positions:
            avg_positions[code] /= len(history)

        # 检测偏离
        actions = []
        for code, avg_weight in avg_positions.items():
            current_weight = current_positions.get(code, 0)
            deviation = current_weight - avg_weight

            # 如果当前持仓显著低于历史平均，可能是加仓机会
            if deviation < -5 and avg_weight > 3:
                actions.append(PositionAdjustment(
                    date=date.today(),
                    stock_code=code,
                    stock_name='',
                    action='add',
                    weight_change=min(abs(deviation) * 0.5, 3),
                    reason=f'持仓低于历史平均{abs(deviation):.1f}%，可考虑加仓'
                ))

            # 如果当前持仓显著高于历史平均，可能是减仓机会
            elif deviation > 5:
                actions.append(PositionAdjustment(
                    date=date.today(),
                    stock_code=code,
                    stock_name='',
                    action='reduce',
                    weight_change=min(deviation * 0.5, 3),
                    reason=f'持仓高于历史平均{deviation:.1f}%，可考虑减仓'
                ))

        if actions:
            return RebalanceSignal(
                date=date.today(),
                signal_type='mean_revert',
                suggested_actions=actions,
                confidence=0.5,
                reason='根据均值回归策略生成调仓建议'
            )
        return None

    def _generate_risk_control_signal(self, current_positions: Dict[str, float],
                                       market_data: Dict[str, Dict]) -> Optional[RebalanceSignal]:
        """生成风险控制信号"""
        if not market_data:
            return None

        actions = []
        total_weight = sum(current_positions.values())

        # 检测单只股票权重过高
        for code, weight in current_positions.items():
            if weight > 20:  # 单只股票超过20%
                actions.append(PositionAdjustment(
                    date=date.today(),
                    stock_code=code,
                    stock_name=market_data.get(code, {}).get('name', ''),
                    action='reduce',
                    weight_change=weight - 15,  # 减到15%
                    reason=f'单只股票权重过高({weight:.1f}%)，建议分散风险'
                ))

        # 检测连续下跌股票
        for code, weight in current_positions.items():
            data = market_data.get(code, {})
            change_5d = data.get('change_5d', 0)

            if change_5d < -10:  # 5日跌幅超过10%
                actions.append(PositionAdjustment(
                    date=date.today(),
                    stock_code=code,
                    stock_name=data.get('name', ''),
                    action='reduce',
                    weight_change=min(weight * 0.5, 5),
                    reason=f'近期跌幅较大({change_5d:.1f}%)，建议控制风险'
                ))

        if actions:
            return RebalanceSignal(
                date=date.today(),
                signal_type='risk_control',
                suggested_actions=actions,
                confidence=0.8,
                reason='风险控制信号'
            )
        return None


class StyleDriftReporter:
    """风格漂移报告生成器"""

    def __init__(self, tracker: StyleDriftTracker):
        self.tracker = tracker

    def generate_report(self, blogger_name: str, days: int = 30) -> str:
        """
        生成风格漂移报告

        Args:
            blogger_name: 博主名称
            days: 报告周期

        Returns:
            Markdown格式报告
        """
        # 获取数据
        style_history = self.tracker.get_style_history(blogger_name, days=days)
        drift_events = self.tracker.get_drift_events(blogger_name, days=days)
        adjustments = self.tracker.get_adjustment_history(blogger_name, days=days)

        report = f"""
# 风格漂移与调仓追踪报告

> 博主：{blogger_name}
> 报告周期：近{days}天
> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 一、风格概览

"""

        if style_history:
            current_style = style_history[-1]
            report += f"""
### 当前风格配置

| 风格维度 | 当前配置 |
|----------|----------|
| **规模风格** | 大盘 {current_style.large_cap_ratio:.0%} / 中盘 {current_style.mid_cap_ratio:.0%} / 小盘 {current_style.small_cap_ratio:.0%} |
| **投资风格** | 价值 {current_style.value_ratio:.0%} / 成长 {current_style.growth_ratio:.0%} / 混合 {current_style.blend_ratio:.0%} |
| **行业风格** | 科技 {current_style.tech_ratio:.0%} / 消费 {current_style.consume_ratio:.0%} / 金融 {current_style.finance_ratio:.0%} / 周期 {current_style.cycle_ratio:.0%} |
| **持仓集中度** | {current_style.concentration:.0%} |

"""
        else:
            report += "暂无风格数据\n\n"

        # 风格漂移事件
        report += "---\n\n## 二、风格漂移事件\n\n"

        if drift_events:
            report += "| 日期 | 漂移类型 | 变化 | 描述 |\n"
            report += "|------|----------|------|------|\n"
            for event in drift_events:
                report += f"| {event.date} | {event.drift_type} | {event.from_value} → {event.to_value} | {event.description} |\n"
        else:
            report += "近{days}天无显著风格漂移事件\n"

        # 调仓记录
        report += f"\n---\n\n## 三、调仓记录\n\n"

        if adjustments:
            report += "| 日期 | 股票 | 操作 | 权重变化 | 原因 |\n"
            report += "|------|------|------|----------|------|\n"
            for adj in adjustments[:20]:  # 只显示最近20条
                report += f"| {adj.date} | {adj.stock_code} {adj.stock_name} | {adj.action} | {adj.weight_change:.1f}% | {adj.reason} |\n"
        else:
            report += f"近{days}天无调仓记录\n"

        # 换手率统计
        turnover = self.tracker.calculate_turnover_rate(blogger_name, days=days)
        report += f"\n**换手率统计**: 近{days}天换手率约 {turnover:.1f}%\n"

        # 风格趋势图（文字描述）
        report += "\n---\n\n## 四、风格趋势分析\n\n"

        if len(style_history) >= 2:
            first = style_history[0]
            last = style_history[-1]

            report += f"""
### 规模风格变化
- 期初：大盘 {first.large_cap_ratio:.0%} / 中盘 {first.mid_cap_ratio:.0%} / 小盘 {first.small_cap_ratio:.0%}
- 期末：大盘 {last.large_cap_ratio:.0%} / 中盘 {last.mid_cap_ratio:.0%} / 小盘 {last.small_cap_ratio:.0%}
- 变化：大盘{'增加' if last.large_cap_ratio > first.large_cap_ratio else '减少'}{abs(last.large_cap_ratio - first.large_cap_ratio):.0%}

### 行业配置变化
- 科技：{first.tech_ratio:.0%} → {last.tech_ratio:.0%} ({'+' if last.tech_ratio > first.tech_ratio else ''}{(last.tech_ratio - first.tech_ratio):.0%})
- 消费：{first.consume_ratio:.0%} → {last.consume_ratio:.0%} ({'+' if last.consume_ratio > first.consume_ratio else ''}{(last.consume_ratio - first.consume_ratio):.0%})
- 金融：{first.finance_ratio:.0%} → {last.finance_ratio:.0%} ({'+' if last.finance_ratio > first.finance_ratio else ''}{(last.finance_ratio - first.finance_ratio):.0%})
- 周期：{first.cycle_ratio:.0%} → {last.cycle_ratio:.0%} ({'+' if last.cycle_ratio > first.cycle_ratio else ''}{(last.cycle_ratio - first.cycle_ratio):.0%})

### 持仓集中度变化
- 期初：{first.concentration:.0%}
- 期末：{last.concentration:.0%}
- 变化：{'上升' if last.concentration > first.concentration else '下降'}{abs(last.concentration - first.concentration):.0%}
"""
        else:
            report += "数据不足，无法分析风格趋势\n"

        # 投资建议
        report += "\n---\n\n## 五、投资建议\n\n"

        if drift_events:
            report += "### 风险提示\n"
            for event in drift_events[:3]:
                report += f"- {event.description}\n"
            report += "\n"

        if adjustments:
            buy_count = len([a for a in adjustments if a.action in ['buy', 'add']])
            sell_count = len([a for a in adjustments if a.action in ['sell', 'reduce']])
            if buy_count > sell_count * 1.5:
                report += "### 操作倾向\n近期以**加仓**为主，显示对后市较为乐观。\n"
            elif sell_count > buy_count * 1.5:
                report += "### 操作倾向\n近期以**减仓**为主，显示对后市较为谨慎。\n"
            else:
                report += "### 操作倾向\n近期操作相对均衡，维持中性策略。\n"

        report += """
---

*本报告仅供参考，不构成投资建议。*
"""
        return report


def get_style_tracker(db_path: str = 'data/weibo.db') -> StyleDriftTracker:
    """获取风格漂移追踪器实例"""
    return StyleDriftTracker(db_path)


if __name__ == "__main__":
    # 测试风格漂移追踪
    tracker = get_style_tracker()

    # 创建测试持仓快照
    snapshot1 = PositionSnapshot(
        date=date.today() - timedelta(days=7),
        positions={'300308': 15, '600519': 10, '000858': 8, '002475': 5},
        sectors={'科技': 20, '消费': 18},
        total_value=100
    )

    snapshot2 = PositionSnapshot(
        date=date.today(),
        positions={'300308': 12, '600519': 15, '000858': 6, '002475': 8},
        sectors={'科技': 20, '消费': 21},
        total_value=100
    )

    # 保存快照
    tracker.save_position_snapshot('测试博主', snapshot1)
    tracker.save_position_snapshot('测试博主', snapshot2)

    # 追踪调仓
    adjustments = tracker.track_position_adjustments('测试博主', snapshot2, snapshot1)
    print(f"检测到 {len(adjustments)} 条调仓记录:")
    for adj in adjustments:
        print(f"  {adj.stock_code}: {adj.action} {adj.weight_change:.1f}%")

    # 生成报告
    reporter = StyleDriftReporter(tracker)
    report = reporter.generate_report('测试博主', days=30)
    print("\n" + report)
