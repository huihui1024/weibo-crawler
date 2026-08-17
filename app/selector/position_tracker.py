"""持仓跟踪状态机
让系统从"每次都是新推荐"进化为"持续跟踪、动态调整的投资伙伴"

状态流转:
  observing → ready_to_buy → bought → holding → sold
                  ↓              ↓          ↓
              超时降级(3天)   止损/止盈  信号衰减减仓

每只推荐过的股票都有持续更新的状态记录，系统知道自己推荐过什么，
跟踪推荐后的命运，并在适当时机发出操作信号。
"""
import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_sync_db


# ── 状态定义 ──

STATE_OBSERVING = 'observing'        # 观察期：时机评分<70
STATE_READY_TO_BUY = 'ready_to_buy'  # 待买入：时机评分>70
STATE_BOUGHT = 'bought'              # 已买入（刚成交）
STATE_HOLDING = 'holding'            # 持有中
STATE_SOLD = 'sold'                  # 已卖出
STATE_EXPIRED = 'expired'            # 超时降级（待买入3天未执行）

ALL_STATES = [STATE_OBSERVING, STATE_READY_TO_BUY, STATE_BOUGHT,
              STATE_HOLDING, STATE_SOLD, STATE_EXPIRED]

# 超时阈值
READY_TO_BUY_TIMEOUT_DAYS = 3   # 待买入3天未执行则降级
HOLDING_MAX_DAYS_DEFAULT = 20   # 持有超过20天需重新评估


@dataclass
class TrackedPosition:
    """跟踪持仓"""
    id: int = 0
    code: str = ''
    name: str = ''
    sector: str = ''
    state: str = STATE_OBSERVING

    # 推荐信息
    recommend_date: str = ''          # 首次推荐日期
    recommend_score: float = 0        # 推荐时的综合评分
    recommend_reasons: List[str] = field(default_factory=list)
    recommend_source: str = ''        # 基本面驱动/量化策略/市场热点
    recommend_price: float = 0        # 推荐时价格

    # 买入信息
    entry_date: str = ''              # 买入日期
    entry_price: float = 0            # 买入价
    stop_loss_price: float = 0        # 止损价
    take_profit_price: float = 0      # 止盈价
    position_pct: float = 0           # 仓位

    # 持仓跟踪
    current_price: float = 0          # 当前价
    highest_price: float = 0          # 持仓期间最高价
    lowest_price: float = 0           # 持仓期间最低价
    hold_days: int = 0                # 持仓天数
    profit_pct: float = 0             # 当前盈亏%
    timing_score: float = 0           # 最新时机评分

    # 卖出信息
    sell_date: str = ''
    sell_price: float = 0
    sell_reason: str = ''
    actual_return_pct: float = 0      # 实际收益率

    # 预测信息
    expected_return_pct: float = 0    # 预期收益率
    expected_hold_days: int = 0       # 预期持仓天数

    # 元数据
    last_update: str = ''
    state_history: List[Dict] = field(default_factory=list)  # 状态变更历史


class PositionTracker:
    """持仓跟踪器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'data/weibo.db'
        self._ensure_table()

    def _ensure_table(self):
        """确保跟踪表存在"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracked_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                sector TEXT DEFAULT '',
                state TEXT DEFAULT 'observing',

                recommend_date TEXT DEFAULT '',
                recommend_score REAL DEFAULT 0,
                recommend_reasons TEXT DEFAULT '[]',
                recommend_source TEXT DEFAULT '',
                recommend_price REAL DEFAULT 0,

                entry_date TEXT DEFAULT '',
                entry_price REAL DEFAULT 0,
                stop_loss_price REAL DEFAULT 0,
                take_profit_price REAL DEFAULT 0,
                position_pct REAL DEFAULT 0,

                current_price REAL DEFAULT 0,
                highest_price REAL DEFAULT 0,
                lowest_price REAL DEFAULT 0,
                hold_days INTEGER DEFAULT 0,
                profit_pct REAL DEFAULT 0,
                timing_score REAL DEFAULT 0,

                sell_date TEXT DEFAULT '',
                sell_price REAL DEFAULT 0,
                sell_reason TEXT DEFAULT '',
                actual_return_pct REAL DEFAULT 0,

                expected_return_pct REAL DEFAULT 0,
                expected_hold_days INTEGER DEFAULT 0,

                last_update TEXT DEFAULT '',
                state_history TEXT DEFAULT '[]',

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stock_code, state)
            )
        """)

        # 兼容旧表：添加缺失列
        for col, default in [
            ('timing_score', 0), ('expected_return_pct', 0), ('expected_hold_days', 0),
            ('state_history', "'[]'"), ('recommend_source', "''"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE tracked_positions ADD COLUMN {col} TEXT DEFAULT {default}")
            except sqlite3.OperationalError:
                pass

        conn.commit()
        conn.close()

    # ── 核心操作 ──

    def add_recommendation(self, code: str, name: str, sector: str = '',
                            score: float = 0, reasons: List[str] = None,
                            source: str = '', price: float = 0,
                            timing_score: float = 0,
                            expected_return: float = 0, expected_hold_days: int = 10) -> TrackedPosition:
        """添加推荐股票到跟踪系统

        如果已存在同代码的活跃记录(observing/ready_to_buy/holding)，则更新而非重复添加
        """
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        today = date.today().isoformat()

        # 检查是否已有活跃记录
        cursor.execute("""
            SELECT id, state FROM tracked_positions
            WHERE stock_code = ? AND state IN ('observing', 'ready_to_buy', 'bought', 'holding')
            ORDER BY last_update DESC LIMIT 1
        """, (code,))
        existing = cursor.fetchone()

        # 确定初始状态
        initial_state = STATE_READY_TO_BUY if timing_score >= 70 else STATE_OBSERVING

        if existing:
            # 更新已有记录
            pos_id, old_state = existing
            new_state = initial_state
            cursor.execute("""
                UPDATE tracked_positions SET
                    state = ?,
                    recommend_score = ?,
                    recommend_reasons = ?,
                    recommend_source = ?,
                    recommend_price = ?,
                    timing_score = ?,
                    expected_return_pct = ?,
                    expected_hold_days = ?,
                    last_update = ?
                WHERE id = ?
            """, (
                new_state, score,
                json.dumps(reasons or [], ensure_ascii=False),
                source, price, timing_score,
                expected_return, expected_hold_days,
                today, pos_id,
            ))
            conn.commit()
            conn.close()
            return self.get_position(pos_id)

        # 新增记录
        cursor.execute("""
            INSERT INTO tracked_positions
            (stock_code, stock_name, sector, state,
             recommend_date, recommend_score, recommend_reasons, recommend_source, recommend_price,
             timing_score, expected_return_pct, expected_hold_days,
             current_price, highest_price, lowest_price, last_update, state_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            code, name, sector, initial_state,
            today, score,
            json.dumps(reasons or [], ensure_ascii=False),
            source, price,
            timing_score, expected_return, expected_hold_days,
            price, price, price, today,
            json.dumps([{'date': today, 'from': 'new', 'to': initial_state}], ensure_ascii=False),
        ))

        pos_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return self.get_position(pos_id)

    def mark_bought(self, code: str, entry_price: float,
                    stop_loss: float = 0, take_profit: float = 0,
                    position_pct: float = 0) -> bool:
        """标记为已买入"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        today = date.today().isoformat()

        cursor.execute("""
            SELECT id, state, state_history FROM tracked_positions
            WHERE stock_code = ? AND state IN ('ready_to_buy', 'observing')
            ORDER BY last_update DESC LIMIT 1
        """, (code,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        pos_id, old_state, history_json = row
        history = json.loads(history_json) if history_json else []
        history.append({'date': today, 'from': old_state, 'to': STATE_BOUGHT})

        cursor.execute("""
            UPDATE tracked_positions SET
                state = ?,
                entry_date = ?,
                entry_price = ?,
                stop_loss_price = ?,
                take_profit_price = ?,
                position_pct = ?,
                current_price = ?,
                highest_price = ?,
                lowest_price = ?,
                hold_days = 0,
                profit_pct = 0,
                last_update = ?,
                state_history = ?
            WHERE id = ?
        """, (
            STATE_BOUGHT, today, entry_price,
            stop_loss, take_profit, position_pct,
            entry_price, entry_price, entry_price,
            today, json.dumps(history, ensure_ascii=False), pos_id,
        ))

        conn.commit()
        conn.close()
        return True

    def mark_sold(self, code: str, sell_price: float, reason: str = '') -> bool:
        """标记为已卖出"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        today = date.today().isoformat()

        cursor.execute("""
            SELECT id, state, entry_price, state_history FROM tracked_positions
            WHERE stock_code = ? AND state IN ('bought', 'holding')
            ORDER BY last_update DESC LIMIT 1
        """, (code,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        pos_id, old_state, entry_price, history_json = row
        history = json.loads(history_json) if history_json else []
        history.append({'date': today, 'from': old_state, 'to': STATE_SOLD})

        actual_return = (sell_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

        cursor.execute("""
            UPDATE tracked_positions SET
                state = ?,
                sell_date = ?,
                sell_price = ?,
                sell_reason = ?,
                actual_return_pct = ?,
                last_update = ?,
                state_history = ?
            WHERE id = ?
        """, (
            STATE_SOLD, today, sell_price, reason,
            round(actual_return, 2),
            today, json.dumps(history, ensure_ascii=False), pos_id,
        ))

        conn.commit()
        conn.close()
        return True

    # ── 状态更新（每日调用）──

    def update_all_positions(self) -> Dict:
        """更新所有活跃持仓的状态

        Returns:
            Dict with summary of updates
        """
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        today = date.today().isoformat()
        updates = {'price_updated': 0, 'state_changed': 0, 'expired': 0}

        # 获取所有活跃持仓
        cursor.execute("""
            SELECT id, stock_code, stock_name, state, entry_price,
                   stop_loss_price, take_profit_price, highest_price, lowest_price,
                   hold_days, position_pct, expected_hold_days, state_history
            FROM tracked_positions
            WHERE state IN ('observing', 'ready_to_buy', 'bought', 'holding')
        """)
        positions = cursor.fetchall()

        for row in positions:
            (pos_id, code, name, state, entry_price,
             sl_price, tp_price, highest, lowest,
             hold_days, pos_pct, expected_days, history_json) = row

            # 获取最新价格
            current_price = self._fetch_current_price(code)
            if current_price <= 0:
                continue

            updates['price_updated'] += 1

            # 计算盈亏
            profit_pct = 0
            if entry_price > 0 and state in ('bought', 'holding'):
                profit_pct = (current_price - entry_price) / entry_price * 100

            # 更新最高/最低价
            new_highest = max(highest or current_price, current_price)
            new_lowest = min(lowest or current_price, current_price) if lowest else current_price
            new_hold_days = hold_days + 1 if state in ('bought', 'holding') else hold_days

            # 状态转换检查
            new_state = state
            sell_reason = ''

            if state == STATE_READY_TO_BUY:
                # 检查是否超时
                cursor.execute("SELECT recommend_date FROM tracked_positions WHERE id = ?", (pos_id,))
                rd_row = cursor.fetchone()
                if rd_row and rd_row[0]:
                    try:
                        rd = date.fromisoformat(rd_row[0])
                        if (date.today() - rd).days > READY_TO_BUY_TIMEOUT_DAYS:
                            new_state = STATE_EXPIRED
                            sell_reason = f'待买入超{READY_TO_BUY_TIMEOUT_DAYS}天未执行，降级为过期'
                            updates['expired'] += 1
                    except ValueError:
                        pass

            elif state == STATE_BOUGHT:
                # 买入后次日转为持有
                new_state = STATE_HOLDING

            elif state == STATE_HOLDING:
                # 检查止损止盈
                if sl_price > 0 and current_price <= sl_price:
                    new_state = STATE_SOLD
                    sell_reason = f'触发止损: 现价{current_price:.2f}<=止损价{sl_price:.2f}'
                elif tp_price > 0 and current_price >= tp_price:
                    new_state = STATE_SOLD
                    sell_reason = f'触发止盈: 现价{current_price:.2f}>=止盈价{tp_price:.2f}'

                # 移动止损检查
                if new_state == STATE_HOLDING and entry_price > 0 and new_highest > entry_price:
                    from app.selector.adaptive_strategy import AdaptiveStrategyManager
                    asm = AdaptiveStrategyManager()
                    trailing_stop = asm.calculate_trailing_stop(entry_price, new_highest, 'oscillation')
                    if current_price <= trailing_stop:
                        new_state = STATE_SOLD
                        sell_reason = f'移动止损: 现价{current_price:.2f}<=移动止损价{trailing_stop:.2f}'

            # 更新状态历史
            if new_state != state:
                history = json.loads(history_json) if history_json else []
                history.append({'date': today, 'from': state, 'to': new_state,
                               'reason': sell_reason})
                updates['state_changed'] += 1

            # 执行更新
            if new_state == STATE_SOLD:
                actual_return = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                cursor.execute("""
                    UPDATE tracked_positions SET
                        state = ?, current_price = ?, highest_price = ?, lowest_price = ?,
                        hold_days = ?, profit_pct = ?,
                        sell_date = ?, sell_price = ?, sell_reason = ?,
                        actual_return_pct = ?, last_update = ?, state_history = ?
                    WHERE id = ?
                """, (
                    new_state, current_price, new_highest, new_lowest,
                    new_hold_days, round(profit_pct, 2),
                    today, current_price, sell_reason,
                    round(actual_return, 2), today,
                    json.dumps(history, ensure_ascii=False), pos_id,
                ))
            else:
                cursor.execute("""
                    UPDATE tracked_positions SET
                        state = ?, current_price = ?, highest_price = ?, lowest_price = ?,
                        hold_days = ?, profit_pct = ?,
                        last_update = ?, state_history = ?
                    WHERE id = ?
                """, (
                    new_state, current_price, new_highest, new_lowest,
                    new_hold_days, round(profit_pct, 2),
                    today, json.dumps(history, ensure_ascii=False) if new_state != state else history_json,
                    pos_id,
                ))

        conn.commit()
        conn.close()
        return updates

    def _fetch_current_price(self, code: str) -> float:
        """获取当前价格"""
        try:
            from app.market import get_market_client
            client = get_market_client()
            quote = client.get_stock_price(code)
            return quote.get('price', 0) or 0
        except Exception:
            # 从stock_pool获取
            try:
                conn = get_sync_db(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT price FROM stock_pool WHERE stock_code = ? AND status = 'active'", (code,))
                row = cursor.fetchone()
                conn.close()
                return row[0] if row and row[0] else 0
            except Exception:
                return 0

    # ── 查询 ──

    def get_position(self, pos_id: int) -> Optional[TrackedPosition]:
        """获取单个跟踪持仓"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracked_positions WHERE id = ?", (pos_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_position(row)

    def get_active_positions(self) -> List[TrackedPosition]:
        """获取所有活跃持仓(observing/ready_to_buy/bought/holding)"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM tracked_positions
            WHERE state IN ('observing', 'ready_to_buy', 'bought', 'holding')
            ORDER BY
                CASE state
                    WHEN 'holding' THEN 0
                    WHEN 'bought' THEN 1
                    WHEN 'ready_to_buy' THEN 2
                    WHEN 'observing' THEN 3
                END,
                recommend_score DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_position(r) for r in rows]

    def get_sold_positions(self, days: int = 30) -> List[TrackedPosition]:
        """获取最近N天已卖出持仓"""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM tracked_positions
            WHERE state IN ('sold', 'expired') AND sell_date >= ?
            ORDER BY sell_date DESC
        """, (cutoff,))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_position(r) for r in rows]

    def get_position_summary(self) -> Dict:
        """获取持仓概览"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        summary = {}
        for state in ALL_STATES + ['expired']:
            cursor.execute("SELECT COUNT(*) FROM tracked_positions WHERE state = ?", (state,))
            summary[state] = cursor.fetchone()[0]

        # 持仓盈亏
        cursor.execute("""
            SELECT SUM(profit_pct), AVG(profit_pct), COUNT(*)
            FROM tracked_positions WHERE state = 'holding' AND entry_price > 0
        """)
        row = cursor.fetchone()
        summary['holding_total_profit_pct'] = row[0] or 0
        summary['holding_avg_profit_pct'] = row[1] or 0
        summary['holding_count_with_entry'] = row[2] or 0

        # 已卖出统计
        cursor.execute("""
            SELECT AVG(actual_return_pct), COUNT(*),
                   SUM(CASE WHEN actual_return_pct > 0 THEN 1 ELSE 0 END)
            FROM tracked_positions WHERE state = 'sold' AND actual_return_pct != 0
        """)
        row = cursor.fetchone()
        summary['sold_avg_return'] = row[0] or 0
        summary['sold_count'] = row[1] or 0
        summary['sold_win_count'] = row[2] or 0
        summary['sold_win_rate'] = round(row[2] / row[1] * 100, 1) if row[1] and row[1] > 0 else 0

        conn.close()
        return summary

    def sync_from_stock_pool(self):
        """从stock_pool同步推荐到跟踪系统

        将stock_pool中的活跃股票添加到跟踪系统（如果尚未跟踪）
        """
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT stock_code, stock_name, sector, price, style_match_score,
                   selection_source, selection_reasons
            FROM stock_pool WHERE status = 'active'
        """)
        pool_stocks = cursor.fetchall()
        conn.close()

        added = 0
        updated = 0
        for row in pool_stocks:
            code, name, sector, price, score, source, reasons_json = row
            reasons = json.loads(reasons_json) if reasons_json else []

            # 检查是否已在跟踪
            existing = self._find_active(code)
            if existing:
                # 更新时机评分
                updated += 1
                continue

            # 添加到跟踪
            self.add_recommendation(
                code=code, name=name, sector=sector,
                score=score or 0, reasons=reasons,
                source=source or '', price=price or 0,
                timing_score=score or 0,  # 用style_match_score近似
            )
            added += 1

        return {'added': added, 'updated': updated}

    def _find_active(self, code: str) -> Optional[TrackedPosition]:
        """查找活跃持仓"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM tracked_positions
            WHERE stock_code = ? AND state IN ('observing', 'ready_to_buy', 'bought', 'holding')
            LIMIT 1
        """, (code,))
        row = cursor.fetchone()
        conn.close()
        return self._row_to_position(row) if row else None

    def _row_to_position(self, row) -> TrackedPosition:
        """数据库行转TrackedPosition"""
        # 获取列名
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(tracked_positions)")
        col_names = [c[1] for c in cursor.fetchall()]
        conn.close()

        d = dict(zip(col_names, row))

        return TrackedPosition(
            id=d.get('id', 0),
            code=d.get('stock_code', ''),
            name=d.get('stock_name', ''),
            sector=d.get('sector', ''),
            state=d.get('state', STATE_OBSERVING),
            recommend_date=d.get('recommend_date', ''),
            recommend_score=d.get('recommend_score', 0),
            recommend_reasons=json.loads(d.get('recommend_reasons', '[]')) if d.get('recommend_reasons') else [],
            recommend_source=d.get('recommend_source', ''),
            recommend_price=d.get('recommend_price', 0),
            entry_date=d.get('entry_date', ''),
            entry_price=d.get('entry_price', 0),
            stop_loss_price=d.get('stop_loss_price', 0),
            take_profit_price=d.get('take_profit_price', 0),
            position_pct=d.get('position_pct', 0),
            current_price=d.get('current_price', 0),
            highest_price=d.get('highest_price', 0),
            lowest_price=d.get('lowest_price', 0),
            hold_days=d.get('hold_days', 0),
            profit_pct=d.get('profit_pct', 0),
            timing_score=d.get('timing_score', 0),
            sell_date=d.get('sell_date', ''),
            sell_price=d.get('sell_price', 0),
            sell_reason=d.get('sell_reason', ''),
            actual_return_pct=d.get('actual_return_pct', 0),
            expected_return_pct=d.get('expected_return_pct', 0),
            expected_hold_days=d.get('expected_hold_days', 0),
            last_update=d.get('last_update', ''),
            state_history=json.loads(d.get('state_history', '[]')) if d.get('state_history') else [],
        )
