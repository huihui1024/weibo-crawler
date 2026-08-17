"""组合构建模块
从候选股票池构建投资组合：仓位管理、风险预算、相关性约束、板块分散
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_sync_db


@dataclass
class PortfolioPosition:
    """组合持仓"""
    code: str
    name: str
    sector: str
    signal_score: float          # 综合评分(0-100)
    position_pct: float          # 建议仓位(%)
    position_grade: str          # "A"/"B"/"C"/"D"
    risk_budget: float           # 风险预算(CVaR贡献)
    volatility: float            # 个股波动率
    correlation_group: int       # 相关性分组ID
    selection_source: str = ""
    selection_reasons: List[str] = field(default_factory=list)


@dataclass
class Portfolio:
    """投资组合"""
    positions: List[PortfolioPosition] = field(default_factory=list)
    total_position_pct: float = 0       # 总仓位
    num_sectors: int = 0                # 覆盖行业数
    max_sector_pct: float = 0           # 最大板块仓位
    portfolio_cvar: float = 0           # 组合CVaR
    concentration_score: float = 0      # 集中度评分(0-100, 越高越分散)
    strategy_name: str = ""             # 自适应策略名称
    stop_loss_pct: float = 0            # 止损百分比
    take_profit_pct: float = 0          # 止盈百分比
    trailing_stop_pct: float = 0        # 移动止损百分比
    risk_warning: str = ""
    created_at: str = ""


class PortfolioConstructor:
    """组合构建器"""

    # 仓位映射：信号置信度 → 建议仓位
    POSITION_MAP = {
        'A': {'min_score': 80, 'position': 5.0},   # A级：强烈关注，5%
        'B': {'min_score': 65, 'position': 3.0},   # B级：适度关注，3%
        'C': {'min_score': 50, 'position': 1.0},   # C级：观察为主，1%
        'D': {'min_score': 0,  'position': 0.0},   # D级：不推荐
    }

    # 约束参数
    MAX_SINGLE_POSITION = 5.0       # 单票上限5%
    MAX_SECTOR_POSITION = 20.0      # 同板块上限20%
    MIN_SECTORS = 3                 # 最少覆盖3个行业
    MAX_CORRELATION = 0.7           # 相关性阈值
    MAX_PORTFOLIO_CVAR = 2.0        # 组合CVaR上限(%)
    TARGET_PORTFOLIO_VOL = 15.0     # 目标组合波动率(%)

    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'data/weibo.db'

    def construct(self, candidates: List[Dict], market_regime: str = 'oscillation') -> Portfolio:
        """从候选股票构建组合

        Args:
            candidates: 候选股票列表(需含composite_score/style_match_score等)
            market_regime: 市场状态

        Returns:
            Portfolio: 构建好的投资组合
        """
        if not candidates:
            return Portfolio(risk_warning="无候选股票，组合为空")

        # 1. 信号质量分级
        graded = self._grade_signals(candidates)

        # 2. 过滤D级（不推荐）
        viable = [s for s in graded if s['grade'] != 'D']
        if not viable:
            return Portfolio(risk_warning="所有候选股票信号质量不足(D级)，不推荐建仓")

        # 3. 相关性去重
        decorrelated = self._decorrelate(viable)

        # 4. 板块分散约束
        diversified = self._enforce_sector_diversification(decorrelated)

        # 5. 仓位分配（风险平价）
        positioned = self._allocate_positions(diversified, market_regime)

        # 6. 组合风险检查
        portfolio = self._build_portfolio(positioned, market_regime)

        return portfolio

    def _grade_signals(self, candidates: List[Dict]) -> List[Dict]:
        """信号质量分级

        A级: 基本面+技术面+资金面三重共振
        B级: 两重共振+一重不拖后腿
        C级: 单一维度突出
        D级: 信号矛盾
        """
        graded = []
        for s in candidates:
            score = s.get('style_match_score', s.get('composite_score', 50))
            fund = s.get('fundamental_score', 0)
            flow = s.get('capital_flow_score', 0)
            tech_score = score  # style_match_score已含技术面

            # 三重共振检测
            fund_pass = fund > 60
            flow_pass = flow > 0
            tech_pass = tech_score > 65

            resonance = sum([fund_pass, flow_pass, tech_pass])

            if resonance >= 3 and score >= 80:
                grade = 'A'
            elif resonance >= 2 and score >= 65:
                grade = 'B'
            elif score >= 50:
                grade = 'C'
            else:
                grade = 'D'

            s['grade'] = grade
            s['signal_score'] = score
            graded.append(s)

        return graded

    def _decorrelate(self, candidates: List[Dict]) -> List[Dict]:
        """相关性去重：同板块内相关性>0.7的只保留最高分

        简化实现：同板块内按score排序，保留前N只（N取决于板块总数）
        精确实现需要计算收益率相关系数矩阵，此处用板块+概念近似
        """
        # 按板块分组
        sector_groups: Dict[str, List[Dict]] = {}
        for s in candidates:
            sector = s.get('sector', '未知')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(s)

        # 每个板块按score排序，保留前2只（近似低相关性）
        result = []
        for sector, stocks in sector_groups.items():
            sorted_stocks = sorted(stocks, key=lambda x: -x.get('signal_score', 0))
            # 同板块最多2只（降低相关性风险）
            result.extend(sorted_stocks[:2])

        # 恢复总体排序
        result.sort(key=lambda x: -x.get('signal_score', 0))
        return result

    def _enforce_sector_diversification(self, candidates: List[Dict]) -> List[Dict]:
        """板块分散约束

        - 至少覆盖MIN_SECTORS个行业
        - 同板块仓位不超过MAX_SECTOR_POSITION
        """
        # 统计已有板块
        sectors = set(s.get('sector', '未知') for s in candidates)

        if len(sectors) < self.MIN_SECTORS:
            # 板块不足，尝试从D级候选中补充不同板块
            # （此处简化处理，实际应回溯到原始候选池）
            pass

        # 按板块仓位约束裁剪
        sector_position: Dict[str, float] = {}
        result = []
        for s in candidates:
            sector = s.get('sector', '未知')
            grade = s.get('grade', 'C')
            pos = self.POSITION_MAP.get(grade, self.POSITION_MAP['C'])['position']
            current_sector_pos = sector_position.get(sector, 0)

            if current_sector_pos + pos <= self.MAX_SECTOR_POSITION:
                sector_position[sector] = current_sector_pos + pos
                result.append(s)
            # else: 超过板块仓位上限，跳过

        return result

    def _allocate_positions(self, candidates: List[Dict], market_regime: str) -> List[Dict]:
        """仓位分配 — 基于风险平价 + 自适应策略

        核心思路：波动率低的股票分配更多仓位，使每只股票的风险贡献相等
        自适应策略：根据市场状态调整总仓位、单票仓位上限
        """
        # 自适应策略仓位指引
        from app.selector.adaptive_strategy import AdaptiveStrategyManager
        asm = AdaptiveStrategyManager()

        try:
            from app.selector.stock_selector import get_dynamic_win_rate
            dyn = get_dynamic_win_rate()
            signal = dyn.get('signal_strength', 'normal')
        except Exception:
            signal = 'normal'

        guidance = asm.get_position_guidance(market_regime, signal)
        max_total = guidance['max_total_position']
        single_max = guidance['single_position']
        max_positions = guidance['max_positions']

        # 计算每只股票实际可达的目标总仓位
        # regime_multiplier = 目标总仓位 / 100
        regime_multiplier = max_total / 100

        # 获取波动率（简化：用ATR或历史波动率近似）
        for s in candidates:
            tech = s.get('technical_indicators', {})
            # 用20日振幅近似波动率
            vol = tech.get('amplitude_20d', 20.0) or 20.0
            s['volatility'] = max(vol, 5.0)  # 最低5%避免除零

        # 风险平价：仓位 ∝ 1/波动率
        inv_vols = [1.0 / s['volatility'] for s in candidates]
        total_inv_vol = sum(inv_vols)

        for i, s in enumerate(candidates):
            grade = s.get('grade', 'C')
            base_position = self.POSITION_MAP.get(grade, self.POSITION_MAP['C'])['position']

            # 风险平价调整：波动率低的给更多仓位
            risk_parity_weight = inv_vols[i] / total_inv_vol if total_inv_vol > 0 else 1.0 / len(candidates)
            risk_parity_position = risk_parity_weight * 100  # 转为百分比

            # 取base_position和risk_parity_position的加权平均
            # 信号质量权重0.6，风险平价权重0.4
            final_position = base_position * 0.6 + risk_parity_position * 0.4

            # 应用市场状态系数
            final_position *= regime_multiplier

            # 钳位：使用自适应策略的单票上限
            final_position = max(0, min(single_max, round(final_position, 1)))

            s['position_pct'] = final_position
            s['risk_budget'] = round(final_position * s['volatility'] / 100, 2)  # 风险预算

        return candidates

    def _build_portfolio(self, candidates: List[Dict], market_regime: str = 'oscillation') -> Portfolio:
        """构建最终组合"""
        # 自适应策略信息
        from app.selector.adaptive_strategy import AdaptiveStrategyManager
        asm = AdaptiveStrategyManager()
        try:
            from app.selector.stock_selector import get_dynamic_win_rate
            dyn = get_dynamic_win_rate()
            signal = dyn.get('signal_strength', 'normal')
        except Exception:
            signal = 'normal'
        guidance = asm.get_position_guidance(market_regime, signal)

        positions = []
        total_pos = 0
        sectors = set()
        sector_pos: Dict[str, float] = {}

        for i, s in enumerate(candidates):
            pos = PortfolioPosition(
                code=s.get('code', ''),
                name=s.get('name', ''),
                sector=s.get('sector', ''),
                signal_score=s.get('signal_score', 0),
                position_pct=s.get('position_pct', 0),
                position_grade=s.get('grade', 'C'),
                risk_budget=s.get('risk_budget', 0),
                volatility=s.get('volatility', 20),
                correlation_group=i,  # 简化：每只独立组
                selection_source=s.get('selection_source', ''),
                selection_reasons=s.get('reasons', s.get('selection_reasons', [])),
            )
            positions.append(pos)
            total_pos += pos.position_pct
            sectors.add(pos.sector)
            sector_pos[pos.sector] = sector_pos.get(pos.sector, 0) + pos.position_pct

        max_sector = max(sector_pos.values()) if sector_pos else 0

        # 集中度评分：越分散越高
        if len(positions) > 0:
            # Herfindahl指数的倒数归一化
            hhi = sum((p.position_pct / total_pos) ** 2 for p in positions if total_pos > 0)
            concentration_score = round((1 / hhi - 1) / (len(positions) - 1) * 100, 1) if hhi > 0 and len(positions) > 1 else 0
        else:
            concentration_score = 0

        # 组合CVaR近似：加权波动率
        portfolio_cvar = sum(p.risk_budget for p in positions)

        # 风险警告
        warnings = []
        if total_pos > 80:
            warnings.append(f"总仓位{total_pos:.1f}%偏高，建议降至80%以下")
        if max_sector > self.MAX_SECTOR_POSITION:
            warnings.append(f"最大板块仓位{max_sector:.1f}%超限(>{self.MAX_SECTOR_POSITION}%)")
        if len(sectors) < self.MIN_SECTORS:
            warnings.append(f"仅覆盖{len(sectors)}个行业，建议至少{self.MIN_SECTORS}个")
        if portfolio_cvar > self.MAX_PORTFOLIO_CVAR:
            warnings.append(f"组合CVaR={portfolio_cvar:.1f}%超限(>{self.MAX_PORTFOLIO_CVAR}%)，建议减仓")

        return Portfolio(
            positions=positions,
            total_position_pct=round(total_pos, 1),
            num_sectors=len(sectors),
            max_sector_pct=round(max_sector, 1),
            portfolio_cvar=round(portfolio_cvar, 2),
            concentration_score=concentration_score,
            strategy_name=guidance['strategy_name'],
            stop_loss_pct=guidance['stop_loss_pct'],
            take_profit_pct=guidance['take_profit_pct'],
            trailing_stop_pct=guidance['trailing_stop_pct'],
            risk_warning="；".join(warnings) if warnings else "",
            created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )

    def save_portfolio_to_db(self, portfolio: Portfolio, blogger_name: str = None):
        """保存组合到数据库"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                blogger_name TEXT DEFAULT '',
                total_position_pct REAL,
                num_sectors INTEGER,
                max_sector_pct REAL,
                portfolio_cvar REAL,
                concentration_score REAL,
                risk_warning TEXT DEFAULT '',
                positions_json TEXT DEFAULT '[]'
            )
        """)

        positions_json = json.dumps([{
            'code': p.code,
            'name': p.name,
            'sector': p.sector,
            'signal_score': p.signal_score,
            'position_pct': p.position_pct,
            'position_grade': p.position_grade,
            'risk_budget': p.risk_budget,
            'volatility': p.volatility,
            'selection_source': p.selection_source,
            'selection_reasons': p.selection_reasons,
        } for p in portfolio.positions], ensure_ascii=False)

        cursor.execute("""
            INSERT INTO portfolios
            (created_at, blogger_name, total_position_pct, num_sectors,
             max_sector_pct, portfolio_cvar, concentration_score, risk_warning, positions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            portfolio.created_at,
            blogger_name or '',
            portfolio.total_position_pct,
            portfolio.num_sectors,
            portfolio.max_sector_pct,
            portfolio.portfolio_cvar,
            portfolio.concentration_score,
            portfolio.risk_warning,
            positions_json,
        ))

        conn.commit()
        conn.close()


def get_latest_portfolio(blogger_name: str = None, db_path: str = None) -> Optional[Dict]:
    """获取最新组合"""
    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolios'")
        if not cursor.fetchone():
            conn.close()
            return None

        if blogger_name:
            cursor.execute("""
                SELECT created_at, total_position_pct, num_sectors, max_sector_pct,
                       portfolio_cvar, concentration_score, risk_warning, positions_json
                FROM portfolios WHERE blogger_name = ?
                ORDER BY created_at DESC LIMIT 1
            """, (blogger_name,))
        else:
            cursor.execute("""
                SELECT created_at, total_position_pct, num_sectors, max_sector_pct,
                       portfolio_cvar, concentration_score, risk_warning, positions_json
                FROM portfolios
                ORDER BY created_at DESC LIMIT 1
            """)

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        return {
            'created_at': row[0],
            'total_position_pct': row[1],
            'num_sectors': row[2],
            'max_sector_pct': row[3],
            'portfolio_cvar': row[4],
            'concentration_score': row[5],
            'risk_warning': row[6],
            'positions': json.loads(row[7]) if row[7] else [],
        }
    finally:
        conn.close()
