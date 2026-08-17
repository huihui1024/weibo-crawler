"""IC衰减监控与因子轮动模块
在现有FactorICTester基础上增加：

1. 滚动IC时间序列 — 按周计算IC，跟踪衰减趋势
2. IC衰减检测 — IC连续2周<0.02降权，IC连续2周<0暂停
3. 因子轮动 — 根据IC趋势动态调整多因子模型权重
4. 因子健康度仪表盘 — 一目了然看哪个因子有效哪个失效
"""
import numpy as np
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_sync_db


@dataclass
class FactorICTimeSeries:
    """因子IC时间序列"""
    factor_name: str
    dates: List[str] = field(default_factory=list)
    ic_values: List[float] = field(default_factory=list)
    current_ic: float = 0.0
    ic_trend: str = "stable"       # "improving"/"stable"/"decaying"/"negative"
    consecutive_low_weeks: int = 0  # 连续低IC周数
    consecutive_neg_weeks: int = 0  # 连续负IC周数
    health_status: str = "healthy"  # "healthy"/"watch"/"degraded"/"suspended"


@dataclass
class FactorHealthReport:
    """因子健康度报告"""
    report_date: str
    regime: str
    factors: List[FactorICTimeSeries] = field(default_factory=list)
    weight_adjustments: Dict[str, float] = field(default_factory=dict)  # factor_name → 新权重系数
    actions_taken: List[str] = field(default_factory=list)
    summary: str = ""


# 因子名到模型因子类名的映射
FACTOR_NAME_MAP = {
    '趋势因子': 'TrendFactor',
    '动量因子': 'MomentumFactor',
    '量能因子': 'VolumeFactor',
    '资金因子': 'MoneyFlowFactor',
    '波动因子': 'VolatilityFactor',
    '情绪因子': 'SentimentFactor',
    '财务质量因子': 'FinancialQualityFactor',
    '成长因子': 'GrowthFactor',
    '北向资金因子': 'NorthBoundFactor',
    '龙虎榜因子': 'DragonTigerFactor',
}


class ICDecayMonitor:
    """IC衰减监控器"""

    # 衰减检测阈值
    IC_LOW_THRESHOLD = 0.02        # IC<0.02视为低效
    IC_NEGATIVE_THRESHOLD = 0.0    # IC<0视为反向
    CONSECUTIVE_LOW_LIMIT = 2      # 连续2周低IC→降权
    CONSECUTIVE_NEG_LIMIT = 2      # 连续2周负IC→暂停

    # 权重调整幅度
    WEIGHT_REDUCE_FACTOR = 0.5     # 降权：原权重×0.5
    WEIGHT_SUSPEND_FACTOR = 0.1    # 暂停：原权重×0.1
    WEIGHT_BOOST_FACTOR = 1.3      # 增权：原权重×1.3

    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'data/weibo.db'

    def run_weekly_ic_analysis(self, weeks: int = 8) -> FactorHealthReport:
        """运行滚动IC分析

        按周计算各因子IC，检测衰减趋势

        Args:
            weeks: 回溯周数

        Returns:
            FactorHealthReport
        """
        report = FactorHealthReport(
            report_date=datetime.now().strftime('%Y-%m-%d'),
            regime='unknown',
        )

        # 加载历史因子数据和收益
        records = self._load_historical_records(weeks * 7)
        if len(records) < 10:
            report.summary = f"历史数据不足({len(records)}条)，无法进行IC衰减分析"
            return report

        # 按周分组
        weekly_groups = self._group_by_week(records)

        # 对每个因子计算滚动IC
        factor_names = self._get_factor_names(records)
        for fname in factor_names:
            ts = self._compute_rolling_ic(fname, weekly_groups)
            report.factors.append(ts)

        # 检测衰减和生成权重调整建议
        report.weight_adjustments, report.actions_taken = self._detect_decay_and_adjust(report.factors)

        # 生成摘要
        report.summary = self._build_summary(report)

        # 保存报告
        self._save_health_report(report)

        return report

    def _load_historical_records(self, lookback_days: int) -> List[Dict]:
        """加载历史因子得分和实际收益数据"""
        cutoff = date.today() - timedelta(days=lookback_days)
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        seen = set()
        records = []

        # 从review_results + predictions获取
        cursor.execute("""
            SELECT r.stock_code, r.actual_change, r.review_date,
                   r.prediction_date, p.technical_basis
            FROM review_results r
            LEFT JOIN predictions p ON r.stock_code = p.stock_code
                AND r.prediction_date = p.prediction_date
            WHERE r.review_date >= ?
        """, (cutoff,))

        for row in cursor.fetchall():
            key = (row[0], str(row[3]))
            if key in seen:
                continue
            tech_basis = json.loads(row[4]) if row[4] else {}
            factor_scores = tech_basis.get('factor_scores', {})
            if factor_scores and row[1] is not None:
                seen.add(key)
                records.append({
                    'stock_code': row[0],
                    'actual_change': row[1],
                    'factor_scores': factor_scores,
                    'date': str(row[2]) if row[2] else str(row[3]),
                })

        conn.close()
        return records

    def _group_by_week(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        """按周分组记录"""
        groups = {}
        for r in records:
            try:
                d = r['date'][:10] if r['date'] else 'unknown'
                # 转为ISO周数
                dt = date.fromisoformat(d)
                week_key = dt.isocalendar()[:2].__str__()  # (year, week)
                if week_key not in groups:
                    groups[week_key] = []
                groups[week_key].append(r)
            except Exception:
                pass
        return groups

    def _get_factor_names(self, records: List[Dict]) -> List[str]:
        """获取所有因子名"""
        names = set()
        for r in records:
            names.update(r.get('factor_scores', {}).keys())
        return sorted(names)

    def _compute_rolling_ic(self, factor_name: str,
                             weekly_groups: Dict[str, List[Dict]]) -> FactorICTimeSeries:
        """计算单因子滚动IC时间序列"""
        ts = FactorICTimeSeries(factor_name=factor_name)

        sorted_weeks = sorted(weekly_groups.keys())

        for week_key in sorted_weeks:
            week_records = weekly_groups[week_key]
            scores = []
            returns = []

            for r in week_records:
                fs = r.get('factor_scores', {})
                if factor_name in fs and r.get('actual_change') is not None:
                    try:
                        score = float(fs[factor_name])
                        ret = float(r['actual_change'])
                        scores.append(score)
                        returns.append(ret)
                    except (ValueError, TypeError):
                        continue

            if len(scores) >= 5:
                try:
                    ic, _ = stats.spearmanr(scores, returns)
                    if np.isnan(ic):
                        ic = 0
                except Exception:
                    ic = 0

                # 用该周第一条记录的日期近似
                first_date = week_records[0].get('date', '')[:10]
                ts.dates.append(first_date)
                ts.ic_values.append(round(ic, 4))

        # 计算趋势和健康度
        if ts.ic_values:
            ts.current_ic = ts.ic_values[-1]

            # 检测连续低IC/负IC
            for ic in reversed(ts.ic_values):
                if ic < self.IC_LOW_THRESHOLD:
                    ts.consecutive_low_weeks += 1
                else:
                    break

            for ic in reversed(ts.ic_values):
                if ic < self.IC_NEGATIVE_THRESHOLD:
                    ts.consecutive_neg_weeks += 1
                else:
                    break

            # IC趋势判断
            if len(ts.ic_values) >= 3:
                recent = ts.ic_values[-3:]
                if all(r < p for r, p in zip(recent[1:], recent[:-1])):
                    ts.ic_trend = "decaying"
                elif all(r > p for r, p in zip(recent[1:], recent[:-1])):
                    ts.ic_trend = "improving"
                else:
                    ts.ic_trend = "stable"
            else:
                ts.ic_trend = "stable"

            # 健康度判定
            if ts.consecutive_neg_weeks >= self.CONSECUTIVE_NEG_LIMIT:
                ts.health_status = "suspended"
            elif ts.consecutive_low_weeks >= self.CONSECUTIVE_LOW_LIMIT:
                ts.health_status = "degraded"
            elif ts.consecutive_low_weeks >= 1:
                ts.health_status = "watch"
            else:
                ts.health_status = "healthy"

        return ts

    def _detect_decay_and_adjust(self, factors: List[FactorICTimeSeries]) -> Tuple[Dict[str, float], List[str]]:
        """检测衰减并生成权重调整建议"""
        adjustments = {}
        actions = []

        for f in factors:
            # 获取当前因子在模型中的权重
            current_weight = self._get_current_factor_weight(f.factor_name)

            if f.health_status == "suspended":
                new_weight = current_weight * self.WEIGHT_SUSPEND_FACTOR
                adjustments[f.factor_name] = new_weight
                actions.append(f"🔴 暂停 {f.factor_name}: 连续{f.consecutive_neg_weeks}周IC<0，权重{current_weight:.3f}→{new_weight:.3f}")

            elif f.health_status == "degraded":
                new_weight = current_weight * self.WEIGHT_REDUCE_FACTOR
                adjustments[f.factor_name] = new_weight
                actions.append(f"🟡 降权 {f.factor_name}: 连续{f.consecutive_low_weeks}周IC<{self.IC_LOW_THRESHOLD}，权重{current_weight:.3f}→{new_weight:.3f}")

            elif f.health_status == "watch":
                # 观察期不调整权重，但记录
                actions.append(f"⚪ 观察 {f.factor_name}: IC={f.current_ic:.4f} 趋势={f.ic_trend}")

            elif f.ic_trend == "improving" and f.current_ic > 0.05:
                # IC持续改善且较强 → 增权
                new_weight = current_weight * self.WEIGHT_BOOST_FACTOR
                adjustments[f.factor_name] = new_weight
                actions.append(f"🟢 增权 {f.factor_name}: IC={f.current_ic:.4f}持续改善，权重{current_weight:.3f}→{new_weight:.3f}")

            else:
                actions.append(f"✅ 正常 {f.factor_name}: IC={f.current_ic:.4f} 趋势={f.ic_trend}")

        return adjustments, actions

    def _get_current_factor_weight(self, factor_name: str) -> float:
        """获取因子当前在多因子模型中的权重"""
        try:
            from app.selector.multi_factor_model import MultiFactorModel
            model = MultiFactorModel(use_feedback=False)
            class_name = FACTOR_NAME_MAP.get(factor_name, '')
            for f in model.factors:
                if f.name == factor_name or type(f).__name__ == class_name:
                    return f.weight
        except Exception:
            pass
        return 0.1  # 默认

    def _build_summary(self, report: FactorHealthReport) -> str:
        """生成摘要"""
        healthy = sum(1 for f in report.factors if f.health_status == 'healthy')
        watch = sum(1 for f in report.factors if f.health_status == 'watch')
        degraded = sum(1 for f in report.factors if f.health_status == 'degraded')
        suspended = sum(1 for f in report.factors if f.health_status == 'suspended')
        total = len(report.factors)

        improving = sum(1 for f in report.factors if f.ic_trend == 'improving')
        decaying = sum(1 for f in report.factors if f.ic_trend == 'decaying')

        parts = [f"因子健康度: {healthy}/{total}正常, {watch}观察, {degraded}降权, {suspended}暂停"]
        if improving:
            parts.append(f"IC改善: {improving}个因子")
        if decaying:
            parts.append(f"IC衰减: {decaying}个因子")

        return " | ".join(parts)

    def _save_health_report(self, report: FactorHealthReport):
        """保存因子健康度报告"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_health_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                summary TEXT,
                factors_json TEXT DEFAULT '[]',
                weight_adjustments_json TEXT DEFAULT '{}',
                actions_json TEXT DEFAULT '[]',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        factors_data = [{
            'name': f.factor_name,
            'current_ic': f.current_ic,
            'ic_trend': f.ic_trend,
            'consecutive_low_weeks': f.consecutive_low_weeks,
            'consecutive_neg_weeks': f.consecutive_neg_weeks,
            'health_status': f.health_status,
            'ic_series': list(zip(f.dates, f.ic_values)),
        } for f in report.factors]

        cursor.execute("""
            INSERT INTO factor_health_reports
            (report_date, summary, factors_json, weight_adjustments_json, actions_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            report.report_date,
            report.summary,
            json.dumps(factors_data, ensure_ascii=False),
            json.dumps(report.weight_adjustments, ensure_ascii=False),
            json.dumps(report.actions_taken, ensure_ascii=False),
        ))

        conn.commit()
        conn.close()

    def apply_weight_adjustments(self, adjustments: Dict[str, float]) -> bool:
        """将IC衰减检测结果应用到多因子模型

        Args:
            adjustments: {factor_name: new_weight}

        Returns:
            是否成功应用
        """
        try:
            from app.selector.multi_factor_model import MultiFactorModel
            model = MultiFactorModel(use_feedback=False)

            for factor_name, new_weight in adjustments.items():
                class_name = FACTOR_NAME_MAP.get(factor_name, '')
                for f in model.factors:
                    if f.name == factor_name or type(f).__name__ == class_name:
                        old_weight = f.weight
                        f.weight = new_weight
                        print(f"因子权重调整: {f.name} {old_weight:.4f} → {new_weight:.4f}")

            # 归一化
            total = sum(f.weight for f in model.factors)
            for f in model.factors:
                f.weight = f.weight / total

            return True
        except Exception as e:
            print(f"应用因子权重调整失败: {e}")
            return False

    def get_factor_health_dashboard(self) -> Dict:
        """生成因子健康度仪表盘数据"""
        report = self.run_weekly_ic_analysis(weeks=8)

        dashboard = {
            'report_date': report.report_date,
            'summary': report.summary,
            'factors': [],
            'actions': report.actions_taken,
            'weight_adjustments': report.weight_adjustments,
        }

        for f in report.factors:
            dashboard['factors'].append({
                'name': f.factor_name,
                'current_ic': f.current_ic,
                'ic_trend': f.ic_trend,
                'health_status': f.health_status,
                'consecutive_low_weeks': f.consecutive_low_weeks,
                'consecutive_neg_weeks': f.consecutive_neg_weeks,
                'ic_series': [{'date': d, 'ic': ic} for d, ic in zip(f.dates, f.ic_values)],
            })

        return dashboard


def get_latest_factor_health(db_path: str = None) -> Optional[Dict]:
    """获取最新因子健康度报告"""
    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='factor_health_reports'")
        if not cursor.fetchone():
            conn.close()
            return None

        cursor.execute("""
            SELECT report_date, summary, factors_json, weight_adjustments_json, actions_json
            FROM factor_health_reports ORDER BY created_at DESC LIMIT 1
        """)

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        return {
            'report_date': row[0],
            'summary': row[1],
            'factors': json.loads(row[2]) if row[2] else [],
            'weight_adjustments': json.loads(row[3]) if row[3] else {},
            'actions': json.loads(row[4]) if row[4] else [],
        }
    finally:
        conn.close()
