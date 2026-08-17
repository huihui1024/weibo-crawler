"""推荐质量反馈闭环
让系统从"每次都是新推荐"进化为"持续跟踪、动态调整的投资伙伴"

核心功能:
1. 推荐N日回顾 — 每条推荐在N天后自动回顾实际表现 vs 预期
2. 推荐质量评分 — 方向准确度 + 幅度偏差 + 时机准确度
3. 自动降级警告 — 当近期推荐质量差时触发系统降级
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


# ── 质量评分维度 ──

@dataclass
class RecommendationReview:
    """单条推荐回顾"""
    id: int = 0
    stock_code: str = ''
    stock_name: str = ''
    recommend_date: str = ''
    review_date: str = ''
    recommend_source: str = ''

    # 预期 vs 实际
    expected_return_pct: float = 0
    actual_return_pct: float = 0
    expected_hold_days: int = 10
    actual_hold_days: int = 0

    # 评分
    direction_correct: bool = False   # 方向是否正确(同涨同跌)
    magnitude_error: float = 0        # 幅度偏差 = |actual - expected|
    timing_score: float = 0           # 时机评分(0-100)

    # 综合质量
    quality_score: float = 0          # 0-100, 越高越好
    quality_grade: str = 'C'          # A/B/C/D/F

    # 风险
    max_drawdown_pct: float = 0       # 持仓期间最大回撤
    hit_stop_loss: bool = False       # 是否触发过止损


@dataclass
class QualityReport:
    """推荐质量报告"""
    report_date: str = ''
    period_days: int = 30

    # 统计
    total_reviews: int = 0
    direction_accuracy: float = 0     # 方向准确率
    avg_magnitude_error: float = 0    # 平均幅度偏差
    avg_quality_score: float = 0      # 平均质量分
    grade_distribution: Dict[str, int] = field(default_factory=dict)

    # 胜率
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0

    # 按来源统计
    source_stats: Dict[str, Dict] = field(default_factory=dict)

    # 降级判断
    system_health: str = 'healthy'     # healthy/watch/degraded/suspended
    degradation_reasons: List[str] = field(default_factory=list)

    # 详细回顾
    reviews: List[RecommendationReview] = field(default_factory=list)


class RecommendationFeedback:
    """推荐质量反馈闭环"""

    # 降级阈值
    DEGRADATION_WINDOW = 10           # 看最近10条推荐
    DIRECTION_ACCURACY_LOW = 0.5      # 方向准确率<50% → 降级
    AVG_QUALITY_LOW = 40              # 平均质量<40 → 降级
    AVG_MAGNITUDE_ERROR_HIGH = 5.0    # 平均幅度偏差>5% → 观察
    WIN_RATE_LOW = 0.4                # 胜率<40% → 降级

    # 回顾周期
    DEFAULT_REVIEW_DAYS = [3, 5, 10]  # 在3日、5日、10日时回顾

    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'data/weibo.db'
        self._ensure_table()

    def _ensure_table(self):
        """确保反馈表存在"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT DEFAULT '',
                recommend_date TEXT NOT NULL,
                review_date TEXT NOT NULL,
                recommend_source TEXT DEFAULT '',
                expected_return_pct REAL DEFAULT 0,
                actual_return_pct REAL DEFAULT 0,
                expected_hold_days INTEGER DEFAULT 10,
                actual_hold_days INTEGER DEFAULT 0,
                direction_correct INTEGER DEFAULT 0,
                magnitude_error REAL DEFAULT 0,
                timing_score REAL DEFAULT 0,
                quality_score REAL DEFAULT 0,
                quality_grade TEXT DEFAULT 'C',
                max_drawdown_pct REAL DEFAULT 0,
                hit_stop_loss INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stock_code, recommend_date, review_date)
            )
        """)

        conn.commit()
        conn.close()

    def review_recommendation(self, code: str, recommend_date: str,
                               expected_return: float = 0,
                               expected_hold_days: int = 10,
                               source: str = '') -> Optional[RecommendationReview]:
        """回顾单条推荐

        对比推荐时的预期收益 vs 实际收益，计算质量评分
        """
        today = date.today().isoformat()

        # 获取实际收益
        actual_return, actual_hold_days, max_dd, hit_sl = self._get_actual_performance(
            code, recommend_date
        )

        if actual_return is None:
            return None

        review = RecommendationReview(
            stock_code=code,
            recommend_date=recommend_date,
            review_date=today,
            recommend_source=source,
            expected_return_pct=expected_return,
            actual_return_pct=actual_return,
            expected_hold_days=expected_hold_days,
            actual_hold_days=actual_hold_days,
            max_drawdown_pct=max_dd,
            hit_stop_loss=hit_sl,
        )

        # 计算评分
        self._score_review(review)

        # 保存
        self._save_review(review)

        return review

    def _get_actual_performance(self, code: str, recommend_date: str) -> Tuple:
        """获取推荐后的实际表现

        Returns: (actual_return_pct, actual_hold_days, max_drawdown_pct, hit_stop_loss)
        """
        actual_return = None
        actual_hold_days = 0
        max_dd = 0
        hit_sl = False

        # 1. 从tracked_positions获取（如果已跟踪）
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT actual_return_pct, hold_days, state, sell_reason
            FROM tracked_positions
            WHERE stock_code = ? AND recommend_date = ?
            ORDER BY last_update DESC LIMIT 1
        """, (code, recommend_date))
        row = cursor.fetchone()

        if row and row[2] in ('sold', 'expired'):
            actual_return = row[0] or 0
            actual_hold_days = row[1] or 0
            hit_sl = '止损' in (row[3] or '')
            conn.close()
            return actual_return, actual_hold_days, max_dd, hit_sl

        # 2. 从review_results获取
        cursor.execute("""
            SELECT actual_change, review_date
            FROM review_results
            WHERE stock_code = ? AND prediction_date >= ?
            ORDER BY review_date DESC LIMIT 1
        """, (code, recommend_date))
        row = cursor.fetchone()

        if row and row[0] is not None:
            actual_return = row[0]
            try:
                rd = date.fromisoformat(str(row[1])[:10])
                pdate = date.fromisoformat(recommend_date)
                actual_hold_days = (rd - pdate).days
            except ValueError:
                actual_hold_days = 0

        # 3. 从stock_pool的当前价格估算收益
        if actual_return is None:
            cursor.execute("""
                SELECT price FROM stock_pool
                WHERE stock_code = ? AND status = 'active'
                LIMIT 1
            """, (code,))
            current_row = cursor.fetchone()

            # 从tracked_positions获取推荐时价格
            cursor.execute("""
                SELECT recommend_price FROM tracked_positions
                WHERE stock_code = ? AND recommend_date = ?
                LIMIT 1
            """, (code, recommend_date))
            entry_row = cursor.fetchone()

            if current_row and entry_row and entry_row[0] and entry_row[0] > 0:
                actual_return = (current_row[0] - entry_row[0]) / entry_row[0] * 100
                try:
                    pdate = date.fromisoformat(recommend_date)
                    actual_hold_days = (date.today() - pdate).days
                except ValueError:
                    pass

        conn.close()
        return actual_return or 0, actual_hold_days, max_dd, hit_sl

    def _score_review(self, review: RecommendationReview):
        """计算推荐质量评分"""
        # 方向准确度
        review.direction_correct = (
            (review.expected_return_pct >= 0 and review.actual_return_pct >= 0) or
            (review.expected_return_pct < 0 and review.actual_return_pct < 0)
        )

        # 幅度偏差
        review.magnitude_error = abs(review.actual_return_pct - review.expected_return_pct)

        # 时机评分：持仓天数接近预期天数 → 高分
        if review.expected_hold_days > 0:
            timing_ratio = min(review.actual_hold_days, review.expected_hold_days * 2) / review.expected_hold_days
            review.timing_score = min(100, timing_ratio * 100)
        else:
            review.timing_score = 50

        # 综合质量分 = 方向40% + 幅度30% + 时机30%
        direction_score = 100 if review.direction_correct else 0
        magnitude_score = max(0, 100 - review.magnitude_error * 10)  # 偏差1%扣10分
        review.quality_score = direction_score * 0.4 + magnitude_score * 0.3 + review.timing_score * 0.3

        # 等级
        if review.quality_score >= 80:
            review.quality_grade = 'A'
        elif review.quality_score >= 60:
            review.quality_grade = 'B'
        elif review.quality_score >= 40:
            review.quality_grade = 'C'
        elif review.quality_score >= 20:
            review.quality_grade = 'D'
        else:
            review.quality_grade = 'F'

        # 止损惩罚
        if review.hit_stop_loss:
            review.quality_score = min(review.quality_score, 30)
            if review.quality_score < 20:
                review.quality_grade = 'F'
            elif review.quality_score < 40:
                review.quality_grade = 'D'

    def _save_review(self, review: RecommendationReview):
        """保存回顾记录"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO recommendation_reviews
            (stock_code, stock_name, recommend_date, review_date, recommend_source,
             expected_return_pct, actual_return_pct, expected_hold_days, actual_hold_days,
             direction_correct, magnitude_error, timing_score,
             quality_score, quality_grade, max_drawdown_pct, hit_stop_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            review.stock_code, review.stock_name,
            review.recommend_date, review.review_date, review.recommend_source,
            review.expected_return_pct, review.actual_return_pct,
            review.expected_hold_days, review.actual_hold_days,
            1 if review.direction_correct else 0,
            round(review.magnitude_error, 2),
            round(review.timing_score, 1),
            round(review.quality_score, 1),
            review.quality_grade,
            round(review.max_drawdown_pct, 2),
            1 if review.hit_stop_loss else 0,
        ))

        conn.commit()
        conn.close()

    def run_batch_review(self, days: int = 30) -> QualityReport:
        """批量回顾最近N天的推荐

        Args:
            days: 回顾最近多少天的推荐

        Returns:
            QualityReport
        """
        report = QualityReport(
            report_date=date.today().isoformat(),
            period_days=days,
        )

        # 获取需要回顾的推荐
        recommendations = self._get_recommendations_to_review(days)

        for rec in recommendations:
            review = self.review_recommendation(
                code=rec['code'],
                recommend_date=rec['recommend_date'],
                expected_return=rec.get('expected_return', 0),
                expected_hold_days=rec.get('expected_hold_days', 10),
                source=rec.get('source', ''),
            )
            if review:
                review.stock_name = rec.get('name', '')
                review.id = len(report.reviews) + 1
                report.reviews.append(review)

        # 计算统计
        self._compute_report_stats(report)

        # 降级判断
        self._evaluate_system_health(report)

        # 保存报告
        self._save_quality_report(report)

        return report

    def _get_recommendations_to_review(self, days: int) -> List[Dict]:
        """获取需要回顾的推荐列表"""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()
        recs = []

        # 从tracked_positions获取
        cursor.execute("""
            SELECT stock_code, stock_name, recommend_date, recommend_source,
                   expected_return_pct, expected_hold_days, state
            FROM tracked_positions
            WHERE recommend_date >= ? AND state IN ('sold', 'expired', 'holding')
            ORDER BY recommend_date DESC
        """, (cutoff,))

        for row in cursor.fetchall():
            recs.append({
                'code': row[0],
                'name': row[1],
                'recommend_date': row[2],
                'source': row[3],
                'expected_return': row[4] or 0,
                'expected_hold_days': row[5] or 10,
                'state': row[6],
            })

        # 从stock_pool获取（历史推荐）
        cursor.execute("""
            SELECT stock_code, stock_name, added_date, selection_source
            FROM stock_pool
            WHERE added_date >= ? AND status != 'removed'
            ORDER BY added_date DESC
        """, (cutoff,))

        seen = {(r['code'], r['recommend_date']) for r in recs}
        for row in cursor.fetchall():
            key = (row[0], str(row[2])[:10] if row[2] else '')
            if key not in seen and key[1]:
                recs.append({
                    'code': row[0],
                    'name': row[1],
                    'recommend_date': key[1],
                    'source': row[3] or '',
                    'expected_return': 0,
                    'expected_hold_days': 10,
                })

        conn.close()
        return recs

    def _compute_report_stats(self, report: QualityReport):
        """计算报告统计"""
        if not report.reviews:
            return

        report.total_reviews = len(report.reviews)
        report.direction_accuracy = sum(1 for r in report.reviews if r.direction_correct) / report.total_reviews
        report.avg_magnitude_error = np.mean([r.magnitude_error for r in report.reviews])
        report.avg_quality_score = np.mean([r.quality_score for r in report.reviews])

        # 等级分布
        for r in report.reviews:
            report.grade_distribution[r.quality_grade] = report.grade_distribution.get(r.quality_grade, 0) + 1

        # 胜率
        report.win_count = sum(1 for r in report.reviews if r.actual_return_pct > 0)
        report.loss_count = sum(1 for r in report.reviews if r.actual_return_pct <= 0)
        report.win_rate = report.win_count / report.total_reviews if report.total_reviews > 0 else 0

        # 按来源统计
        for r in report.reviews:
            src = r.recommend_source or 'unknown'
            if src not in report.source_stats:
                report.source_stats[src] = {'count': 0, 'win': 0, 'avg_quality': 0, 'qualities': []}
            report.source_stats[src]['count'] += 1
            if r.actual_return_pct > 0:
                report.source_stats[src]['win'] += 1
            report.source_stats[src]['qualities'].append(r.quality_score)

        for src in report.source_stats:
            qs = report.source_stats[src]['qualities']
            report.source_stats[src]['avg_quality'] = np.mean(qs) if qs else 0
            report.source_stats[src]['win_rate'] = (
                report.source_stats[src]['win'] / report.source_stats[src]['count']
                if report.source_stats[src]['count'] > 0 else 0
            )
            del report.source_stats[src]['qualities']

    def _evaluate_system_health(self, report: QualityReport):
        """评估系统健康度，决定是否降级"""
        if report.total_reviews < 3:
            report.system_health = 'healthy'
            report.degradation_reasons.append(f"样本不足({report.total_reviews}<3)，暂不判断")
            return

        # 看最近N条
        recent = report.reviews[:self.DEGRADATION_WINDOW]
        n = len(recent)

        recent_dir_acc = sum(1 for r in recent if r.direction_correct) / n
        recent_avg_quality = np.mean([r.quality_score for r in recent])
        recent_win_rate = sum(1 for r in recent if r.actual_return_pct > 0) / n
        recent_avg_error = np.mean([r.magnitude_error for r in recent])

        # 降级判断
        if recent_dir_acc < self.DIRECTION_ACCURACY_LOW:
            report.system_health = 'degraded'
            report.degradation_reasons.append(
                f"方向准确率{recent_dir_acc:.0%}<{self.DIRECTION_ACCURACY_LOW:.0%}，推荐方向不可靠"
            )

        if recent_win_rate < self.WIN_RATE_LOW:
            if report.system_health != 'degraded':
                report.system_health = 'degraded'
            report.degradation_reasons.append(
                f"胜率{recent_win_rate:.0%}<{self.WIN_RATE_LOW:.0%}，推荐盈利能力不足"
            )

        if recent_avg_quality < self.AVG_QUALITY_LOW:
            if report.system_health == 'healthy':
                report.system_health = 'watch'
            report.degradation_reasons.append(
                f"平均质量{recent_avg_quality:.0f}<{self.AVG_QUALITY_LOW}，推荐质量偏低"
            )

        if recent_avg_error > self.AVG_MAGNITUDE_ERROR_HIGH:
            if report.system_health == 'healthy':
                report.system_health = 'watch'
            report.degradation_reasons.append(
                f"平均幅度偏差{recent_avg_error:.1f}%>{self.AVG_MAGNITUDE_ERROR_HIGH}%，预测精度不足"
            )

        if not report.degradation_reasons:
            report.system_health = 'healthy'

    def _save_quality_report(self, report: QualityReport):
        """保存质量报告 — 同一日期去重，仅保留最新一条"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                period_days INTEGER DEFAULT 30,
                total_reviews INTEGER DEFAULT 0,
                direction_accuracy REAL DEFAULT 0,
                avg_quality_score REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                system_health TEXT DEFAULT 'healthy',
                degradation_reasons TEXT DEFAULT '[]',
                grade_distribution TEXT DEFAULT '{}',
                source_stats TEXT DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 检查今天是否已有报告
        cursor.execute("""
            SELECT id FROM quality_reports WHERE report_date = ? LIMIT 1
        """, (report.report_date,))
        existing = cursor.fetchone()

        if existing:
            # 更新已有报告
            cursor.execute("""
                UPDATE quality_reports SET
                    period_days=?, total_reviews=?, direction_accuracy=?,
                    avg_quality_score=?, win_rate=?, system_health=?,
                    degradation_reasons=?, grade_distribution=?, source_stats=?,
                    created_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, (
                report.period_days, report.total_reviews,
                round(report.direction_accuracy, 3),
                round(report.avg_quality_score, 1),
                round(report.win_rate, 3),
                report.system_health,
                json.dumps(report.degradation_reasons, ensure_ascii=False),
                json.dumps(report.grade_distribution),
                json.dumps(report.source_stats, ensure_ascii=False),
                existing[0],
            ))
        else:
            cursor.execute("""
                INSERT INTO quality_reports
                (report_date, period_days, total_reviews, direction_accuracy,
                 avg_quality_score, win_rate, system_health, degradation_reasons,
                 grade_distribution, source_stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.report_date, report.period_days, report.total_reviews,
                round(report.direction_accuracy, 3),
                round(report.avg_quality_score, 1),
                round(report.win_rate, 3),
                report.system_health,
                json.dumps(report.degradation_reasons, ensure_ascii=False),
                json.dumps(report.grade_distribution),
                json.dumps(report.source_stats, ensure_ascii=False),
            ))

        conn.commit()
        conn.close()

    def get_latest_quality_report(self) -> Optional[Dict]:
        """获取最新质量报告"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quality_reports'")
            if not cursor.fetchone():
                conn.close()
                return None

            cursor.execute("""
                SELECT report_date, period_days, total_reviews, direction_accuracy,
                       avg_quality_score, win_rate, system_health, degradation_reasons,
                       grade_distribution, source_stats
                FROM quality_reports ORDER BY created_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None

            return {
                'report_date': row[0],
                'period_days': row[1],
                'total_reviews': row[2],
                'direction_accuracy': row[3],
                'avg_quality_score': row[4],
                'win_rate': row[5],
                'system_health': row[6],
                'degradation_reasons': json.loads(row[7]) if row[7] else [],
                'grade_distribution': json.loads(row[8]) if row[8] else {},
                'source_stats': json.loads(row[9]) if row[9] else {},
            }
        finally:
            conn.close()

    def get_quality_dashboard(self) -> Dict:
        """生成质量仪表盘数据"""
        report = self.run_batch_review(days=30)

        dashboard = {
            'report_date': report.report_date,
            'system_health': report.system_health,
            'degradation_reasons': report.degradation_reasons,
            'total_reviews': report.total_reviews,
            'direction_accuracy': round(report.direction_accuracy, 3),
            'avg_quality_score': round(report.avg_quality_score, 1),
            'win_rate': round(report.win_rate, 3),
            'win_count': report.win_count,
            'loss_count': report.loss_count,
            'avg_magnitude_error': round(report.avg_magnitude_error, 2),
            'grade_distribution': report.grade_distribution,
            'source_stats': report.source_stats,
            'reviews': [],
        }

        for r in report.reviews[:20]:
            dashboard['reviews'].append({
                'code': r.stock_code,
                'name': r.stock_name,
                'recommend_date': r.recommend_date,
                'source': r.recommend_source,
                'expected_return': r.expected_return_pct,
                'actual_return': r.actual_return_pct,
                'direction_correct': r.direction_correct,
                'magnitude_error': r.magnitude_error,
                'quality_score': r.quality_score,
                'quality_grade': r.quality_grade,
                'hit_stop_loss': r.hit_stop_loss,
            })

        return dashboard
