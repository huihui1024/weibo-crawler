"""
复盘→预测反馈闭环模块
读取历史复盘数据，分析各因子偏差方向和幅度，自动调整预测器参数
比如：过去5天趋势因子预测偏差偏正，自动降低sentiment_factor
"""
import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_sync_db


@dataclass
class FactorBias:
    """因子偏差统计"""
    factor_name: str
    avg_bias: float          # 平均偏差（正=预测偏高，负=预测偏低）
    direction_accuracy: float # 方向正确率 0-1
    sample_count: int        # 样本数
    adjustment: float = 0.0  # 建议调整系数（1.0=不变）


@dataclass
class FeedbackResult:
    """反馈调整结果"""
    review_days: int
    total_samples: int
    factor_biases: List[FactorBias] = field(default_factory=list)
    sentiment_factor_adj: float = 1.0   # sentiment_factor调整系数
    trend_weight_adj: float = 1.0       # 趋势因子权重调整系数
    momentum_weight_adj: float = 1.0   # 动量因子权重调整系数
    overall_direction_accuracy: float = 0.0
    summary: str = ""


class FeedbackAdjuster:
    """复盘数据反馈调整器"""

    # 偏差阈值：超过此值触发调整
    BIAS_THRESHOLD = 0.5
    # 最大调整幅度：避免单次调整过于激进
    MAX_ADJUSTMENT = 0.3
    # 调整灵敏度：偏差每增加1%，调整系数变化量
    SENSITIVITY = 0.08

    def __init__(self, db_path: str = None):
        self.db_path = db_path or 'data/weibo.db'

    def analyze_feedback(self, days: int = 5) -> FeedbackResult:
        """分析最近N天复盘数据，生成反馈调整建议

        Args:
            days: 回看天数

        Returns:
            FeedbackResult: 反馈调整结果
        """
        cutoff = date.today() - timedelta(days=days)
        records = self._load_review_data(cutoff)

        if not records:
            return FeedbackResult(
                review_days=days,
                total_samples=0,
                summary=f"最近{days}天无复盘数据，不调整"
            )

        # 计算整体方向正确率
        direction_correct = sum(1 for r in records if r['trend_match'])
        total = len(records)
        overall_accuracy = direction_correct / total if total > 0 else 0

        # 分析各因子偏差
        factor_biases = self._compute_factor_biases(records)

        # 根据偏差计算调整系数
        sentiment_adj = self._compute_sentiment_adjustment(factor_biases)
        trend_adj = self._compute_trend_weight_adjustment(factor_biases, overall_accuracy)
        momentum_adj = self._compute_momentum_weight_adjustment(factor_biases)

        summary = self._build_summary(factor_biases, sentiment_adj,
                                       trend_adj, momentum_adj,
                                       overall_accuracy, total)

        return FeedbackResult(
            review_days=days,
            total_samples=total,
            factor_biases=factor_biases,
            sentiment_factor_adj=sentiment_adj,
            trend_weight_adj=trend_adj,
            momentum_weight_adj=momentum_adj,
            overall_direction_accuracy=round(overall_accuracy, 3),
            summary=summary
        )

    def _load_review_data(self, cutoff_date: date) -> List[Dict]:
        """从数据库加载复盘数据"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT stock_code, expected_change, actual_change, change_diff,
                   expected_trend, actual_trend, trend_match,
                   expected_amplitude, actual_amplitude, amplitude_diff,
                   prediction_score, review_date
            FROM review_results
            WHERE review_date >= ?
            ORDER BY review_date DESC
        """, (cutoff_date,))

        records = []
        for row in cursor.fetchall():
            records.append({
                'stock_code': row[0],
                'expected_change': row[1] or 0,
                'actual_change': row[2] or 0,
                'change_diff': row[3] or 0,
                'expected_trend': row[4] or '',
                'actual_trend': row[5] or '',
                'trend_match': bool(row[6]),
                'expected_amplitude': row[7] or 0,
                'actual_amplitude': row[8] or 0,
                'amplitude_diff': row[9] or 0,
                'prediction_score': row[10] or 50,
                'review_date': row[11],
            })

        conn.close()
        return records

    def _compute_factor_biases(self, records: List[Dict]) -> List[FactorBias]:
        """计算各因子的偏差统计"""
        biases = []

        # 趋势因子偏差：涨跌幅偏差（expected - actual 的方向性）
        trend_bias = self._calc_bias(
            records, 'trend',
            lambda r: r['expected_change'] - r['actual_change'],
            lambda r: r['trend_match']
        )
        biases.append(trend_bias)

        # 动量因子偏差：振幅偏差
        momentum_bias = self._calc_bias(
            records, 'momentum',
            lambda r: r['expected_amplitude'] - r['actual_amplitude'],
            lambda r: abs(r['amplitude_diff']) < 2.0
        )
        biases.append(momentum_bias)

        # 资金/情绪因子偏差：看涨跌幅偏差是否系统性偏正或偏负
        # 如果expected_change系统性高于actual_change，说明情绪因子偏乐观
        sentiment_bias = self._calc_bias(
            records, 'sentiment',
            lambda r: r['change_diff'],
            lambda r: r['trend_match']
        )
        biases.append(sentiment_bias)

        return biases

    def _calc_bias(self, records: List[Dict], factor_name: str,
                   diff_fn, correct_fn) -> FactorBias:
        """通用偏差计算"""
        diffs = [diff_fn(r) for r in records]
        corrects = [1 if correct_fn(r) else 0 for r in records]

        avg_bias = sum(diffs) / len(diffs) if diffs else 0
        accuracy = sum(corrects) / len(corrects) if corrects else 0

        # 根据偏差方向和幅度计算调整系数
        # 偏差为正→预测偏高→应降低→系数<1
        if abs(avg_bias) > self.BIAS_THRESHOLD:
            adj = -avg_bias * self.SENSITIVITY
            adj = max(-self.MAX_ADJUSTMENT, min(self.MAX_ADJUSTMENT, adj))
            adjustment = 1.0 + adj
        else:
            adjustment = 1.0

        return FactorBias(
            factor_name=factor_name,
            avg_bias=round(avg_bias, 3),
            direction_accuracy=round(accuracy, 3),
            sample_count=len(records),
            adjustment=round(adjustment, 3)
        )

    def _compute_sentiment_adjustment(self, biases: List[FactorBias]) -> float:
        """计算sentiment_factor调整系数

        如果sentiment偏差偏正（预测偏高），降低sentiment_factor
        """
        sentiment_bias = next((b for b in biases if b.factor_name == 'sentiment'), None)
        if sentiment_bias is None:
            return 1.0

        return sentiment_bias.adjustment

    def _compute_trend_weight_adjustment(self, biases: List[FactorBias],
                                          accuracy: float) -> float:
        """计算趋势因子权重调整

        方向正确率高→趋势因子可靠→增加权重
        方向正确率低→趋势因子不可靠→降低权重
        """
        trend_bias = next((b for b in biases if b.factor_name == 'trend'), None)
        if trend_bias is None:
            return 1.0

        if accuracy >= 0.6:
            return min(1.0 + (accuracy - 0.5) * 0.2, 1.0 + self.MAX_ADJUSTMENT)
        elif accuracy < 0.4:
            return max(1.0 - (0.5 - accuracy) * 0.3, 1.0 - self.MAX_ADJUSTMENT)
        return 1.0

    def _compute_momentum_weight_adjustment(self, biases: List[FactorBias]) -> float:
        """计算动量因子权重调整"""
        momentum_bias = next((b for b in biases if b.factor_name == 'momentum'), None)
        if momentum_bias is None:
            return 1.0

        return momentum_bias.adjustment

    def _build_summary(self, biases: List[FactorBias],
                       sentiment_adj: float, trend_adj: float,
                       momentum_adj: float, accuracy: float,
                       sample_count: int) -> str:
        """生成调整摘要"""
        parts = [f"近{sample_count}条复盘数据，方向正确率{accuracy:.0%}"]

        for b in biases:
            if abs(b.avg_bias) > self.BIAS_THRESHOLD:
                direction = "偏高" if b.avg_bias > 0 else "偏低"
                parts.append(
                    f"{b.factor_name}预测{direction}(均值{b.avg_bias:+.2f})，"
                    f"调整系数{b.adjustment:.3f}"
                )

        parts.append(
            f"sentiment_factor调整→{sentiment_adj:.3f}，"
            f"趋势权重调整→{trend_adj:.3f}，"
            f"动量权重调整→{momentum_adj:.3f}"
        )

        return "；".join(parts)

    def save_adjustment(self, result: FeedbackResult, db_path: str = None):
        """保存调整记录到数据库，便于追踪和审计"""
        conn = get_sync_db(db_path or self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                adjust_date DATE NOT NULL,
                review_days INTEGER,
                total_samples INTEGER,

                sentiment_factor_adj REAL,
                trend_weight_adj REAL,
                momentum_weight_adj REAL,
                overall_direction_accuracy REAL,

                factor_biases TEXT,
                summary TEXT,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        import json
        cursor.execute("""
            INSERT INTO feedback_adjustments
            (adjust_date, review_days, total_samples,
             sentiment_factor_adj, trend_weight_adj, momentum_weight_adj,
             overall_direction_accuracy, factor_biases, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date.today(),
            result.review_days,
            result.total_samples,
            result.sentiment_factor_adj,
            result.trend_weight_adj,
            result.momentum_weight_adj,
            result.overall_direction_accuracy,
            json.dumps([{
                'factor': b.factor_name,
                'avg_bias': b.avg_bias,
                'accuracy': b.direction_accuracy,
                'adjustment': b.adjustment,
                'samples': b.sample_count
            } for b in result.factor_biases], ensure_ascii=False),
            result.summary
        ))

        conn.commit()
        conn.close()

    def get_latest_adjustment(self, db_path: str = None) -> Optional[FeedbackResult]:
        """获取最近一次调整记录"""
        conn = get_sync_db(db_path or self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT review_days, total_samples,
                       sentiment_factor_adj, trend_weight_adj, momentum_weight_adj,
                       overall_direction_accuracy, factor_biases, summary
                FROM feedback_adjustments
                ORDER BY created_at DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
        except sqlite3.OperationalError:
            conn.close()
            return None

        conn.close()

        if not row:
            return None

        import json
        factor_biases_raw = json.loads(row[6]) if row[6] else []
        factor_biases = [
            FactorBias(
                factor_name=f['factor'],
                avg_bias=f['avg_bias'],
                direction_accuracy=f['accuracy'],
                sample_count=f['samples'],
                adjustment=f['adjustment']
            )
            for f in factor_biases_raw
        ]

        return FeedbackResult(
            review_days=row[0],
            total_samples=row[1],
            factor_biases=factor_biases,
            sentiment_factor_adj=row[2],
            trend_weight_adj=row[3],
            momentum_weight_adj=row[4],
            overall_direction_accuracy=row[5],
    summary=row[7] or ""
        )


def run_feedback_adjustment(days: int = 5) -> FeedbackResult:
    """运行反馈调整并保存"""
    adjuster = FeedbackAdjuster()
    result = adjuster.analyze_feedback(days=days)

    print("=" * 70)
    print("复盘→预测 反馈调整")
    print("=" * 70)
    print(result.summary)
    print()

    if result.total_samples > 0:
        adjuster.save_adjustment(result)
        print("调整记录已保存到数据库")

    return result