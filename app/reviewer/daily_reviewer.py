"""
每日复盘模块
每日尾盘自动复盘，分析股票真实走势与预期的差异，提出改进建议
改进版本：增加市场情绪分析、波动性指标对比
复盘逻辑：复盘前一日预测，验证当日实际走势
"""
import json
import sys
import os
import requests
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database import get_sync_db
from app.market import get_market_client


# 改进建议阈值配置
IMPROVEMENT_THRESHOLDS = {
    'change_diff_warning': 2.0,      # 涨跌幅偏差预警阈值
    'change_diff_critical': 3.0,     # 涨跌幅偏差严重阈值
    'amplitude_diff_warning': 2.0,   # 振幅偏差预警阈值
    'amplitude_diff_critical': 3.0,  # 振幅偏差严重阈值
    'accuracy_warning': 60.0,         # 准确率预警阈值
}


@dataclass
class ReviewResult:
    """复盘结果"""
    stock_code: str
    stock_name: str
    review_date: date
    prediction_date: date  # 原始预测目标日期，用于 UPDATE 匹配

    # 预测数据
    expected_change: float
    expected_amplitude: float
    expected_trend: str
    expected_hold_days: int

    # 实际数据
    actual_open: float
    actual_high: float
    actual_low: float
    actual_close: float
    actual_change: float  # 实际涨跌幅
    actual_amplitude: float  # 实际振幅
    actual_trend: str  # 实际走势

    # 差异分析
    change_diff: float  # 涨跌幅差异
    amplitude_diff: float  # 振幅差异
    trend_match: bool  # 趋势是否匹配
    diff_reason: str  # 差异原因分析

    # 改进建议
    improvement_suggestion: str

    # 评分
    prediction_score: int  # 预测准确度评分 (0-100)

    # 博主
    blogger_name: str = None


@dataclass
class DailyReviewSummary:
    """每日复盘总结"""
    review_date: date
    review_time: datetime

    total_predictions: int
    correct_predictions: int  # 趋势预测正确的数量
    accuracy_rate: float  # 准确率

    avg_change_diff: float  # 平均涨跌幅差异
    avg_amplitude_diff: float  # 平均振幅差异

    avg_prediction_score: float  # 平均预测评分

    # 教训总结
    lessons_learned: List[str] = field(default_factory=list)

    # 改进建议
    improvement_suggestions: List[str] = field(default_factory=list)

    # 详细复盘结果
    review_results: List[ReviewResult] = field(default_factory=list)


class DailyReviewer:
    """每日复盘器"""

    def __init__(self, db_path: str = None):
        self.market_client = get_market_client()
        self.db_path = db_path or 'data/weibo.db'

    def review_predictions(self, review_date: date = None) -> DailyReviewSummary:
        """
        复盘当日预测（实际是复盘前一日对今日的预测）

        Args:
            review_date: 复盘日期，默认为今天（验证昨天对今天的预测）

        Returns:
            DailyReviewSummary: 复盘总结
        """
        if review_date is None:
            review_date = date.today()

        # 获取前一日对今日的预测记录（prediction_date = 今天，表示预测目标是今天）
        predictions = self._get_predictions(review_date)

        if not predictions:
            print(f"没有找到 {review_date} 的预测记录")
            return DailyReviewSummary(
                review_date=review_date,
                review_time=datetime.now(),
                total_predictions=0,
                correct_predictions=0,
                accuracy_rate=0,
                avg_change_diff=0,
                avg_amplitude_diff=0,
                avg_prediction_score=0
            )

        # 对每只股票进行复盘
        review_results = []
        for pred in predictions:
            result = self._review_single_stock(pred)
            if result:
                review_results.append(result)

        # 生成总结
        summary = self._generate_summary(review_results, review_date)

        # 保存复盘结果
        self._save_review_results(review_results)
        self._save_review_summary(summary)

        # 复盘→预测反馈闭环：根据本次复盘数据自动调整预测参数
        if review_results:
            self._trigger_feedback_adjustment()

        # 复盘后生成组合风险报告（VaR/压力测试）
        self._generate_risk_report()

        return summary

    def _get_predictions(self, pred_date: date) -> List[Dict]:
        """从数据库获取预测记录

        Args:
            pred_date: 预测目标日期（获取预测这个日期的记录）
                       同时也会捞取 prediction_date < pred_date 且 status='pending' 的过期未复盘记录
        """
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        # 查找所有已到期但仍未复盘的预测（prediction_date <= 今天 且 status = 'pending')
        cursor.execute("""
            SELECT stock_code, stock_name, expected_close_change, expected_amplitude,
                   expected_trend, expected_hold_days, selection_reason, technical_basis,
                   prediction_date, blogger_name
            FROM predictions
            WHERE prediction_date <= ? AND status = 'pending'
            ORDER BY prediction_date ASC
        """, (pred_date,))

        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                'code': row[0],
                'name': row[1],
                'expected_change': row[2],
                'expected_amplitude': row[3],
                'expected_trend': row[4],
                'expected_hold_days': row[5],
                'selection_reason': row[6],
                'technical_basis': json.loads(row[7]) if row[7] else {},
                'prediction_date': row[8],
                'blogger_name': row[9]
            })

        conn.close()
        return predictions

    def _fetch_kline_for_date(self, code: str, target_date: date) -> Optional[Dict]:
        """通过腾讯财经K线API获取指定股票在指定日期的K线数据

        Args:
            code: 股票代码 (如 600316, 000001, sh600000, sz000001)
            target_date: 目标日期

        Returns:
            Dict with open/high/low/close/change_pct/amplitude, or None
        """
        try:
            # 转换为腾讯代码格式 (sh600316, sz000001)
            pure_code = code.replace('sh', '').replace('sz', '').replace('SH', '').replace('SZ', '')
            if pure_code.startswith('6') or pure_code.startswith('9'):
                tencent_code = f"sh{pure_code}"
            else:
                tencent_code = f"sz{pure_code}"

            # 获取目标日期前后30天的K线数据
            start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=3)).strftime('%Y-%m-%d')
            url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tencent_code},day,{start_date},{end_date},30,qfq"
            resp = requests.get(url, timeout=10)
            data = resp.json()

            # 解析腾讯返回格式: data -> {tencent_code: {qfqday: [[date, open, close, high, low, volume], ...]}}
            stock_data = data.get('data', {}).get(tencent_code, {})
            klines = stock_data.get('qfqday', [])
            if not klines:
                klines = stock_data.get('day', [])

            target_str = target_date.strftime('%Y-%m-%d')
            for i, k in enumerate(klines):
                if k[0] == target_str:
                    open_p = float(k[1])
                    close_p = float(k[2])
                    high_p = float(k[3])
                    low_p = float(k[4])
                    # 用前一根K线收盘价计算涨跌幅
                    prev_close = open_p
                    if i > 0:
                        prev_close = float(klines[i - 1][2])
                    change_pct = (close_p - prev_close) / prev_close * 100 if prev_close > 0 else 0
                    amplitude = (high_p - low_p) / prev_close * 100 if prev_close > 0 else 0
                    return {
                        'open': open_p,
                        'high': high_p,
                        'low': low_p,
                        'close': close_p,
                        'change_pct': round(change_pct, 2),
                        'amplitude': round(amplitude, 2)
                    }

            print(f"腾讯K线未找到 {code} 在 {target_str} 的数据")
            return None
        except Exception as e:
            print(f"获取 {code} K线数据失败: {e}")
            return None

    def _fetch_realtime_quote(self, code: str) -> Optional[Dict]:
        """通过新浪实时API获取股票当天行情

        Args:
            code: 股票代码

        Returns:
            Dict with open/high/low/close/change_pct/amplitude, or None
        """
        try:
            pure_code = code.replace('sh', '').replace('sz', '').replace('SH', '').replace('SZ', '')
            if code.upper().startswith('SZ') or code.upper().startswith('0'):
                sina_code = f"sz{pure_code}"
            elif code.upper().startswith('SH') or code.upper().startswith('6'):
                sina_code = f"sh{pure_code}"
            elif pure_code.startswith('6'):
                sina_code = f"sh{pure_code}"
            else:
                sina_code = f"sz{pure_code}"

            url = f"https://hq.sinajs.cn/list={sina_code}"
            headers = {'Referer': 'https://finance.sina.com.cn'}
            resp = requests.get(url, timeout=10, headers=headers)
            resp.encoding = 'gbk'
            lines = resp.text.strip().split('\n')

            for line in lines:
                if '="' not in line:
                    continue
                data_str = line.split('="')[1].rstrip('"; ')
                fields = data_str.split(',')
                if len(fields) < 32:
                    continue

                open_p = float(fields[1]) if fields[1] else 0
                prev_close = float(fields[2]) if fields[2] else 0
                close_p = float(fields[3]) if fields[3] else 0
                high_p = float(fields[4]) if fields[4] else 0
                low_p = float(fields[5]) if fields[5] else 0

                if prev_close <= 0:
                    continue

                change_pct = (close_p - prev_close) / prev_close * 100
                amplitude = (high_p - low_p) / prev_close * 100

                return {
                    'open': open_p,
                    'high': high_p,
                    'low': low_p,
                    'close': close_p,
                    'change_pct': round(change_pct, 2),
                    'amplitude': round(amplitude, 2)
                }

            return None
        except Exception as e:
            print(f"获取 {code} 实时行情失败: {e}")
            return None

    def _get_actual_data(self, code: str, prediction_date: date) -> Optional[Dict]:
        """获取股票实际行情数据，根据 prediction_date 选择数据源

        如果 prediction_date == 今天 → 使用实时行情
        如果 prediction_date < 今天 → 使用历史K线数据
        """
        today = date.today()

        if prediction_date >= today:
            # 当天或未来的预测，用实时行情
            return self._fetch_realtime_quote(code)
        else:
            # 过去日期，用历史K线
            return self._fetch_kline_for_date(code, prediction_date)

    def _review_single_stock(self, prediction: Dict) -> Optional[ReviewResult]:
        """复盘单只股票"""
        code = prediction['code']

        # 解析 prediction_date
        pred_date = prediction['prediction_date']
        if isinstance(pred_date, str):
            pred_date = date.fromisoformat(pred_date)

        # 获取实际数据（根据 prediction_date 自动选择实时/历史）
        actual = self._get_actual_data(code, pred_date)

        if not actual:
            print(f"无法获取股票 {code} 在 {pred_date} 的实际数据，跳过")
            return None

        # 计算差异
        expected_change = prediction['expected_change']
        actual_change = actual['change_pct']
        change_diff = actual_change - expected_change

        expected_amplitude = prediction['expected_amplitude']
        actual_amplitude = actual['amplitude']
        amplitude_diff = actual_amplitude - expected_amplitude

        # 判断实际趋势
        actual_trend = self._determine_actual_trend(actual)

        # 判断趋势是否匹配（方向+幅度双验证）
        expected_trend = prediction['expected_trend']
        trend_match = self._check_trend_match(
            expected_trend, actual_trend,
            expected_change=expected_change, actual_change=actual_change
        )

        # 分析差异原因
        diff_reason = self._analyze_diff_reason(
            prediction, actual, change_diff, amplitude_diff
        )

        # 生成改进建议
        improvement = self._generate_improvement(
            prediction, actual, change_diff, trend_match
        )

        # 计算预测评分
        score = self._calculate_prediction_score(
            change_diff, amplitude_diff, trend_match,
            expected_change=expected_change, actual_change=actual_change
        )

        return ReviewResult(
            stock_code=code,
            stock_name=prediction['name'],
            review_date=date.today(),
            prediction_date=prediction['prediction_date'],
            expected_change=expected_change,
            expected_amplitude=expected_amplitude,
            expected_trend=expected_trend,
            expected_hold_days=prediction['expected_hold_days'],
            actual_open=actual['open'],
            actual_high=actual['high'],
            actual_low=actual['low'],
            actual_close=actual['close'],
            actual_change=actual_change,
            actual_amplitude=actual_amplitude,
            actual_trend=actual_trend,
            change_diff=change_diff,
            amplitude_diff=amplitude_diff,
            trend_match=trend_match,
            diff_reason=diff_reason,
            improvement_suggestion=improvement,
            prediction_score=score,
            blogger_name=prediction.get('blogger_name')
        )

    def _determine_actual_trend(self, actual: Dict) -> str:
        """判断实际趋势"""
        change = actual['change_pct']

        if change > 3:
            return "强势上涨"
        elif change > 1:
            return "上涨"
        elif change > 0:
            return "微涨"
        elif change < -3:
            return "强势下跌"
        elif change < -1:
            return "下跌"
        elif change < 0:
            return "微跌"
        else:
            return "横盘震荡"

    def _check_trend_match(self, expected: str, actual: str,
                           expected_change: float = 0, actual_change: float = 0) -> bool:
        """检查趋势是否匹配——方向一致即可，幅度差距过大则降级为部分匹配

        核心原则：方向对就算匹配，但幅度偏差超过3个百分点不算
        特殊：横盘/震荡 与 微涨/微跌 算方向一致（±1%内都算横盘区间）
        """
        # 提取方向：上涨/下跌/横盘
        expected_up = any(kw in expected for kw in ['上涨', '强', '涨'])
        expected_down = any(kw in expected for kw in ['下跌', '弱', '跌'])
        expected_flat = '震荡' in expected or '横盘' in expected

        actual_up = any(kw in actual for kw in ['上涨', '强', '涨'])
        actual_down = any(kw in actual for kw in ['下跌', '弱', '跌'])
        actual_flat = '震荡' in actual or '横盘' in actual
        actual_slight = '微' in actual  # 微涨/微跌

        # 横盘/震荡 与 微涨/微跌 算方向一致
        if expected_flat and (actual_flat or actual_slight or abs(actual_change) <= 1.0):
            return True
        if actual_flat and (expected_flat or abs(expected_change) <= 1.0):
            return True

        # 方向一致 + 幅度偏差不超过3个百分点
        if expected_up and actual_up:
            return abs(expected_change - actual_change) <= 3.0
        if expected_down and actual_down:
            return abs(expected_change - actual_change) <= 3.0

        # 震荡上涨 与 上涨 方向一致
        if expected_up and actual_up:
            return abs(expected_change - actual_change) <= 3.0

        return False

    def _analyze_diff_reason(self, prediction: Dict, actual: Dict,
                              change_diff: float, amplitude_diff: float) -> str:
        """分析差异原因 - 改进版"""
        reasons = []

        # 涨跌幅差异分析
        if abs(change_diff) > IMPROVEMENT_THRESHOLDS['change_diff_critical']:
            if change_diff > 0:
                reasons.append("实际涨幅大幅超出预期，市场情绪或板块热点带动强劲")
            else:
                reasons.append("实际涨幅大幅低于预期，可能受大盘拖累或个股利空影响")
        elif abs(change_diff) > IMPROVEMENT_THRESHOLDS['change_diff_warning']:
            if change_diff > 0:
                reasons.append("实际涨幅超出预期，可能受市场情绪或板块热点带动")
            else:
                reasons.append("实际涨幅低于预期，可能受大盘拖累或个股利空影响")

        # 振幅差异分析
        if abs(amplitude_diff) > IMPROVEMENT_THRESHOLDS['amplitude_diff_critical']:
            if amplitude_diff > 0:
                reasons.append("波动大幅大于预期，市场分歧剧烈，需关注风险")
            else:
                reasons.append("波动大幅小于预期，市场观望情绪浓厚")
        elif abs(amplitude_diff) > IMPROVEMENT_THRESHOLDS['amplitude_diff_warning']:
            if amplitude_diff > 0:
                reasons.append("波动大于预期，市场分歧较大")
            else:
                reasons.append("波动小于预期，市场观望情绪浓厚")

        # 技术指标失效分析
        tech = prediction.get('technical_basis', {})
        if tech.get('trend') == '上涨' and actual['change_pct'] < 0:
            reasons.append("技术趋势判断失效，需关注趋势反转信号")

        # 【改进】市场情绪影响分析
        market_basis = prediction.get('market_basis', {})
        sentiment = market_basis.get('sentiment', {})
        if sentiment:
            avg_change = sentiment.get('avg_change', 0)
            if abs(avg_change) > 1 and abs(change_diff) > 2:
                if avg_change > 0 and change_diff > 0:
                    reasons.append("大盘上涨带动个股走强")
                elif avg_change < 0 and change_diff < 0:
                    reasons.append("大盘下跌拖累个股走弱")

        if reasons:
            return "；".join(reasons)
        return "预测与实际基本吻合"

    def _generate_improvement(self, prediction: Dict, actual: Dict,
                               change_diff: float, trend_match: bool) -> str:
        """生成改进建议 - 改进版"""
        suggestions = []

        if not trend_match:
            suggestions.append("需加强对市场整体趋势的判断")

        # 【改进】根据偏差程度给出针对性建议
        if abs(change_diff) > IMPROVEMENT_THRESHOLDS['change_diff_critical']:
            suggestions.append("涨跌幅预测偏差严重，应重点加强市场研判和板块轮动分析")
        elif abs(change_diff) > IMPROVEMENT_THRESHOLDS['change_diff_warning']:
            suggestions.append("应更关注当日市场情绪和板块轮动")

        # 根据具体情况给出建议
        tech = prediction.get('technical_basis', {})
        if tech.get('volume_change_pct', 0) > 50 and actual['change_pct'] < 0:
            suggestions.append("放量下跌需谨慎，应结合资金流向判断")

        # 【改进】波动性相关建议
        volatility = tech.get('volatility', 0)
        if volatility > 40 and abs(change_diff) > 2:
            suggestions.append("高波动性个股需更保守预测，可参考ATR指标")

        # 【改进】振幅预测建议
        amplitude_diff = actual.get('amplitude', 0) - prediction.get('expected_amplitude', 0)
        if abs(amplitude_diff) > IMPROVEMENT_THRESHOLDS['amplitude_diff_critical']:
            suggestions.append("振幅预测偏差较大，应关注波动性指标(ATR、历史振幅)")

        # 【改进】市场情绪建议
        market_basis = prediction.get('market_basis', {})
        sentiment = market_basis.get('sentiment', {})
        if sentiment:
            sentiment_type = sentiment.get('sentiment', 'neutral')
            if sentiment_type in ['strong_bull', 'strong_bear'] and abs(change_diff) > 2:
                suggestions.append("极端市场情绪下，应更重视大盘对个股的影响")

        if suggestions:
            return "改进建议：" + "；".join(suggestions)
        return "预测准确，可继续沿用当前分析方法"

    def _calculate_prediction_score(self, change_diff: float,
                                     amplitude_diff: float, trend_match: bool,
                                     expected_change: float = 0,
                                     actual_change: float = 0) -> int:
        """计算预测评分——连续评分 + 方向错误惩罚

        评分构成：
        - 基础分 50
        - 涨跌幅连续评分：max(0, 40 - abs(change_diff) * 15)
        - 振幅连续评分：max(0, 20 - abs(amplitude_diff) * 8)
        - 趋势方向错误惩罚：-25
        """
        score = 50  # 基础分

        # 涨跌幅连续评分（替代阶梯加分，分数分布更宽）
        change_score = max(0, 40 - abs(change_diff) * 15)
        score += change_score

        # 振幅连续评分
        amplitude_score = max(0, 20 - abs(amplitude_diff) * 8)
        score += amplitude_score

        # 方向错误惩罚：预期涨实际跌，或预期跌实际涨
        if not trend_match:
            expected_up = expected_change > 0
            actual_down = actual_change < 0
            expected_down = expected_change < 0
            actual_up = actual_change > 0
            if (expected_up and actual_down) or (expected_down and actual_up):
                score -= 25

        return max(0, min(round(score), 100))

    def _generate_summary(self, results: List[ReviewResult],
                          review_date: date) -> DailyReviewSummary:
        """生成复盘总结 - 改进版"""
        total = len(results)
        correct = sum(1 for r in results if r.trend_match)

        avg_change_diff = sum(r.change_diff for r in results) / total if total > 0 else 0
        avg_amplitude_diff = sum(r.amplitude_diff for r in results) / total if total > 0 else 0
        avg_score = sum(r.prediction_score for r in results) / total if total > 0 else 0

        # 总结教训
        lessons = []
        low_score_results = [r for r in results if r.prediction_score < 60]
        for r in low_score_results[:3]:
            lessons.append(f"{r.stock_name}: {r.diff_reason}")

        # 【改进】更精细的改进建议
        improvements = []

        # 准确率分析
        accuracy_rate = correct / total * 100 if total > 0 else 0
        if accuracy_rate < IMPROVEMENT_THRESHOLDS['accuracy_warning']:
            improvements.append("整体预测准确率偏低，需优化选股逻辑")

        # 涨跌幅偏差分析
        if abs(avg_change_diff) > IMPROVEMENT_THRESHOLDS['change_diff_critical']:
            improvements.append("涨跌幅预测偏差严重，应重点加强市场研判，考虑大盘情绪因子")
        elif abs(avg_change_diff) > IMPROVEMENT_THRESHOLDS['change_diff_warning']:
            improvements.append("涨跌幅预测偏差较大，应加强市场研判")

        # 振幅偏差分析
        if abs(avg_amplitude_diff) > IMPROVEMENT_THRESHOLDS['amplitude_diff_critical']:
            improvements.append("振幅预测偏差严重，应重点关注波动性指标(ATR、历史振幅)")
        elif abs(avg_amplitude_diff) > IMPROVEMENT_THRESHOLDS['amplitude_diff_warning']:
            improvements.append("振幅预测偏差较大，应关注波动性指标")

        # 【改进】统计偏差方向
        positive_change_diff = sum(1 for r in results if r.change_diff > 0)
        negative_change_diff = sum(1 for r in results if r.change_diff < 0)
        if positive_change_diff > negative_change_diff * 2:
            improvements.append("预测偏保守，实际表现优于预期，可适当提高预测值")
        elif negative_change_diff > positive_change_diff * 2:
            improvements.append("预测偏乐观，实际表现弱于预期，应更保守预测")

        return DailyReviewSummary(
            review_date=review_date,
            review_time=datetime.now(),
            total_predictions=total,
            correct_predictions=correct,
            accuracy_rate=round(accuracy_rate, 2),
            avg_change_diff=round(avg_change_diff, 2),
            avg_amplitude_diff=round(avg_amplitude_diff, 2),
            avg_prediction_score=round(avg_score, 2),
            lessons_learned=lessons,
            improvement_suggestions=improvements,
            review_results=results
        )

    def _generate_risk_report(self):
        """复盘后生成组合风险报告（VaR/压力测试）"""
        try:
            from app.selector.stock_selector import generate_risk_report
            result = generate_risk_report(db_path=self.db_path)
            if result:
                var_info = result.get('portfolio_var', {})
                print(f"[风险度量] VaR={var_info.get('var_value', 0):.2f}% "
                      f"CVaR={var_info.get('cvar', 0):.2f}% "
                      f"评级={result.get('risk_rating', '未知')}")
        except Exception as e:
            print(f"风险报告生成失败: {e}")

    def _trigger_feedback_adjustment(self):
        """复盘完成后触发反馈调整，将复盘偏差写回预测器参数"""
        try:
            from app.reviewer.feedback_adjuster import FeedbackAdjuster
            adjuster = FeedbackAdjuster(db_path=self.db_path)
            result = adjuster.analyze_feedback(days=5)
            if result.total_samples > 0:
                adjuster.save_adjustment(result)
                print(f"[反馈闭环] {result.summary}")
        except Exception as e:
            print(f"反馈调整失败: {e}")

    def _save_review_results(self, results: List[ReviewResult]):
        """保存复盘结果到数据库"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        # 创建复盘表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT,
                prediction_date DATE,
                review_date DATE NOT NULL,

                expected_change REAL,
                actual_change REAL,
                change_diff REAL,

                expected_amplitude REAL,
                actual_amplitude REAL,
                amplitude_diff REAL,

                expected_trend TEXT,
                actual_trend TEXT,
                trend_match INTEGER,

                diff_reason TEXT,
                improvement_suggestion TEXT,
                prediction_score INTEGER,

                blogger_name TEXT,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 兼容旧表：添加新列
        for col, default in [('prediction_date', 'NULL'), ('blogger_name', 'NULL')]:
            try:
                cursor.execute(f"ALTER TABLE review_results ADD COLUMN {col} TEXT DEFAULT {default}")
            except sqlite3.OperationalError:
                pass

        # 插入复盘结果
        for r in results:
            cursor.execute("""
                INSERT INTO review_results
                (stock_code, stock_name, prediction_date, review_date,
                 expected_change, actual_change, change_diff,
                 expected_amplitude, actual_amplitude, amplitude_diff,
                 expected_trend, actual_trend, trend_match,
                 diff_reason, improvement_suggestion, prediction_score,
                 blogger_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.stock_code, r.stock_name, r.prediction_date, r.review_date,
                r.expected_change, r.actual_change, r.change_diff,
                r.expected_amplitude, r.actual_amplitude, r.amplitude_diff,
                r.expected_trend, r.actual_trend, 1 if r.trend_match else 0,
                r.diff_reason, r.improvement_suggestion, r.prediction_score,
                r.blogger_name
            ))

        # 更新预测状态（用 prediction_date 匹配，而非 review_date）
        for r in results:
            cursor.execute("""
                UPDATE predictions
                SET status = 'reviewed',
                    actual_change = ?,
                    actual_amplitude = ?,
                    diff_reason = ?,
                    improvement_suggestion = ?
                WHERE stock_code = ? AND prediction_date = ?
            """, (
                r.actual_change, r.actual_amplitude, r.diff_reason, r.improvement_suggestion,
                r.stock_code, r.prediction_date
            ))

        conn.commit()
        conn.close()

    def _save_review_summary(self, summary: DailyReviewSummary):
        """保存复盘总结"""
        conn = get_sync_db(self.db_path)
        cursor = conn.cursor()

        # 创建总结表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_date DATE NOT NULL,
                review_time DATETIME,

                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy_rate REAL,

                avg_change_diff REAL,
                avg_amplitude_diff REAL,
                avg_prediction_score REAL,

                lessons_learned TEXT,
                improvement_suggestions TEXT,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            INSERT INTO daily_reviews
            (review_date, review_time, total_predictions, correct_predictions,
             accuracy_rate, avg_change_diff, avg_amplitude_diff, avg_prediction_score,
             lessons_learned, improvement_suggestions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.review_date, summary.review_time,
            summary.total_predictions, summary.correct_predictions,
            summary.accuracy_rate, summary.avg_change_diff,
            summary.avg_amplitude_diff, summary.avg_prediction_score,
            json.dumps(summary.lessons_learned, ensure_ascii=False),
            json.dumps(summary.improvement_suggestions, ensure_ascii=False)
        ))

        conn.commit()
        conn.close()

    def generate_review_report(self, summary: DailyReviewSummary) -> str:
        """生成复盘报告"""
        report = f"""
# 📊 每日复盘报告

> 复盘日期：{summary.review_date}
> 复盘时间：{summary.review_time.strftime('%Y-%m-%d %H:%M')}

---

## 一、整体表现

| 指标 | 数值 |
|------|------|
| 预测总数 | {summary.total_predictions} |
| 预测正确 | {summary.correct_predictions} |
| 准确率 | {summary.accuracy_rate:.1f}% |
| 平均涨跌幅偏差 | {summary.avg_change_diff:+.2f}% |
| 平均振幅偏差 | {summary.avg_amplitude_diff:.2f}% |
| 平均预测评分 | {summary.avg_prediction_score:.1f}分 |

---

## 二、个股复盘详情

"""

        for r in summary.review_results:
            status = "✅" if r.trend_match else "❌"
            report += f"""
### {status} {r.stock_code} {r.stock_name}

| 对比项 | 预测 | 实际 | 差异 |
|------|------|------|------|
| 涨跌幅 | {r.expected_change:+.2f}% | {r.actual_change:+.2f}% | {r.change_diff:+.2f}% |
| 振幅 | {r.expected_amplitude:.2f}% | {r.actual_amplitude:.2f}% | {r.amplitude_diff:.2f}% |
| 趋势 | {r.expected_trend} | {r.actual_trend} | {'匹配' if r.trend_match else '不匹配'} |

**差异原因**
> {r.diff_reason}

**改进建议**
> {r.improvement_suggestion}

**预测评分**: {r.prediction_score}分

---

"""

        report += """
## 三、今日教训

"""
        for lesson in summary.lessons_learned:
            report += f"- {lesson}\n"

        report += """
## 四、改进方向

"""
        for suggestion in summary.improvement_suggestions:
            report += f"- {suggestion}\n"

        report += """
---

## 📝 复盘说明

本复盘基于当日实际行情数据，对比预测结果：
- 准确率统计基于趋势方向判断
- 评分综合考虑趋势、涨跌幅、振幅预测准确度
- 改进建议用于优化后续预测模型
"""

        return report


def run_daily_review():
    """运行每日复盘"""
    reviewer = DailyReviewer()

    print("=" * 70)
    print("开始每日复盘")
    print("=" * 70)

    summary = reviewer.review_predictions()

    if summary.total_predictions > 0:
        print(f"\n复盘完成，共复盘 {summary.total_predictions} 只股票")
        print(f"准确率: {summary.accuracy_rate:.1f}%")
        print(f"平均评分: {summary.avg_prediction_score:.1f}分")

        print("\n" + reviewer.generate_review_report(summary))
    else:
        print("没有需要复盘的预测记录")


def get_review_history(days: int = 30, db_path: str = None) -> List[Dict]:
    """获取历史复盘记录

    Args:
        days: 获取最近多少天的记录
        db_path: 数据库路径

    Returns:
        List[Dict]: 历史复盘记录列表
    """
    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT review_date, review_time, total_predictions, correct_predictions,
               accuracy_rate, avg_change_diff, avg_amplitude_diff, avg_prediction_score,
               lessons_learned, improvement_suggestions
        FROM daily_reviews
        ORDER BY review_date DESC
        LIMIT ?
    """, (days,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'review_date': row[0],
            'review_time': row[1],
            'total_predictions': row[2],
            'correct_predictions': row[3],
            'accuracy_rate': row[4],
            'avg_change_diff': row[5],
            'avg_amplitude_diff': row[6],
            'avg_prediction_score': row[7],
            'lessons_learned': json.loads(row[8]) if row[8] else [],
            'improvement_suggestions': json.loads(row[9]) if row[9] else []
        })

    conn.close()
    return results


def get_review_detail_by_date(review_date: date, db_path: str = None) -> List[Dict]:
    """获取指定日期的复盘详情

    Args:
        review_date: 复盘日期
        db_path: 数据库路径

    Returns:
        List[Dict]: 该日期的复盘详情列表
    """
    if db_path is None:
        db_path = 'data/weibo.db'

    conn = get_sync_db(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT stock_code, stock_name, review_date,
               expected_change, actual_change, change_diff,
               expected_amplitude, actual_amplitude, amplitude_diff,
               expected_trend, actual_trend, trend_match,
               diff_reason, improvement_suggestion, prediction_score
        FROM review_results
        WHERE review_date = ?
        ORDER BY prediction_score DESC
    """, (review_date,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'stock_code': row[0],
            'stock_name': row[1],
            'review_date': row[2],
            'expected_change': row[3],
            'actual_change': row[4],
            'change_diff': row[5],
            'expected_amplitude': row[6],
            'actual_amplitude': row[7],
            'amplitude_diff': row[8],
            'expected_trend': row[9],
            'actual_trend': row[10],
            'trend_match': bool(row[11]),
            'diff_reason': row[12],
            'improvement_suggestion': row[13],
            'prediction_score': row[14]
        })

    conn.close()
    return results


def generate_history_report(days: int = 30) -> str:
    """生成历史复盘汇总报告

    Args:
        days: 统计最近多少天

    Returns:
        str: 报告内容
    """
    history = get_review_history(days)

    if not history:
        return f"最近 {days} 天没有复盘记录"

    # 计算统计数据
    total_predictions = sum(h['total_predictions'] for h in history)
    total_correct = sum(h['correct_predictions'] for h in history)
    avg_accuracy = sum(h['accuracy_rate'] for h in history) / len(history) if history else 0
    avg_score = sum(h['avg_prediction_score'] for h in history) / len(history) if history else 0
    avg_change_diff = sum(h['avg_change_diff'] for h in history) / len(history) if history else 0
    avg_amplitude_diff = sum(h['avg_amplitude_diff'] for h in history) / len(history) if history else 0

    # 统计改进建议
    all_suggestions = []
    for h in history:
        all_suggestions.extend(h['improvement_suggestions'])

    suggestion_counts = {}
    for s in all_suggestions:
        suggestion_counts[s] = suggestion_counts.get(s, 0) + 1

    # 生成报告
    report = f"""
# 📊 历史复盘汇总报告

> 统计周期：最近 {days} 天
> 复盘天数：{len(history)} 天
> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 一、整体统计

| 指标 | 数值 |
|------|------|
| 总预测数 | {total_predictions} |
| 总正确数 | {total_correct} |
| 平均准确率 | {avg_accuracy:.1f}% |
| 平均评分 | {avg_score:.1f}分 |
| 平均涨跌幅偏差 | {avg_change_diff:+.2f}% |
| 平均振幅偏差 | {avg_amplitude_diff:.2f}% |

---

## 二、每日复盘记录

| 日期 | 预测数 | 正确数 | 准确率 | 评分 | 涨跌幅偏差 | 振幅偏差 |
|------|--------|--------|--------|------|------------|----------|
"""

    for h in history:
        report += f"| {h['review_date']} | {h['total_predictions']} | {h['correct_predictions']} | {h['accuracy_rate']:.1f}% | {h['avg_prediction_score']:.1f} | {h['avg_change_diff']:+.2f}% | {h['avg_amplitude_diff']:.2f}% |\n"

    report += """
---

## 三、常见改进建议统计

"""

    if suggestion_counts:
        sorted_suggestions = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)
        for suggestion, count in sorted_suggestions[:10]:
            report += f"- {suggestion}（出现 {count} 次）\n"
    else:
        report += "暂无改进建议记录\n"

    report += """
---

## 四、趋势分析

"""

    # 分析准确率趋势
    if len(history) >= 3:
        recent_3 = history[:3]
        recent_3_avg = sum(h['accuracy_rate'] for h in recent_3) / 3

        if len(history) >= 7:
            recent_7 = history[:7]
            recent_7_avg = sum(h['accuracy_rate'] for h in recent_7) / 7
            older_7 = history[7:14] if len(history) >= 14 else history[7:]
            older_7_avg = sum(h['accuracy_rate'] for h in older_7) / len(older_7) if older_7 else recent_7_avg

            if recent_7_avg > older_7_avg:
                report += f"- 📈 准确率呈上升趋势（近7天 {recent_7_avg:.1f}% vs 前7天 {older_7_avg:.1f}%）\n"
            elif recent_7_avg < older_7_avg:
                report += f"- 📉 准确率呈下降趋势（近7天 {recent_7_avg:.1f}% vs 前7天 {older_7_avg:.1f}%）\n"
            else:
                report += f"- ➡️ 准确率保持稳定（近7天 {recent_7_avg:.1f}%）\n"

        # 涨跌幅偏差趋势
        recent_change_diff = sum(h['avg_change_diff'] for h in recent_3) / 3
        if abs(recent_change_diff) > 2:
            if recent_change_diff > 0:
                report += f"- ⚠️ 近期预测偏保守，实际表现优于预期（偏差 {recent_change_diff:+.2f}%）\n"
            else:
                report += f"- ⚠️ 近期预测偏乐观，实际表现弱于预期（偏差 {recent_change_diff:+.2f}%）\n"

    report += """
---

## 📝 说明

- 本报告基于历史复盘数据自动生成
- 建议定期查看，持续优化预测模型
"""

    return report


if __name__ == "__main__":
    run_daily_review()