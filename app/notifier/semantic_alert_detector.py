"""语义分析预警检测器 - 将语义分析结果转化为实时预警

P1功能：桥接 EnhancedSemanticAnalyzer 和 AlertManager，
当博主发布包含明确买卖信号、情绪转变或实盘操作的微博时，
自动触发预警通知。
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SemanticResult:
    """语义分析结果（轻量级，用于检测器输入）"""
    action_type: str  # buy/sell/hold/watch/no_position
    action_confidence: float  # 0.0-1.0
    scenario_type: str  # real_trading/simulation/example/discussion/other
    is_formal_opinion: bool
    true_sentiment: str  # bullish/bearish/neutral/sarcastic
    reasoning: str
    stock_names: List[str] = field(default_factory=list)
    quality_score: float = 1.0


class SemanticAlertDetector:
    """语义分析预警检测器

    将增强版语义分析器的输出转化为预警事件：
    - 买入/卖出信号预警：博主明确表达买卖意图
    - 情绪转变预警：博主对某只股票的情绪从看多转看空或反之
    - 实盘操作预警：高置信度的实盘交易信号
    """

    def __init__(self, alert_manager=None,
                 min_action_confidence: float = 0.6,
                 min_sentiment_confidence: float = 0.7,
                 cooldown_seconds: int = 600):
        self.alert_manager = alert_manager
        self.min_action_confidence = min_action_confidence
        self.min_sentiment_confidence = min_sentiment_confidence
        self.cooldown_seconds = cooldown_seconds

        # 博主情绪追踪：{blogger_id: {stock_name: sentiment}}
        self._blogger_sentiments: Dict[str, Dict[str, str]] = {}
        # 冷却追踪：{(blogger_id, alert_type, stock_name): last_trigger_time}
        self._cooldown_tracker: Dict[Tuple[str, str, str], float] = {}

    def process_analysis_result(self, result: SemanticResult,
                                 blogger_id: str, blogger_name: str) -> List:
        """处理语义分析结果，生成预警

        Args:
            result: 语义分析结果
            blogger_id: 博主ID
            blogger_name: 博主名称

        Returns:
            触发的预警列表
        """
        if self.alert_manager is None:
            return []

        triggered_alerts = []

        # 1. 买卖信号检测
        action_alert = self._check_action_alert(result, blogger_id, blogger_name)
        if action_alert:
            triggered_alerts.append(action_alert)

        # 2. 情绪转变检测
        sentiment_alert = self._check_sentiment_shift(result, blogger_id, blogger_name)
        if sentiment_alert:
            triggered_alerts.append(sentiment_alert)

        # 3. 实盘操作场景检测
        scenario_alert = self._check_scenario_alert(result, blogger_id, blogger_name)
        if scenario_alert:
            triggered_alerts.append(scenario_alert)

        # 更新情绪追踪
        for stock_name in result.stock_names:
            self._blogger_sentiments.setdefault(blogger_id, {})[stock_name] = result.true_sentiment

        return triggered_alerts

    def _check_action_alert(self, result: SemanticResult,
                             blogger_id: str, blogger_name: str) -> Optional[object]:
        """检测买卖信号预警

        触发条件：
        - action_confidence >= min_action_confidence
        - scenario_type 为 real_trading
        - is_formal_opinion 为 True
        - true_sentiment 不是 sarcastic
        """
        if result.action_confidence < self.min_action_confidence:
            return None

        if result.scenario_type not in ("real_trading", "other"):
            return None

        if not result.is_formal_opinion:
            return None

        if result.true_sentiment == "sarcastic":
            return None

        if result.action_type == "buy":
            alert_type_key = "semantic_buy_signal"
            from app.notifier.realtime_alert import AlertType, AlertLevel
            alert_type = AlertType.SEMANTIC_BUY_SIGNAL
            title_prefix = "买入信号"
        elif result.action_type == "sell":
            alert_type_key = "semantic_sell_signal"
            from app.notifier.realtime_alert import AlertType, AlertLevel
            alert_type = AlertType.SEMANTIC_SELL_SIGNAL
            title_prefix = "卖出信号"
        else:
            return None

        # 冷却检查
        for stock_name in result.stock_names:
            if self._is_in_cooldown(blogger_id, alert_type_key, stock_name):
                return None

        stock_display = ", ".join(result.stock_names) if result.stock_names else "未知股票"
        title = f"{blogger_name} {title_prefix}: {stock_display}"
        message = (f"博主【{blogger_name}】发布{title_prefix}，"
                   f"置信度: {result.action_confidence:.0%}\n"
                   f"股票: {stock_display}\n"
                   f"分析: {result.reasoning}")

        alert = self.alert_manager.create_alert(
            alert_type=alert_type,
            level=AlertLevel.HIGH if result.action_confidence >= 0.8 else AlertLevel.MEDIUM,
            title=title,
            message=message,
            detail={
                "blogger_id": blogger_id,
                "blogger_name": blogger_name,
                "action_type": result.action_type,
                "confidence": result.action_confidence,
                "scenario_type": result.scenario_type,
                "stock_names": result.stock_names,
                "reasoning": result.reasoning,
            }
        )

        # 记录冷却
        for stock_name in result.stock_names:
            self._set_cooldown(blogger_id, alert_type_key, stock_name)

        return alert

    def _check_sentiment_shift(self, result: SemanticResult,
                                blogger_id: str, blogger_name: str) -> Optional[object]:
        """检测情绪转变预警

        触发条件：
        - 博主之前对某股票有明确情绪（bullish/bearish）
        - 当前情绪与之前相反
        - action_confidence >= min_sentiment_confidence
        """
        if result.action_confidence < self.min_sentiment_confidence:
            return None

        if result.true_sentiment not in ("bullish", "bearish"):
            return None

        for stock_name in result.stock_names:
            prev = self._blogger_sentiments.get(blogger_id, {}).get(stock_name)
            if prev is None or prev == result.true_sentiment:
                continue

            # 情绪发生了转变
            if {prev, result.true_sentiment} != {"bullish", "bearish"}:
                continue

            if self._is_in_cooldown(blogger_id, "semantic_sentiment_shift", stock_name):
                continue

            from app.notifier.realtime_alert import AlertType, AlertLevel

            shift_desc = "看多→看空" if prev == "bullish" else "看空→看多"
            title = f"{blogger_name} 情绪转变: {stock_name} {shift_desc}"
            message = (f"博主【{blogger_name}】对【{stock_name}】情绪发生转变: {shift_desc}\n"
                       f"当前情绪: {result.true_sentiment}\n"
                       f"置信度: {result.action_confidence:.0%}\n"
                       f"分析: {result.reasoning}")

            alert = self.alert_manager.create_alert(
                alert_type=AlertType.SEMANTIC_SENTIMENT_SHIFT,
                level=AlertLevel.HIGH,
                title=title,
                message=message,
                detail={
                    "blogger_id": blogger_id,
                    "blogger_name": blogger_name,
                    "stock_name": stock_name,
                    "previous_sentiment": prev,
                    "current_sentiment": result.true_sentiment,
                    "confidence": result.action_confidence,
                    "reasoning": result.reasoning,
                }
            )

            self._set_cooldown(blogger_id, "semantic_sentiment_shift", stock_name)
            return alert

        return None

    def _check_scenario_alert(self, result: SemanticResult,
                               blogger_id: str, blogger_name: str) -> Optional[object]:
        """检测实盘操作场景预警

        触发条件：
        - scenario_type 为 real_trading
        - action_confidence >= 0.8（高置信度）
        - action_type 为 buy 或 sell
        """
        if result.scenario_type != "real_trading":
            return None

        if result.action_confidence < 0.8:
            return None

        if result.action_type not in ("buy", "sell"):
            return None

        for stock_name in result.stock_names:
            if self._is_in_cooldown(blogger_id, "semantic_scenario_alert", stock_name):
                return None

        from app.notifier.realtime_alert import AlertType, AlertLevel

        stock_display = ", ".join(result.stock_names) if result.stock_names else "未知股票"
        action_desc = "买入" if result.action_type == "buy" else "卖出"
        title = f"实盘操作预警: {blogger_name} {action_desc} {stock_display}"
        message = (f"博主【{blogger_name}】发布高置信度实盘{action_desc}信号！\n"
                   f"股票: {stock_display}\n"
                   f"置信度: {result.action_confidence:.0%}\n"
                   f"场景: 实盘交易\n"
                   f"分析: {result.reasoning}")

        alert = self.alert_manager.create_alert(
            alert_type=AlertType.SEMANTIC_SCENARIO_ALERT,
            level=AlertLevel.CRITICAL,
            title=title,
            message=message,
            detail={
                "blogger_id": blogger_id,
                "blogger_name": blogger_name,
                "action_type": result.action_type,
                "confidence": result.action_confidence,
                "scenario_type": result.scenario_type,
                "stock_names": result.stock_names,
                "reasoning": result.reasoning,
            }
        )

        for stock_name in result.stock_names:
            self._set_cooldown(blogger_id, "semantic_scenario_alert", stock_name)

        return alert

    def _is_in_cooldown(self, blogger_id: str, alert_type: str, stock_name: str) -> bool:
        """检查是否在冷却期内"""
        key = (blogger_id, alert_type, stock_name)
        last_time = self._cooldown_tracker.get(key, 0)
        return (time.time() - last_time) < self.cooldown_seconds

    def _set_cooldown(self, blogger_id: str, alert_type: str, stock_name: str):
        """设置冷却"""
        key = (blogger_id, alert_type, stock_name)
        self._cooldown_tracker[key] = time.time()

    def cleanup_cooldowns(self):
        """清理过期的冷却记录"""
        now = time.time()
        expired = [k for k, v in self._cooldown_tracker.items()
                   if (now - v) > self.cooldown_seconds * 2]
        for key in expired:
            del self._cooldown_tracker[key]
