"""
实时预警系统
功能：
1. 价格预警 - 突破/跌破关键价位
2. 异动预警 - 涨跌幅、成交量异动
3. 博主动态预警 - 新微博、策略变化
4. 风险预警 - VaR超限、回撤预警
5. 预警规则引擎 - 灵活配置预警规则
6. 多渠道通知 - 微信/邮件/Webhook
"""
import os
import json
import asyncio
import threading
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import traceback

import numpy as np
import pandas as pd


# ==================== 预警类型定义 ====================

class AlertType(Enum):
    """预警类型"""
    PRICE_BREAK = "price_break"  # 价格突破
    PRICE_DROP = "price_drop"  # 价格跌破
    SURGE = "surge"  # 涨幅异动
    PLUNGE = "plunge"  # 跌幅异动
    VOLUME_ANOMALY = "volume_anomaly"  # 成交量异动
    BLOGGER_POST = "blogger_post"  # 博主发博
    BLOGGER_ACTION = "blogger_action"  # 博主操作
    VAR_BREACH = "var_breach"  # VaR超限
    DRAWDOWN = "drawdown"  # 回撤预警
    STOP_LOSS = "stop_loss"  # 止损预警
    STOP_PROFIT = "stop_profit"  # 止盈预警
    TREND_CHANGE = "trend_change"  # 趋势变化
    ML_PREDICTION = "ml_prediction"  # ML预测预警
    # P1: 语义分析预警类型
    SEMANTIC_BUY_SIGNAL = "semantic_buy_signal"  # 语义分析买入信号
    SEMANTIC_SELL_SIGNAL = "semantic_sell_signal"  # 语义分析卖出信号
    SEMANTIC_SENTIMENT_SHIFT = "semantic_sentiment_shift"  # 情绪转变
    SEMANTIC_SCENARIO_ALERT = "semantic_scenario_alert"  # 实盘操作预警
    CUSTOM = "custom"  # 自定义


class AlertLevel(Enum):
    """预警级别"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """预警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


# ==================== 预警数据结构 ====================

@dataclass
class Alert:
    """预警消息"""
    id: str
    type: AlertType
    level: AlertLevel
    status: AlertStatus = AlertStatus.ACTIVE

    # 预警内容
    title: str = ""
    message: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    # 关联信息
    stock_code: str = ""
    stock_name: str = ""
    blogger_id: str = ""

    # 数值信息
    current_value: float = 0.0
    threshold_value: float = 0.0
    change_pct: float = 0.0

    # 时间信息
    trigger_time: datetime = field(default_factory=datetime.now)
    expire_time: Optional[datetime] = None

    # 通知状态
    notified_channels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type.value,
            'level': self.level.value,
            'status': self.status.value,
            'title': self.title,
            'message': self.message,
            'detail': self.detail,
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'change_pct': self.change_pct,
            'trigger_time': self.trigger_time.isoformat(),
        }


@dataclass
class AlertRule:
    """预警规则"""
    id: str
    name: str
    type: AlertType
    enabled: bool = True

    # 触发条件
    condition: Dict[str, Any] = field(default_factory=dict)
    # 例: {"field": "change_pct", "operator": ">", "value": 5}
    # 例: {"field": "price", "operator": "cross_above", "value": 100}

    # 预警级别
    level: AlertLevel = AlertLevel.MEDIUM

    # 关联标的
    stock_codes: List[str] = field(default_factory=list)
    blogger_ids: List[str] = field(default_factory=list)

    # 冷却时间（秒）
    cooldown: int = 300  # 默认5分钟

    # 有效期
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # 上次触发时间
    last_trigger_time: Optional[datetime] = None

    def should_trigger(self) -> bool:
        """检查是否应该触发（冷却时间检查）"""
        if not self.enabled:
            return False

        now = datetime.now()

        # 检查有效期
        if self.start_time and now < self.start_time:
            return False
        if self.end_time and now > self.end_time:
            return False

        # 检查冷却时间
        if self.last_trigger_time:
            if (now - self.last_trigger_time).total_seconds() < self.cooldown:
                return False

        return True


# ==================== 预警规则引擎 ====================

class AlertRuleEngine:
    """预警规则引擎"""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self._alert_counter = 0

    def add_rule(self, rule: AlertRule):
        """添加规则"""
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str):
        """移除规则"""
        self.rules.pop(rule_id, None)

    def get_rules(self, type_filter: AlertType = None) -> List[AlertRule]:
        """获取规则"""
        rules = list(self.rules.values())
        if type_filter:
            rules = [r for r in rules if r.type == type_filter]
        return rules

    def evaluate(self, data: Dict[str, Any]) -> List[Alert]:
        """评估所有规则"""
        alerts = []

        for rule in self.rules.values():
            if not rule.should_trigger():
                continue

            # 检查标的是否匹配
            if rule.stock_codes:
                stock_code = data.get('stock_code', '')
                if stock_code not in rule.stock_codes:
                    continue

            if rule.blogger_ids:
                blogger_id = data.get('blogger_id', '')
                if blogger_id not in rule.blogger_ids:
                    continue

            # 评估条件
            if self._evaluate_condition(rule.condition, data):
                alert = self._create_alert(rule, data)
                alerts.append(alert)
                rule.last_trigger_time = datetime.now()

        return alerts

    def _evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """评估单个条件"""
        field_name = condition.get('field', '')
        operator = condition.get('operator', '')
        threshold = condition.get('value', 0)

        current_value = data.get(field_name)
        if current_value is None:
            return False

        try:
            if operator == '>':
                return float(current_value) > float(threshold)
            elif operator == '>=':
                return float(current_value) >= float(threshold)
            elif operator == '<':
                return float(current_value) < float(threshold)
            elif operator == '<=':
                return float(current_value) <= float(threshold)
            elif operator == '==':
                return float(current_value) == float(threshold)
            elif operator == '!=':
                return float(current_value) != float(threshold)
            elif operator == 'cross_above':
                prev_value = data.get(f'prev_{field_name}')
                if prev_value is not None:
                    return float(prev_value) <= float(threshold) and float(current_value) > float(threshold)
            elif operator == 'cross_below':
                prev_value = data.get(f'prev_{field_name}')
                if prev_value is not None:
                    return float(prev_value) >= float(threshold) and float(current_value) < float(threshold)
            elif operator == 'change_pct_above':
                prev_value = data.get(f'prev_{field_name}', current_value)
                if prev_value != 0:
                    change = (float(current_value) - float(prev_value)) / abs(float(prev_value)) * 100
                    return change > float(threshold)
            elif operator == 'in_range':
                range_values = condition.get('range', [])
                return float(range_values[0]) <= float(current_value) <= float(range_values[1])
        except (ValueError, TypeError, ZeroDivisionError):
            return False

        return False

    def _create_alert(self, rule: AlertRule, data: Dict[str, Any]) -> Alert:
        """创建预警消息"""
        self._alert_counter += 1
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._alert_counter}"

        # 生成预警标题和消息
        title, message = self._generate_alert_text(rule, data)

        return Alert(
            id=alert_id,
            type=rule.type,
            level=rule.level,
            title=title,
            message=message,
            detail=data,
            stock_code=data.get('stock_code', ''),
            stock_name=data.get('stock_name', ''),
            blogger_id=data.get('blogger_id', ''),
            current_value=data.get(rule.condition.get('field', ''), 0),
            threshold_value=rule.condition.get('value', 0),
            change_pct=data.get('change_pct', 0),
            expire_time=datetime.now() + timedelta(hours=24)
        )

    def _generate_alert_text(self, rule: AlertRule, data: Dict[str, Any]) -> Tuple[str, str]:
        """生成预警文本"""
        stock_name = data.get('stock_name', data.get('stock_code', ''))
        field_name = rule.condition.get('field', '')
        current = data.get(field_name, '')
        threshold = rule.condition.get('value', '')

        templates = {
            AlertType.PRICE_BREAK: (
                f"[价格突破] {stock_name} 突破关键价位",
                f"{stock_name} 当前价格 {current} 突破 {threshold}"
            ),
            AlertType.PRICE_DROP: (
                f"[价格跌破] {stock_name} 跌破关键价位",
                f"{stock_name} 当前价格 {current} 跌破 {threshold}"
            ),
            AlertType.SURGE: (
                f"[涨幅异动] {stock_name} 涨幅异常",
                f"{stock_name} 当前涨幅 {current}%，超过阈值 {threshold}%"
            ),
            AlertType.PLUNGE: (
                f"[跌幅异动] {stock_name} 跌幅异常",
                f"{stock_name} 当前跌幅 {current}%，超过阈值 {threshold}%"
            ),
            AlertType.VOLUME_ANOMALY: (
                f"[量能异动] {stock_name} 成交量异常",
                f"{stock_name} 成交量异常放大，当前量比 {current}"
            ),
            AlertType.BLOGGER_POST: (
                f"[博主动态] {data.get('blogger_name', '')} 发布新内容",
                f"{data.get('blogger_name', '')} 发布了关于 {stock_name} 的内容"
            ),
            AlertType.BLOGGER_ACTION: (
                f"[博主操作] {data.get('blogger_name', '')} {data.get('action', '操作')}",
                f"{data.get('blogger_name', '')} 对 {stock_name} 执行了 {data.get('action', '操作')}"
            ),
            AlertType.VAR_BREACH: (
                f"[VaR预警] 组合风险超限",
                f"组合VaR {current}% 超过限额 {threshold}%"
            ),
            AlertType.DRAWDOWN: (
                f"[回撤预警] {stock_name} 回撤较大",
                f"{stock_name} 当前回撤 {current}%，超过阈值 {threshold}%"
            ),
            AlertType.STOP_LOSS: (
                f"[止损预警] {stock_name} 触及止损",
                f"{stock_name} 当前价格 {current}，止损位 {threshold}"
            ),
            AlertType.STOP_PROFIT: (
                f"[止盈预警] {stock_name} 触及止盈",
                f"{stock_name} 当前价格 {current}，止盈位 {threshold}"
            ),
            AlertType.TREND_CHANGE: (
                f"[趋势变化] {stock_name} 趋势转变",
                f"{stock_name} 趋势从 {data.get('prev_trend', '?')} 变为 {data.get('new_trend', '?')}"
            ),
            AlertType.ML_PREDICTION: (
                f"[ML预测] {stock_name} 预测信号",
                f"ML模型预测 {stock_name} {data.get('predicted_trend', '?')}，置信度 {data.get('confidence', 0):.0%}"
            ),
        }

        return templates.get(rule.type, (
            f"[预警] {rule.name}",
            f"规则 {rule.name} 已触发"
        ))


# ==================== 预警检测器 ====================

class PriceAlertDetector:
    """价格预警检测器"""

    def __init__(self, rule_engine: AlertRuleEngine):
        self.rule_engine = rule_engine

    def check_price_alert(self, stock_code: str, stock_name: str,
                          current_price: float, prev_price: float = None,
                          change_pct: float = None) -> List[Alert]:
        """检查价格预警"""
        data = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'price': current_price,
            'prev_price': prev_price or current_price,
            'change_pct': change_pct or 0,
        }

        if prev_price and prev_price != 0:
            data['change_pct'] = (current_price - prev_price) / prev_price * 100

        return self.rule_engine.evaluate(data)


class VolumeAlertDetector:
    """成交量异动检测器"""

    def __init__(self, rule_engine: AlertRuleEngine,
                 volume_ratio_threshold: float = 3.0):
        self.rule_engine = rule_engine
        self.volume_ratio_threshold = volume_ratio_threshold

    def check_volume_alert(self, stock_code: str, stock_name: str,
                           current_volume: float, avg_volume: float) -> List[Alert]:
        """检查成交量异动"""
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        data = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
        }

        alerts = self.rule_engine.evaluate(data)

        # 额外检查：量比超阈值
        if volume_ratio >= self.volume_ratio_threshold:
            data['field'] = 'volume_ratio'
            data['value'] = volume_ratio
            # 如果规则引擎没有匹配到，直接生成
            if not alerts:
                alerts.append(Alert(
                    id=f"vol_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    type=AlertType.VOLUME_ANOMALY,
                    level=AlertLevel.MEDIUM if volume_ratio < 5 else AlertLevel.HIGH,
                    title=f"[量能异动] {stock_name} 成交量异常",
                    message=f"{stock_name} 量比 {volume_ratio:.1f}，成交量显著放大",
                    stock_code=stock_code,
                    stock_name=stock_name,
                    current_value=volume_ratio,
                    threshold_value=self.volume_ratio_threshold
                ))

        return alerts


class BloggerAlertDetector:
    """博主动态预警检测器"""

    def __init__(self, rule_engine: AlertRuleEngine):
        self.rule_engine = rule_engine

    def check_blogger_post(self, blogger_id: str, blogger_name: str,
                           content: str, stock_names: List[str] = None,
                           action_type: str = None) -> List[Alert]:
        """检查博主动态预警"""
        data = {
            'blogger_id': blogger_id,
            'blogger_name': blogger_name,
            'content': content,
            'stock_names': stock_names or [],
            'action': action_type or '',
        }

        alerts = self.rule_engine.evaluate(data)

        # 如果有操作信号，额外生成操作预警
        if action_type and action_type in ['buy', 'sell']:
            for stock_name in (stock_names or ['']):
                alerts.append(Alert(
                    id=f"blogger_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    type=AlertType.BLOGGER_ACTION,
                    level=AlertLevel.HIGH,
                    title=f"[博主操作] {blogger_name} {'买入' if action_type == 'buy' else '卖出'} {stock_name}",
                    message=f"{blogger_name} 对 {stock_name} 执行了{'买入' if action_type == 'buy' else '卖出'}操作",
                    stock_name=stock_name,
                    blogger_id=blogger_id,
                    detail=data
                ))

        return alerts


class RiskAlertDetector:
    """风险预警检测器"""

    def __init__(self, rule_engine: AlertRuleEngine):
        self.rule_engine = rule_engine

    def check_var_alert(self, current_var: float, var_limit: float,
                        current_cvar: float = 0, cvar_limit: float = 0) -> List[Alert]:
        """检查VaR预警"""
        data = {
            'var_value': current_var,
            'var_limit': var_limit,
            'cvar_value': current_cvar,
            'cvar_limit': cvar_limit,
        }

        alerts = self.rule_engine.evaluate(data)

        # 直接检查
        if current_var > var_limit:
            level = AlertLevel.CRITICAL if current_var > var_limit * 1.5 else AlertLevel.HIGH
            alerts.append(Alert(
                id=f"var_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                type=AlertType.VAR_BREACH,
                level=level,
                title=f"[VaR预警] 组合VaR超限",
                message=f"当前VaR {current_var:.2f}% 超过限额 {var_limit:.2f}%",
                current_value=current_var,
                threshold_value=var_limit
            ))

        if cvar_limit > 0 and current_cvar > cvar_limit:
            alerts.append(Alert(
                id=f"cvar_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                type=AlertType.VAR_BREACH,
                level=AlertLevel.HIGH,
                title=f"[CVaR预警] 组合CVaR超限",
                message=f"当前CVaR {current_cvar:.2f}% 超过限额 {cvar_limit:.2f}%",
                current_value=current_cvar,
                threshold_value=cvar_limit
            ))

        return alerts

    def check_drawdown_alert(self, stock_code: str, stock_name: str,
                             current_drawdown: float,
                             drawdown_limit: float = -10.0) -> List[Alert]:
        """检查回撤预警"""
        data = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'drawdown': current_drawdown,
        }

        alerts = self.rule_engine.evaluate(data)

        if current_drawdown < drawdown_limit:
            level = AlertLevel.CRITICAL if current_drawdown < drawdown_limit * 2 else AlertLevel.HIGH
            alerts.append(Alert(
                id=f"dd_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                type=AlertType.DRAWDOWN,
                level=level,
                title=f"[回撤预警] {stock_name} 回撤较大",
                message=f"{stock_name} 当前回撤 {current_drawdown:.2f}%，超过阈值 {drawdown_limit:.2f}%",
                stock_code=stock_code,
                stock_name=stock_name,
                current_value=current_drawdown,
                threshold_value=drawdown_limit
            ))

        return alerts


# ==================== 通知渠道 ====================

class NotificationChannel(ABC):
    """通知渠道基类"""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class ConsoleNotificationChannel(NotificationChannel):
    """控制台通知"""

    async def send(self, alert: Alert) -> bool:
        level_icons = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.LOW: "🟢",
            AlertLevel.MEDIUM: "🟡",
            AlertLevel.HIGH: "🔴",
            AlertLevel.CRITICAL: "🚨",
        }
        icon = level_icons.get(alert.level, "📌")
        print(f"\n{icon} [{alert.level.value.upper()}] {alert.title}")
        print(f"   {alert.message}")
        print(f"   时间: {alert.trigger_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True

    def get_name(self) -> str:
        return "console"


class WebhookNotificationChannel(NotificationChannel):
    """Webhook通知"""

    def __init__(self, webhook_url: str, headers: Dict = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}

    async def send(self, alert: Alert) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    headers=self.headers,
                    timeout=10
                )
                return response.status_code == 200
        except Exception as e:
            print(f"Webhook notification failed: {e}")
            return False

    def get_name(self) -> str:
        return "webhook"


class WeChatNotificationChannel(NotificationChannel):
    """微信通知（通过Server酱或企业微信）"""

    def __init__(self, sendkey: str = None, corp_id: str = None,
                 corp_secret: str = None, agent_id: str = None):
        self.sendkey = sendkey
        self.corp_id = corp_id
        self.corp_secret = corp_secret
        self.agent_id = agent_id

    async def send(self, alert: Alert) -> bool:
        if self.sendkey:
            return await self._send_via_serverchan(alert)
        elif self.corp_id:
            return await self._send_via_wecom(alert)
        return False

    async def _send_via_serverchan(self, alert: Alert) -> bool:
        """通过Server酱发送"""
        try:
            import httpx
            url = f"https://sctapi.ftqq.com/{self.sendkey}.send"
            data = {
                'title': alert.title,
                'desp': f"**{alert.message}**\n\n时间: {alert.trigger_time.strftime('%Y-%m-%d %H:%M:%S')}"
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, timeout=10)
                return response.status_code == 200
        except Exception as e:
            print(f"ServerChan notification failed: {e}")
            return False

    async def _send_via_wecom(self, alert: Alert) -> bool:
        """通过企业微信发送"""
        try:
            import httpx
            # 获取access_token
            token_url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"
            async with httpx.AsyncClient() as client:
                token_resp = await client.get(token_url, timeout=10)
                access_token = token_resp.json().get('access_token')

                if not access_token:
                    return False

                # 发送消息
                send_url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={access_token}"
                msg = {
                    'touser': '@all',
                    'msgtype': 'text',
                    'agentid': self.agent_id,
                    'text': {'content': f"{alert.title}\n{alert.message}"}
                }
                response = await client.post(send_url, json=msg, timeout=10)
                return response.status_code == 200
        except Exception as e:
            print(f"WeCom notification failed: {e}")
            return False

    def get_name(self) -> str:
        return "wechat"


class EmailNotificationChannel(NotificationChannel):
    """邮件通知"""

    def __init__(self, smtp_host: str, smtp_port: int,
                 sender: str, password: str, recipients: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.recipients = recipients

    async def send(self, alert: Alert) -> bool:
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(alert.message, 'plain', 'utf-8')
            msg['Subject'] = alert.title
            msg['From'] = self.sender
            msg['To'] = ', '.join(self.recipients)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())

            return True
        except Exception as e:
            print(f"Email notification failed: {e}")
            return False

    def get_name(self) -> str:
        return "email"


# ==================== 预警管理器 ====================

class AlertManager:
    """预警管理器 - 核心调度"""

    def __init__(self):
        self.rule_engine = AlertRuleEngine()
        self.channels: List[NotificationChannel] = [ConsoleNotificationChannel()]
        self.alerts: List[Alert] = []
        self._max_alerts = 1000
        self._lock = threading.Lock()
        self._alert_counter = 0

        # 检测器
        self.price_detector = PriceAlertDetector(self.rule_engine)
        self.volume_detector = VolumeAlertDetector(self.rule_engine)
        self.blogger_detector = BloggerAlertDetector(self.rule_engine)
        self.risk_detector = RiskAlertDetector(self.rule_engine)

        # P1: 语义分析预警检测器
        from .semantic_alert_detector import SemanticAlertDetector
        self.semantic_detector = SemanticAlertDetector(self)

        # 回调
        self._callbacks: List[Callable] = []

    def add_channel(self, channel: NotificationChannel):
        """添加通知渠道"""
        self.channels.append(channel)

    def remove_channel(self, channel_name: str):
        """移除通知渠道"""
        self.channels = [c for c in self.channels if c.get_name() != channel_name]

    def add_callback(self, callback: Callable):
        """添加预警回调"""
        self._callbacks.append(callback)

    def add_rule(self, rule: AlertRule):
        """添加预警规则"""
        self.rule_engine.add_rule(rule)

    def remove_rule(self, rule_id: str):
        """移除预警规则"""
        self.rule_engine.remove_rule(rule_id)

    def fire_alert(self, alert: Alert):
        """触发预警"""
        with self._lock:
            self.alerts.append(alert)
            if len(self.alerts) > self._max_alerts:
                self.alerts = self.alerts[-self._max_alerts:]

        # 持久化到数据库
        try:
            asyncio.create_task(self._persist_alert(alert))
        except RuntimeError:
            pass

        # 异步通知
        try:
            asyncio.create_task(self._notify(alert))
        except RuntimeError:
            pass

        # 回调
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")

    async def _persist_alert(self, alert: Alert):
        """将预警持久化到数据库"""
        try:
            from app.database import save_alert
            await save_alert(alert)
        except Exception as e:
            print(f"预警持久化失败: {e}")

    def create_alert(self, alert_type: AlertType, level: AlertLevel,
                      title: str, message: str, detail: Dict[str, Any] = None,
                      stock_code: str = "", stock_name: str = "",
                      blogger_id: str = "",
                      current_value: float = 0, threshold_value: float = 0) -> Alert:
        """创建并触发预警（便捷方法）"""
        self._alert_counter += 1
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._alert_counter}"

        alert = Alert(
            id=alert_id,
            type=alert_type,
            level=level,
            title=title,
            message=message,
            detail=detail or {},
            stock_code=stock_code,
            stock_name=stock_name,
            blogger_id=blogger_id,
            current_value=current_value,
            threshold_value=threshold_value,
            expire_time=datetime.now() + timedelta(hours=24)
        )

        self.fire_alert(alert)
        return alert

    async def _notify(self, alert: Alert):
        """发送通知"""
        for channel in self.channels:
            try:
                success = await channel.send(alert)
                if success:
                    alert.notified_channels.append(channel.get_name())
            except Exception as e:
                print(f"Notification error ({channel.get_name()}): {e}")

    # ==================== 便捷检测方法 ====================

    def check_price(self, stock_code: str, stock_name: str,
                    current_price: float, prev_price: float = None,
                    change_pct: float = None) -> List[Alert]:
        """检查价格预警"""
        alerts = self.price_detector.check_price_alert(
            stock_code, stock_name, current_price, prev_price, change_pct
        )
        for alert in alerts:
            self.fire_alert(alert)
        return alerts

    def check_volume(self, stock_code: str, stock_name: str,
                     current_volume: float, avg_volume: float) -> List[Alert]:
        """检查成交量预警"""
        alerts = self.volume_detector.check_volume_alert(
            stock_code, stock_name, current_volume, avg_volume
        )
        for alert in alerts:
            self.fire_alert(alert)
        return alerts

    def check_blogger(self, blogger_id: str, blogger_name: str,
                      content: str, stock_names: List[str] = None,
                      action_type: str = None) -> List[Alert]:
        """检查博主动态预警"""
        alerts = self.blogger_detector.check_blogger_post(
            blogger_id, blogger_name, content, stock_names, action_type
        )
        for alert in alerts:
            self.fire_alert(alert)
        return alerts

    def check_risk(self, current_var: float, var_limit: float,
                   current_cvar: float = 0, cvar_limit: float = 0) -> List[Alert]:
        """检查风险预警"""
        alerts = self.risk_detector.check_var_alert(
            current_var, var_limit, current_cvar, cvar_limit
        )
        for alert in alerts:
            self.fire_alert(alert)
        return alerts

    def check_drawdown(self, stock_code: str, stock_name: str,
                       current_drawdown: float, drawdown_limit: float = -10.0) -> List[Alert]:
        """检查回撤预警"""
        alerts = self.risk_detector.check_drawdown_alert(
            stock_code, stock_name, current_drawdown, drawdown_limit
        )
        for alert in alerts:
            self.fire_alert(alert)
        return alerts

    # ==================== 查询方法 ====================

    def get_alerts(self, type_filter: AlertType = None,
                   level_filter: AlertLevel = None,
                   status_filter: AlertStatus = None,
                   limit: int = 50) -> List[Alert]:
        """获取预警列表"""
        alerts = self.alerts

        if type_filter:
            alerts = [a for a in alerts if a.type == type_filter]
        if level_filter:
            alerts = [a for a in alerts if a.level == level_filter]
        if status_filter:
            alerts = [a for a in alerts if a.status == status_filter]

        return alerts[-limit:]

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return [a for a in self.alerts if a.status == AlertStatus.ACTIVE]

    def acknowledge_alert(self, alert_id: str):
        """确认预警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                break

    def resolve_alert(self, alert_id: str):
        """解决预警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                break

    def generate_alert_summary(self) -> str:
        """生成预警摘要"""
        active = self.get_active_alerts()

        if not active:
            return "当前无活跃预警"

        # 按级别统计
        level_counts = defaultdict(int)
        for alert in active:
            level_counts[alert.level.value] += 1

        # 按类型统计
        type_counts = defaultdict(int)
        for alert in active:
            type_counts[alert.type.value] += 1

        summary = f"""
# 📢 预警摘要

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 活跃预警数：{len(active)}

## 预警级别分布
"""
        for level in ['critical', 'high', 'medium', 'low', 'info']:
            if level in level_counts:
                summary += f"- {level.upper()}: {level_counts[level]}\n"

        summary += "\n## 预警类型分布\n"
        for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            summary += f"- {type_name}: {count}\n"

        # 最近的HIGH/CRITICAL预警
        critical_alerts = [a for a in active if a.level in (AlertLevel.HIGH, AlertLevel.CRITICAL)]
        if critical_alerts:
            summary += "\n## 重要预警\n"
            for alert in critical_alerts[-5:]:
                summary += f"- [{alert.level.value.upper()}] {alert.title} ({alert.trigger_time.strftime('%H:%M')})\n"

        return summary


# ==================== 默认预警规则 ====================

def create_default_rules() -> List[AlertRule]:
    """创建默认预警规则"""
    return [
        AlertRule(
            id="rule_surge_5",
            name="涨幅超过5%",
            type=AlertType.SURGE,
            condition={"field": "change_pct", "operator": ">", "value": 5},
            level=AlertLevel.MEDIUM,
            cooldown=300
        ),
        AlertRule(
            id="rule_plunge_5",
            name="跌幅超过5%",
            type=AlertType.PLUNGE,
            condition={"field": "change_pct", "operator": "<", "value": -5},
            level=AlertLevel.HIGH,
            cooldown=300
        ),
        AlertRule(
            id="rule_surge_3",
            name="涨幅超过3%",
            type=AlertType.SURGE,
            condition={"field": "change_pct", "operator": ">", "value": 3},
            level=AlertLevel.LOW,
            cooldown=600
        ),
        AlertRule(
            id="rule_plunge_3",
            name="跌幅超过3%",
            type=AlertType.PLUNGE,
            condition={"field": "change_pct", "operator": "<", "value": -3},
            level=AlertLevel.MEDIUM,
            cooldown=600
        ),
        AlertRule(
            id="rule_volume_3x",
            name="成交量放大3倍",
            type=AlertType.VOLUME_ANOMALY,
            condition={"field": "volume_ratio", "operator": ">", "value": 3},
            level=AlertLevel.MEDIUM,
            cooldown=300
        ),
        AlertRule(
            id="rule_var_breach",
            name="VaR超限",
            type=AlertType.VAR_BREACH,
            condition={"field": "var_value", "operator": ">", "value": 5},
            level=AlertLevel.HIGH,
            cooldown=1800
        ),
        AlertRule(
            id="rule_drawdown_10",
            name="回撤超过10%",
            type=AlertType.DRAWDOWN,
            condition={"field": "drawdown", "operator": "<", "value": -10},
            level=AlertLevel.HIGH,
            cooldown=1800
        ),
    ]


# ==================== 测试 ====================

if __name__ == "__main__":
    print("测试实时预警系统...\n")

    # 创建预警管理器
    manager = AlertManager()

    # 添加默认规则
    for rule in create_default_rules():
        manager.add_rule(rule)

    print("1. 测试价格预警...")
    alerts = manager.check_price(
        stock_code="600519",
        stock_name="贵州茅台",
        current_price=1850.0,
        prev_price=1780.0,
        change_pct=3.93
    )
    print(f"   触发 {len(alerts)} 条预警")

    print("\n2. 测试跌幅预警...")
    alerts = manager.check_price(
        stock_code="000858",
        stock_name="五粮液",
        current_price=145.0,
        prev_price=155.0,
        change_pct=-6.45
    )
    print(f"   触发 {len(alerts)} 条预警")

    print("\n3. 测试成交量预警...")
    alerts = manager.check_volume(
        stock_code="600519",
        stock_name="贵州茅台",
        current_volume=5000000,
        avg_volume=1200000
    )
    print(f"   触发 {len(alerts)} 条预警")

    print("\n4. 测试博主动态预警...")
    alerts = manager.check_blogger(
        blogger_id="blogger_001",
        blogger_name="股市老王",
        content="今天加仓了茅台，目标2000",
        stock_names=["贵州茅台"],
        action_type="buy"
    )
    print(f"   触发 {len(alerts)} 条预警")

    print("\n5. 测试风险预警...")
    alerts = manager.check_risk(
        current_var=6.5,
        var_limit=5.0,
        current_cvar=8.2,
        cvar_limit=7.0
    )
    print(f"   触发 {len(alerts)} 条预警")

    print("\n6. 测试回撤预警...")
    alerts = manager.check_drawdown(
        stock_code="000333",
        stock_name="美的集团",
        current_drawdown=-12.5,
        drawdown_limit=-10.0
    )
    print(f"   触发 {len(alerts)} 条预警")

    # 生成预警摘要
    print("\n" + "=" * 60)
    print(manager.generate_alert_summary())
