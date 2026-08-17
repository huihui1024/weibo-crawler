"""
模拟交易实现
基于实时行情模拟订单撮合，支持滑点、手续费
自动止损止盈、最长持仓强制离场、风险预警子状态
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

from .trading_interface import (
    TradingInterfaceBase, Order, Position, AccountBalance,
    OrderType, OrderDirection, OrderStatus,
)
from .broker_config import BrokerConfig
from ..market import get_market_client, MarketType


class PositionRiskStatus(Enum):
    """持仓风险预警子状态"""
    NORMAL = 'normal'           # 正常
    APPROACHING_STOP = 'approaching_stop'  # 接近止损
    APPROACHING_EXIT = 'approaching_exit'  # 接近最长持仓
    STOP_LOSS_TRIGGERED = 'stop_loss_triggered'  # 止损触发
    TAKE_PROFIT_TRIGGERED = 'take_profit_triggered'  # 止盈触发
    FORCE_EXIT = 'force_exit'   # 强制离场


class SimulatedTrading(TradingInterfaceBase):
    """模拟交易 — 支持自动止损止盈和持仓风险预警"""

    # 自动止损止盈参数
    DEFAULT_STOP_LOSS_PCT = -5.0    # 默认止损-5%
    DEFAULT_TAKE_PROFIT_PCT = 10.0  # 默认止盈+10%
    DEFAULT_MAX_HOLD_DAYS = 20      # 默认最长持仓天数

    def __init__(self, config: BrokerConfig = None):
        self.config = config or BrokerConfig(name="模拟", mode="simulated")
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._balance = AccountBalance(account_id=0)
        self._account_id = 0
        # 持仓风险状态追踪
        self._position_risk_status: Dict[str, Dict] = {}
        # 自定义止损止盈参数(按股票)
        self._custom_stops: Dict[str, Dict] = {}

    def set_stop_params(self, stock_code: str, stop_loss_pct: float = None,
                        take_profit_pct: float = None, max_hold_days: int = None):
        """为单只股票设置自定义止损止盈参数"""
        self._custom_stops[stock_code] = {
            'stop_loss_pct': stop_loss_pct or self.DEFAULT_STOP_LOSS_PCT,
            'take_profit_pct': take_profit_pct or self.DEFAULT_TAKE_PROFIT_PCT,
            'max_hold_days': max_hold_days or self.DEFAULT_MAX_HOLD_DAYS,
        }

    def get_position_risk_status(self, stock_code: str) -> Dict:
        """获取持仓风险预警子状态"""
        return self._position_risk_status.get(stock_code, {
            'status': PositionRiskStatus.NORMAL.value,
            'message': '',
        })

    async def check_auto_stop_loss_profit(self) -> List[Dict]:
        """检查所有持仓的止损止盈条件，返回需要触发的动作

        Returns:
            List of dicts with code, action, reason
        """
        actions = []
        for key, pos in self._positions.items():
            if pos.quantity <= 0:
                continue

            # 获取止损止盈参数
            params = self._custom_stops.get(key, {
                'stop_loss_pct': self.DEFAULT_STOP_LOSS_PCT,
                'take_profit_pct': self.DEFAULT_TAKE_PROFIT_PCT,
                'max_hold_days': self.DEFAULT_MAX_HOLD_DAYS,
            })

            # 更新当前价格
            try:
                client = get_market_client()
                quote = client.get_stock_price(pos.stock_code)
                current_price = quote.get('price', pos.current_price)
                pos.current_price = current_price
            except Exception:
                current_price = pos.current_price

            # 计算盈亏比例
            if pos.avg_cost > 0:
                pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost * 100
            else:
                pnl_pct = 0

            # 更新持仓盈亏
            pos.profit_loss_pct = pnl_pct
            pos.market_value = pos.quantity * current_price
            pos.profit_loss = (current_price - pos.avg_cost) * pos.quantity

            risk_info = {
                'code': pos.stock_code,
                'name': pos.stock_name,
                'pnl_pct': round(pnl_pct, 2),
                'current_price': current_price,
                'avg_cost': pos.avg_cost,
            }

            # 止损检查
            if pnl_pct <= params['stop_loss_pct']:
                risk_info['action'] = 'sell'
                risk_info['reason'] = f"止损触发: {pnl_pct:+.2f}% <= {params['stop_loss_pct']}%"
                risk_info['urgency'] = 'immediate'
                self._position_risk_status[key] = {
                    'status': PositionRiskStatus.STOP_LOSS_TRIGGERED.value,
                    'message': risk_info['reason'],
                }
                actions.append(risk_info)
                continue

            # 止盈检查
            if pnl_pct >= params['take_profit_pct']:
                risk_info['action'] = 'sell'
                risk_info['reason'] = f"止盈触发: {pnl_pct:+.2f}% >= +{params['take_profit_pct']}%"
                risk_info['urgency'] = 'high'
                self._position_risk_status[key] = {
                    'status': PositionRiskStatus.TAKE_PROFIT_TRIGGERED.value,
                    'message': risk_info['reason'],
                }
                actions.append(risk_info)
                continue

            # 最长持仓检查
            if pos.first_buy_date:
                try:
                    buy_date = datetime.strptime(pos.first_buy_date, "%Y-%m-%d")
                    hold_days = (datetime.now() - buy_date).days
                    risk_info['hold_days'] = hold_days

                    if hold_days >= params['max_hold_days']:
                        risk_info['action'] = 'sell'
                        risk_info['reason'] = f"持仓{hold_days}天超限{params['max_hold_days']}天，强制离场"
                        risk_info['urgency'] = 'high'
                        self._position_risk_status[key] = {
                            'status': PositionRiskStatus.FORCE_EXIT.value,
                            'message': risk_info['reason'],
                        }
                        actions.append(risk_info)
                        continue
                    elif hold_days >= params['max_hold_days'] - 3:
                        self._position_risk_status[key] = {
                            'status': PositionRiskStatus.APPROACHING_EXIT.value,
                            'message': f"持仓{hold_days}天，接近{params['max_hold_days']}天限制",
                        }
                except ValueError:
                    pass

            # 接近止损预警
            if pnl_pct <= params['stop_loss_pct'] * 0.6:
                self._position_risk_status[key] = {
                    'status': PositionRiskStatus.APPROACHING_STOP.value,
                    'message': f"接近止损: {pnl_pct:+.2f}% (止损线{params['stop_loss_pct']}%)",
                }
            else:
                self._position_risk_status[key] = {
                    'status': PositionRiskStatus.NORMAL.value,
                    'message': '',
                }

        return actions

    def set_account_id(self, account_id: int):
        self._account_id = account_id
        self._balance.account_id = account_id

    # ── 下单 ─────────────────────────────────
    async def place_order(self, order: Order) -> Order:
        """模拟下单并撮合"""
        order.account_id = self._account_id
        order.order_id = order.order_id or f"SIM-{uuid.uuid4().hex[:8]}"
        order.status = OrderStatus.SUBMITTED
        order.created_at = datetime.now()
        order.updated_at = datetime.now()

        # 获取实时价格
        market_type = MarketType.A_SHARE
        if order.market == 'HK':
            market_type = MarketType.HK
        elif order.market == 'US':
            market_type = MarketType.US

        try:
            client = get_market_client(market_type)
            quote = client.get_stock_price(order.stock_code)
            market_price = quote.get('price', 0)
        except Exception:
            market_price = order.price or 0

        # 模拟撮合
        filled_price = self._simulate_fill(order, market_price)
        if filled_price > 0:
            order.filled_price = filled_price
            order.filled_quantity = order.quantity
            order.filled_amount = filled_price * order.quantity
            order.commission = self._calc_commission(order.filled_amount)
            order.status = OrderStatus.FILLED

            # 更新持仓和资金
            await self._update_position(order)
            await self._update_balance(order)
        else:
            order.status = OrderStatus.REJECTED

        self._orders[order.order_id] = order
        return order

    # ── 撤单 ─────────────────────────────────
    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        return True

    # ── 持仓 ─────────────────────────────────
    async def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    # ── 资金 ─────────────────────────────────
    async def get_balance(self) -> AccountBalance:
        return self._balance

    # ── 订单查询 ─────────────────────────────
    async def get_orders(self, status: OrderStatus = None) -> List[Order]:
        if status:
            return [o for o in self._orders.values() if o.status == status]
        return list(self._orders.values())

    async def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    # ── 内部方法 ─────────────────────────────
    def _simulate_fill(self, order: Order, market_price: float) -> float:
        """模拟撮合"""
        if market_price <= 0:
            return 0

        slippage = self.config.slippage
        if order.direction == OrderDirection.BUY:
            # 买入滑点向上
            filled = market_price * (1 + slippage)
            if order.order_type == OrderType.LIMIT and order.price > 0:
                if order.price < market_price:
                    return 0  # 限价买不到
                filled = min(order.price, filled)
        else:
            # 卖出滑点向下
            filled = market_price * (1 - slippage)
            if order.order_type == OrderType.LIMIT and order.price > 0:
                if order.price > market_price:
                    return 0  # 限价卖不出
                filled = max(order.price, filled)

        return round(filled, 3)

    def _calc_commission(self, amount: float) -> float:
        """计算手续费"""
        rate = self.config.commission_rate
        comm = amount * rate
        return max(comm, 5.0)  # 最低5元

    async def _update_position(self, order: Order):
        """更新持仓"""
        key = f"{order.stock_code}"
        pos = self._positions.get(key)

        if order.direction == OrderDirection.BUY:
            if pos is None:
                pos = Position(
                    account_id=self._account_id,
                    stock_code=order.stock_code,
                    stock_name=order.stock_name,
                    market=order.market,
                    quantity=order.filled_quantity,
                    available_quantity=0,
                    avg_cost=order.filled_price,
                    current_price=order.filled_price,
                    market_value=order.filled_amount,
                    first_buy_date=datetime.now().strftime("%Y-%m-%d"),
                )
            else:
                total_qty = pos.quantity + order.filled_quantity
                total_cost = pos.avg_cost * pos.quantity + order.filled_price * order.filled_quantity
                pos.avg_cost = total_cost / total_qty if total_qty > 0 else 0
                pos.quantity = total_qty
                pos.current_price = order.filled_price
                pos.market_value = pos.quantity * order.filled_price
        else:
            if pos and pos.quantity >= order.filled_quantity:
                pos.quantity -= order.filled_quantity
                pos.market_value = pos.quantity * pos.current_price
                pos.profit_loss = (pos.current_price - pos.avg_cost) * pos.quantity
                pos.profit_loss_pct = (
                    (pos.current_price - pos.avg_cost) / pos.avg_cost * 100
                    if pos.avg_cost > 0 else 0
                )
                if pos.quantity <= 0:
                    del self._positions[key]

        self._positions[key] = pos

    async def _update_balance(self, order: Order):
        """更新资金"""
        if order.direction == OrderDirection.BUY:
            self._balance.available_capital -= (order.filled_amount + order.commission)
        else:
            self._balance.available_capital += (order.filled_amount - order.commission)

        self._balance.market_value = sum(p.market_value for p in self._positions.values())
        self._balance.total_assets = (
            self._balance.available_capital + self._balance.market_value
        )
        self._balance.total_profit = (
            self._balance.total_assets - self._balance.initial_capital
        )
        self._balance.total_profit_pct = (
            self._balance.total_profit / self._balance.initial_capital * 100
            if self._balance.initial_capital > 0 else 0
        )
