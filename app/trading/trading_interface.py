"""
交易接口基类
定义统一的交易接口，支持模拟交易和实盘交易
"""
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"          # 市价单
    LIMIT = "limit"            # 限价单
    STOP = "stop"              # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class OrderDirection(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL_FILLED = "partial_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """交易订单"""
    order_id: str
    account_id: int
    stock_code: str
    stock_name: str
    market: str = "A"
    direction: OrderDirection = OrderDirection.BUY
    order_type: OrderType = OrderType.LIMIT
    price: float = 0
    quantity: int = 0
    filled_quantity: int = 0
    filled_price: float = 0
    filled_amount: float = 0
    commission: float = 0
    status: OrderStatus = OrderStatus.PENDING
    source: str = "manual"
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'account_id': self.account_id,
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'market': self.market,
            'direction': self.direction.value,
            'order_type': self.order_type.value,
            'price': self.price,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'filled_amount': self.filled_amount,
            'commission': self.commission,
            'status': self.status.value,
            'source': self.source,
            'reason': self.reason,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


@dataclass
class Position:
    """持仓"""
    account_id: int
    stock_code: str
    stock_name: str
    market: str = "A"
    quantity: int = 0
    available_quantity: int = 0
    avg_cost: float = 0
    current_price: float = 0
    market_value: float = 0
    profit_loss: float = 0
    profit_loss_pct: float = 0
    stop_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    first_buy_date: Optional[str] = None
    last_update_date: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'account_id': self.account_id,
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'market': self.market,
            'quantity': self.quantity,
            'available_quantity': self.available_quantity,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'profit_loss': self.profit_loss,
            'profit_loss_pct': self.profit_loss_pct,
            'stop_profit_price': self.stop_profit_price,
            'stop_loss_price': self.stop_loss_price,
        }


@dataclass
class AccountBalance:
    """账户资金"""
    account_id: int
    initial_capital: float = 1000000
    available_capital: float = 1000000
    frozen_capital: float = 0
    total_assets: float = 1000000
    market_value: float = 0
    total_profit: float = 0
    total_profit_pct: float = 0

    def to_dict(self) -> Dict:
        return {
            'account_id': self.account_id,
            'initial_capital': self.initial_capital,
            'available_capital': self.available_capital,
            'frozen_capital': self.frozen_capital,
            'total_assets': self.total_assets,
            'market_value': self.market_value,
            'total_profit': self.total_profit,
            'total_profit_pct': self.total_profit_pct,
        }


class TradingInterfaceBase(ABC):
    """交易接口基类"""

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """下单"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """获取持仓"""
        pass

    @abstractmethod
    async def get_balance(self) -> AccountBalance:
        """获取资金"""
        pass

    @abstractmethod
    async def get_orders(self, status: OrderStatus = None) -> List[Order]:
        """获取订单列表"""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """获取单个订单"""
        pass