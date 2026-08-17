"""
订单执行器
接收选股/预测信号，生成交易订单
"""
import uuid
from datetime import datetime, time
from typing import Dict, List, Optional
from .trading_interface import Order, OrderType, OrderDirection, OrderStatus
from .simulated_trading import SimulatedTrading
from .broker_config import BrokerConfig
from ..market import get_market_client, MarketType, MarketInfo


class OrderExecutor:
    """订单执行器"""

    def __init__(self, trading: SimulatedTrading = None, config: BrokerConfig = None):
        self.config = config or BrokerConfig()
        self.trading = trading or SimulatedTrading(self.config)

    # ── 交易时间判断 ──────────────────────
    def is_trading_time(self, market: str = "A") -> bool:
        """判断是否在交易时间"""
        try:
            market_type = MarketType(market)
            return MarketInfo.is_trading_time(market_type)
        except Exception:
            return False

    # ── 信号下单 ──────────────────────────
    async def execute_signal(self, signal: Dict) -> Optional[Order]:
        """根据选股/预测信号下单
        Args:
            signal: {
                'stock_code': '300274',
                'stock_name': '阳光电源',
                'market': 'A',
                'direction': 'buy',
                'price': 0,  # 0=市价
                'quantity': 100,
                'reason': '预测上涨',
                'source': 'prediction',
            }
        """
        market = signal.get('market', 'A')
        if not self.is_trading_time(market):
            return None  # 非交易时间不下单

        order = Order(
            order_id=f"SIG-{uuid.uuid4().hex[:8]}",
            account_id=0,
            stock_code=signal['stock_code'],
            stock_name=signal.get('stock_name', ''),
            market=market,
            direction=OrderDirection(signal.get('direction', 'buy')),
            order_type=OrderType.MARKET if signal.get('price', 0) <= 0 else OrderType.LIMIT,
            price=signal.get('price', 0),
            quantity=signal.get('quantity', 100),
            source=signal.get('source', 'signal'),
            reason=signal.get('reason', ''),
        )
        return await self.trading.place_order(order)

    # ── 分批建仓 ──────────────────────────
    async def batch_buy(self, stock_code: str, stock_name: str,
                       total_quantity: int, batches: int = 3,
                       market: str = "A", reason: str = "") -> List[Order]:
        """分批建仓
        Args:
            total_quantity: 总数量
            batches: 分几批
        """
        if not self.is_trading_time(market):
            return []

        per_batch = total_quantity // batches
        remainder = total_quantity % batches
        orders = []

        for i in range(batches):
            qty = per_batch + (remainder if i == batches - 1 else 0)
            if qty <= 0:
                continue

            order = Order(
                order_id=f"BAT-{uuid.uuid4().hex[:8]}",
                account_id=0,
                stock_code=stock_code,
                stock_name=stock_name,
                market=market,
                direction=OrderDirection.BUY,
                order_type=OrderType.MARKET,
                quantity=qty,
                source="batch_buy",
                reason=f"分批建仓 {i+1}/{batches}: {reason}",
            )
            result = await self.trading.place_order(order)
            orders.append(result)

        return orders

    # ── 止盈止损执行 ──────────────────────
    async def execute_stop(self, stock_code: str, direction: OrderDirection,
                         quantity: int, market: str = "A",
                         reason: str = "") -> Optional[Order]:
        """执行止盈止损"""
        order = Order(
            order_id=f"STP-{uuid.uuid4().hex[:8]}",
            account_id=0,
            stock_code=stock_code,
            stock_name='',
            market=market,
            direction=direction,
            order_type=OrderType.MARKET,
            quantity=quantity,
            source="stop",
            reason=reason,
        )
        return await self.trading.place_order(order)
