"""
持仓管理器
"""
from datetime import datetime
from typing import Dict, List, Optional
from .trading_interface import Position
from .broker_config import BrokerConfig


class PositionManager:
    """持仓管理"""

    def __init__(self, config: BrokerConfig = None):
        self.config = config or BrokerConfig()
        self._positions: Dict[str, Position] = {}

    def update_position(self, position: Position):
        """更新持仓"""
        self._positions[position.stock_code] = position

    def get_position(self, stock_code: str) -> Optional[Position]:
        return self._positions.get(stock_code)

    def get_all_positions(self) -> List[Position]:
        return list(self._positions.values())

    def remove_position(self, stock_code: str):
        """清仓"""
        self._positions.pop(stock_code, None)

    # ── 盈亏计算 ──────────────────────
    def calculate_profit(self, stock_code: str, current_price: float) -> Optional[Dict]:
        """计算单只持仓盈亏"""
        pos = self._positions.get(stock_code)
        if not pos or pos.quantity <= 0:
            return None

        pos.current_price = current_price
        pos.market_value = pos.quantity * current_price
        pos.profit_loss = (current_price - pos.avg_cost) * pos.quantity
        pos.profit_loss_pct = (
            (current_price - pos.avg_cost) / pos.avg_cost * 100
            if pos.avg_cost > 0 else 0
        )
        pos.last_update_date = datetime.now().strftime("%Y-%m-%d")
        return {
            'stock_code': stock_code,
            'quantity': pos.quantity,
            'avg_cost': pos.avg_cost,
            'current_price': current_price,
            'profit_loss': pos.profit_loss,
            'profit_loss_pct': round(pos.profit_loss_pct, 2),
        }

    def calculate_all_profits(self, prices: Dict[str, float]) -> List[Dict]:
        """批量计算盈亏"""
        results = []
        for code, price in prices.items():
            r = self.calculate_profit(code, price)
            if r:
                results.append(r)
        return results

    # ── 止盈止损 ──────────────────────
    def check_stop_conditions(self, stock_code: str, current_price: float) -> Optional[str]:
        """检查止盈止损"""
        pos = self._positions.get(stock_code)
        if not pos:
            return None

        if pos.stop_profit_price and current_price >= pos.stop_profit_price:
            return f"触发止盈: {stock_code} 现价{current_price} >= 止盈价{pos.stop_profit_price}"

        if pos.stop_loss_price and current_price <= pos.stop_loss_price:
            return f"触发止损: {stock_code} 现价{current_price} <= 止损价{pos.stop_loss_price}"

        return None

    def set_stop_prices(self, stock_code: str,
                       stop_profit: float = None,
                       stop_loss: float = None):
        """设置止盈止损"""
        pos = self._positions.get(stock_code)
        if pos:
            pos.stop_profit_price = stop_profit
            pos.stop_loss_price = stop_loss

    # ── 仓位控制 ──────────────────────
    def check_position_limit(self, stock_code: str, new_quantity: int,
                          max_pct: float = 0.1) -> bool:
        """检查单只股票仓位是否超限
        Args:
            max_pct: 单只股票占总资产最大比例(默认10%)
        """
        pos = self._positions.get(stock_code)
        if not pos:
            return True

        total_value = sum(p.market_value for p in self._positions.values())
        if total_value <= 0:
            return True

        new_value = pos.current_price * new_quantity
        return new_value / total_value <= max_pct

    def get_portfolio_summary(self) -> Dict:
        """组合概览"""
        total_mv = sum(p.market_value for p in self._positions.values())
        total_cost = sum(p.avg_cost * p.quantity for p in self._positions.values())
        total_pl = total_mv - total_cost

        return {
            'position_count': len(self._positions),
            'total_market_value': total_mv,
            'total_cost': total_cost,
            'total_profit_loss': total_pl,
            'total_profit_pct': total_pl / total_cost * 100 if total_cost > 0 else 0,
        }
