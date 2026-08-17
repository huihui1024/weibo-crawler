"""
券商配置
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BrokerConfig:
    """券商/模拟交易配置"""
    name: str = "模拟"
    broker: str = ""
    market: str = "A"
    mode: str = "simulated"  # simulated / live
    api_key: str = ""
    api_secret: str = ""
    commission_rate: float = 0.0003
    slippage: float = 0.001
    initial_capital: float = 1000000
    enabled: bool = True

    # ── 预置券商配置 ──────────────────────
    @classmethod
    def futu_hk(cls) -> "BrokerConfig":
        """富途港股"""
        return cls(
            name="富途港股",
            broker="futu",
            market="HK",
            mode="simulated",
            commission_rate=0.0003,
            slippage=0.002,
        )

    @classmethod
    def tiger_us(cls) -> "BrokerConfig":
        """老虎美股"""
        return cls(
            name="老虎美股",
            broker="tiger",
            market="US",
            mode="simulated",
            commission_rate=0.005,  # 美股按股收费，此处简化
            slippage=0.002,
        )

    @classmethod
    def xueqiu_sim(cls) -> "BrokerConfig":
        """雪球模拟"""
        return cls(
            name="雪球模拟",
            broker="xueqiu",
            market="A",
            mode="simulated",
            commission_rate=0.0003,
            slippage=0.001,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'broker': self.broker,
            'market': self.market,
            'mode': self.mode,
            'commission_rate': self.commission_rate,
            'slippage': self.slippage,
            'initial_capital': self.initial_capital,
            'enabled': self.enabled,
        }
