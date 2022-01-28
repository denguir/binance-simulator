from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from order import Order, OrderType, OrderPrice


@dataclass
class TradingStrategy(ABC):
    """Trading strategy template class. A trading strategy must contain:
        - a buy method, that is executed at each time step of the simuation.
        - a sell method, that is executed at each time step of the simuation.
    """
    step: int = 0
    data: dict = field(default_factory=dict)
    portfolio: dict = field(default_factory=dict)
    balance: float = 0.0
    unit: str = ''

    def _update(self, step, data, portfolio, balance, unit):
        self.step = step
        self.data.update(data)
        self.portfolio.update(portfolio)
        self.balance = balance
        self.unit = unit

    @abstractmethod
    def order(self):
        raise NotImplementedError("A list of orders must be provided")


class HoldStrategy(TradingStrategy):
    """Hold strategy that consists of buying 0.1 BTC at each time step,
       until USDT balance is empty. It never sells the BTC.
    """
    
    def __str__(self) -> str:
        return "HoldStrategy"


    def order(self):
        return [Order(side=OrderType.Buy,
                      price=OrderPrice.Open,
                      symbol='BTCUSDT',
                      quantity=0.1
                      )]


class BuySellStrategy(TradingStrategy):
    """Buy and sell iteratively the same amount of BTC
    """

    def __str__(self) -> str:
        return "BuySellStrategy"

    def order(self):
        if self.step % 2 == 0:
            side = OrderType.Buy
            qty = 1
        else:
            side = OrderType.Sell
            qty = 0.5

        return [Order(side=side,
                      price=OrderPrice.Open,
                      symbol='BTCUSDT',
                      quantity=qty
                      )]