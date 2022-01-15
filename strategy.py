from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from order import Order, OrderType, OrderPrice


@dataclass
class TradingStrategy(ABC):
    """Trading strategy template class. A trading strategy must contain:
        - a buy method, that is executed at each time step of the simuation.
        - a sell method, that is executed at each time step of the simuation.
    """

    data: dict = field(default_factory=dict)
    portfolio: dict = field(default_factory=dict)
    balance: float = 0.0
    unit: str = ''

    def _update(self, data, portfolio, balance, unit):
        self.data.update(data)
        self.portfolio.update(portfolio)
        self.balance = balance
        self.unit = unit

    @abstractmethod
    def order(self):
        raise NotImplementedError("A list of orders must be provided")

    @abstractmethod
    def buy(self):
        """inputs:
            - data: {symbol: dataframe of OHCLV for interval [0:t] of the simulation}.
            - portfolio: {asset: amount}. Portfolio state at time t.
            - balance: portfolio value in the unit chosen in the binance simulator.
           outputs:
            - {symbol: amount}. The symbol to buy with the quantity for next time step.
        """
        raise NotImplementedError("A buy strategy is needed.")

    @abstractmethod
    def sell(self):
        """inputs:
            - data: {symbol: dataframe of OHCLV for interval [0:t] of the simulation}.
            - portfolio: {asset: amount}. Portfolio state at time t.
            - balance: portfolio value in the unit chosen in the binance simulator.
           outputs:
            - {symbol: amount}. The symbol to sell with the quantity for next time step.
        """
        raise NotImplementedError("A sell strategy is needed.")


class HoldStrategy(TradingStrategy):
    """Hold strategy that consists of buying 0.1 BTC at each time step,
       until USDT balance is empty. It never sells the BTC.
    """

    def order(self):
        return [Order(side=OrderType.Buy,
                      price=OrderPrice.Open,
                      symbol='BTCUSDT',
                      quantity=1
                      )]

    def buy(self):
        return {'BTCUSDT': 0.1}

    def sell(self):
        return {}
