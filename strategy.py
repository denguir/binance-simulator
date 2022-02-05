import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from order import Order, OrderType, OrderPrice
from ta.trend import MACD


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

    def _update_data(self, kline):
        for symb in kline.keys():
            if symb in self.data.keys():
                self.data[symb] = pd.concat([self.data[symb], kline[symb]])
            else:
                self.data[symb] = kline[symb]

    def _update(self, step, kline, portfolio, balance, unit):
        self.step = step
        self._update_data(kline)
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


class MACDStrategy(TradingStrategy):
    """Follow MACD indicator to buy and sell"""

    def __init__(self):
        super().__init__()
        self.window_fast = 27
        self.window_slow = 35
        self.window_sign = 9
        self.window = 2

    def __str__(self) -> str:
        return f"MACDStrategy ({self.window_fast},{self.window_slow},{self.window_sign})"

    def order(self):
        orders = []

        for symb, df in self.data.items():
            macd = MACD(df['price_close'], 
                            window_fast=self.window_fast, 
                            window_slow=self.window_slow, 
                            window_sign=self.window_sign)
            macd_diff = macd.macd_diff()

            if len(macd_diff) > self.window:
                curr_state = macd_diff.iloc[-1]
                prev_state = macd_diff.iloc[-2]

                if prev_state != np.nan and curr_state != np.nan:
                    if prev_state <= 0 and curr_state > 0:
                        order = Order(side=OrderType.Buy,
                                    price=OrderPrice.Open,
                                    symbol=symb,
                                    quantity=0.1
                                    )
                        orders.append(order)
                        
                    elif prev_state > 0 and curr_state <= 0:
                        order = Order(side=OrderType.Sell,
                                    price=OrderPrice.Open,
                                    symbol=symb,
                                    quantity=0.1
                                    )
                        orders.append(order)
        return orders