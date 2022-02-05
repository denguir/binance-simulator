from enum import Enum
from dataclasses import dataclass


class OrderType(Enum):
    Sell = 'sell'
    Buy = 'buy'


class OrderPrice(Enum):
    Open = 'open'
    High = 'high'
    Close = 'close'
    Low = 'low'
    Mean = 'mean'


class OrderFee(Enum):
    # go to Binance fees web page
    # https://www.binance.com/fr/fee/schedule
    VIP0 = 0.000750


@dataclass
class Order:
    side: OrderType
    symbol: str
    quantity: float
    price: OrderPrice = OrderPrice.Open
    fee: float = 0.000750


