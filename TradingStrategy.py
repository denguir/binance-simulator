from abc import ABC, abstractmethod

class TradingStrategy(ABC):
    """Trading strategy template class. A trading strategy must contain:
        - a buy method, that is executed at each time step of the simuation.
        - a sell method, that is executed at each time step of the simuation.
        
        IMPORTANT:
        At each time step, the sell method is executed before the buy method.
    """

    @abstractmethod
    def buy(self, data, portfolio, balance):
        """inputs:
            - data: {symbol: dataframe of OHCLV for interval [0:t] of the simulation}.
            - portfolio: {asset: amount}. Portfolio state at time t.
            - balance: portfolio value in the unit chosen in the binance simulator.
           outputs:
            - {symbol: amount}. The symbol to buy with the quantity for next time step.
        """
        raise NotImplementedError("A buy strategy is needed.")

    @abstractmethod
    def sell(self, data, portfolio, balance):
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
    
    def buy(self, data, portfolio, balance):
        return {'BTCUSDT': 0.1}

    def sell(self, data, portfolio, balance):
        return {}

 