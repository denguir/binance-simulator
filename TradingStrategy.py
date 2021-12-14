

class TradingStrategy:

    def buy(self, data, portfolio, balance):
        raise NotImplementedError("A buy strategy is needed.")

    def sell(self, data, portfolio, balance):
        raise NotImplementedError("A sell strategy is needed.")


class BTCHoldStrategy(TradingStrategy):
    
    def buy(self, data, portfolio, balance):
        return {'BTCUSDT': 1} 

    def sell(self, data, portfolio, balance):
        return {}

 