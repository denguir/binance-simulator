

class TradingStrategy:

    def buy(self, data, portfolio):
        raise NotImplementedError("A buy strategy is needed.")

    def sell(self, data, portfolio):
        raise NotImplementedError("A sell strategy is needed.")


class HoldStrategy(TradingStrategy):
    
    def buy(self, data, portfolio):
        return {'BTCUSDT': 1} 

    def sell(self, data, portfolio):
        return {}
