import time
import pandas as pd
import numpy as np
from TradingStrategy import TradingStrategy
from datetime import datetime, timedelta, timezone
from binance.client import Client
from joblib import Parallel, delayed, parallel_backend


class BinanceSimulator:
    """Simulator for Binance trading platform. Useful to apply back-testing
       on your trading strategies.
    """

    def __init__(self, unit: str='USDT', balance: float=10000) -> None:
        self.unit = unit
        self.balance = balance

        self._client = Client(None, None)
        self.symbols_info = self.get_symbols_info()
        self._step = 0

        self._data = {}
        self.trades = []

    @property
    def data(self):
        return {symb : df[:self._step] for symb, df in self._data.items()}

    @staticmethod
    def to_timestamp_ms(dt):
        ts = int(1000 * datetime.timestamp(dt))
        return ts

    def get_symbols_info(self):
        info = self._client.get_exchange_info()
        return pd.DataFrame(info['symbols'])
    
    def load_symbol_data(self, symbol:str, date_from:datetime, date_to:datetime, resolution: str='1d'):
        # add logic to read from or query API
        klines = self._client.get_historical_klines(symbol=symbol, 
                                                    interval=resolution,
                                                    start_str=BinanceSimulator.to_timestamp_ms(date_from),
                                                    end_str=BinanceSimulator.to_timestamp_ms(date_to))
        self._data[symbol] = pd.DataFrame(data=klines,
                                         columns=[
                                            'ts_open', 'price_open', 'price_high', 'price_low', 
                                            'price_close', 'volume', 'ts_close', 'quote_asset_volume', 
                                            'number_of_trades', 'taker_buy_base_asset_volume', 
                                            'taker_buy_quote_asset_volume', 'ignore']
                                        )

    def load_data(self, date_from:datetime, date_to:datetime, symbols: list='all', resolution: str='1d', n_jobs=4):
        # make sure to deal with the case where we dont have the same amount of data for the same time window
        # _step should be an index that is the same for every pair of symbols
        with parallel_backend('threading', n_jobs=n_jobs):
            if type(symbols) is list:
                Parallel()(delayed(self.load_symbol_data)(symb, date_from, date_to, resolution) for symb in symbols)
            else:
                Parallel()(delayed(self.load_symbol_data)(symb, date_from, date_to, resolution) for symb in self.symbols_info['symbol'])

    def tick(self, strategy:TradingStrategy, step:int=1):
        self._step += step
        if strategy:
            trades = strategy(self.data)
        else:
            print(f'Step {self._step}')

    def run(self, strategy:TradingStrategy, step:int=1, offset:int=100):
        i = 0
        while i < offset:
            self.tick(None, step)
            i += 1
        while i < self.max_step:
            self.tick(strategy, step)
            i +=1

if __name__ == '__main__':
    bs = BinanceSimulator(
        unit='USDT',
        balance=10000
    )

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'QTUMUSDT']
    t0 = time.time()
    bs.load_data(date_from=datetime(2021, 1, 1), 
                 date_to=datetime(2021, 11, 30),
                 symbols=symbols,
                 n_jobs=4)
    t1 = time.time()
    print(f'Time taken {t1 - t0}')
    


    for symb in symbols:
        print(symb, len(bs.data[symb]))


        
            




    