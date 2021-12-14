import time
from BinanceSimulator import BinanceSimulator
from TradingStrategy import BTCHoldStrategy
from datetime import datetime


if __name__ == '__main__':
    bs = BinanceSimulator(
        unit='USDT',
        balance=10000
    )

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'QTUMUSDT']

    # load data
    t0 = time.time()
    bs.load_data(date_from=datetime(2020, 1, 1), 
                 date_to=datetime(2021, 11, 30),
                 symbols=symbols,
                 n_jobs=4)
    t1 = time.time()
    print(f'time to load: {t1 - t0} seconds.')

    # init strategy
    strategy = BTCHoldStrategy()
    bs.run(strategy)
    bs.render()
    