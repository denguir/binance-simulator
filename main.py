import time
from BinanceSimulator import BinanceSimulator
from TradingStrategy import HoldStrategy
from datetime import datetime, timedelta, timezone


if __name__ == '__main__':
    bs = BinanceSimulator(
        unit='USDT',
        balance=10000
    )

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'QTUMUSDT']

    # load data
    t0 = time.time()
    bs.load_data(date_from=datetime(2021, 1, 1), 
                 date_to=datetime(2021, 11, 30),
                 symbols=symbols,
                 n_jobs=4)
    t1 = time.time()

    # init strategy
    strategy = HoldStrategy()
    print(f'Time taken {t1 - t0}')
    bs.run(None, 1, 100, 1000)
