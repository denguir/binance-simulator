import time
import dash
from simulator import BinanceSimulator
from strategy import BuySellStrategy, HoldStrategy
from datetime import datetime


app = dash.Dash(__name__)

bs = BinanceSimulator(
    unit='USDT',
    balance=10000
)

symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'QTUMUSDT']

# load data
t0 = time.time()
bs.load_data_from_api(date_from=datetime(2020, 1, 1),
                        date_to=datetime(2021, 11, 30),
                        symbols=symbols)
t1 = time.time()
print(f'time to load: {t1 - t0} seconds.')

# init strategy
strategy = HoldStrategy()
# run strategy
bs.run(strategy)
# render performance
bs.render(app)


if __name__ == '__main__':
    app.run_server(debug=False)