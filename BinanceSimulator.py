import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from TradingStrategy import TradingStrategy
from datetime import datetime
from binance.client import Client
from joblib import Parallel, delayed, parallel_backend


logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s :: %(levelname)s :: %(name)s :: %(funcName)s :: %(message)s')
logger = logging.getLogger(__name__)


class BinanceSimulator:
    """Simulator for Binance trading platform. Useful to apply back-testing
       on your trading strategies.
    """

    def __init__(self, unit: str='USDT', balance: float=10000) -> None:
        self._quotes = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']
        self.unit = unit
        self.balance = balance
        self.portfolio = {unit : balance}

        self._client = Client(None, None)
        self.symbols_info = self.get_symbols_info()
        self._step = 0
        self._max_step = 0

        self._data = {}
        self.portfolio_hist = {self._step : self.portfolio}
        self.balance_hist = pd.DataFrame(data=[[self._step, self.balance]],
                                         columns=['step', 'balance'])
        self.trade_hist = pd.DataFrame(
            columns=['step', 'ts', 'side', 'symbol', 'quantity', 'price'])

    @property
    def data(self):
        return {symb : df[:self._step] for symb, df in self._data.items()}

    @property
    def fee(self):
        # VIP 0 fees (taker/maker) with BNB discount allowed
        return 0.000750

    @staticmethod
    def to_timestamp_ms(dt):
        ts = int(1000 * datetime.timestamp(dt))
        return ts

    @staticmethod
    def from_timestamp_ms(ts):
        dt = datetime.fromtimestamp(ts // 1000)
        return dt

    def get_symbols_info(self):
        info = self._client.get_exchange_info()
        return pd.DataFrame(info['symbols'])

    def split_symbol(self, symbol):
        symb_info = self.symbols_info[self.symbols_info['symbol'] == symbol].to_dict('list')
        base_asset = symb_info['baseAsset'][0]
        quote_asset = symb_info['quoteAsset'][0]
        return base_asset, quote_asset

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
                                        ).apply(pd.to_numeric)
        self._max_step = max(self._max_step, len(klines))

    def load_data(self, date_from:datetime, date_to:datetime, symbols: list='all', resolution: str='1d', n_jobs=4):
        # make sure to deal with the case where we dont have the same amount of data for the same time window
        # _step should be an index that is the same for every pair of symbols
        with parallel_backend('threading', n_jobs=n_jobs):
            if type(symbols) is list:
                Parallel()(delayed(self.load_symbol_data)(symb, date_from, date_to, resolution) for symb in symbols)
            else:
                Parallel()(delayed(self.load_symbol_data)(symb, date_from, date_to, resolution) for symb in self.symbols_info['symbol'])

    def is_tradable(self, symbol):
        symb_info = self.symbols_info[self.symbols_info['symbol'] == symbol].to_dict('list')
        tradable = symb_info['isSpotTradingAllowed']
        if tradable:
            if tradable[0]:
                return True
        return False

    def get_min_trading_qty(self, symbol):
        symb_info = self.symbols_info[self.symbols_info['symbol'] == symbol].to_dict('list')
        min_qty = float(symb_info['filters'][0][2]['minQty'])
        return min_qty

    def get_max_trading_qty(self, symbol):
        symb_info = self.symbols_info[self.symbols_info['symbol'] == symbol].to_dict('list')
        max_qty = float(symb_info['filters'][0][2]['maxQty'])
        return max_qty

    def get_last_kline(self, symbol):
        df = self.data[symbol]
        return df.iloc[self._step - 1]

    def get_price(self, base, quote):
        '''Return close price of base w.r.t quote'''
        assert quote in self._quotes, f"{quote} not supported as an exchange quote."
        if base == quote:
            price = 1.0
        else:
            symbol = base + quote
            if self.is_tradable(symbol):
                res = self.get_last_kline(symbol)
                price = res['price_close']
            else:
                symbol = quote + base
                if self.is_tradable(symbol):
                    res = self.get_last_kline(symbol)
                    price = 1.0 / res['price_close']
                else:
                    alt_symbols = [base + alt_quote for alt_quote in self._quotes]
                    for alt_symb in alt_symbols:
                        if self.is_tradable(alt_symb):
                            _, alt_quote = self.split_symbol(alt_symb)
                            alt_kline = self.get_last_kline(alt_symb)
                            alt_price = alt_kline['price_close']
                            quote_price = self.get_price(alt_quote, quote)
                            price = alt_price * quote_price
                            break
                    print(f"base {base} seems to have no exchange with one of the supported quotes {self._quotes}.")
                    price = 0.0
        return price

    def update_balance(self):
        balance = 0.0
        for asset, qty in self.portfolio.items():
            price = self.get_price(asset, self.unit)
            balance += (price * qty)
        self.balance = balance

    def create_buy_order(self, symbol, quantity, handle='max'):
        base_asset, quote_asset = self.split_symbol(symbol)
        last_kline = self.get_last_kline(symbol)
        if self.portfolio.get(quote_asset, 0) >= quantity * last_kline['price_close']:
            self.portfolio[base_asset] = self.portfolio.get(base_asset, 0) + ((1 - self.fee) * quantity)
            self.portfolio[quote_asset] -= quantity * last_kline['price_close']

            self.trade_hist.loc[len(self.trade_hist)] = \
                [self._step, last_kline['ts_close'], 'buy', symbol, quantity, last_kline['price_close']]

        else:
            logger.warning(f"step {self._step}: Not enough liquidity to buy {quantity} of {symbol}.")
            if handle == 'max':
                max_qty = self.portfolio.get(quote_asset, 0) / last_kline['price_close']
                if max_qty >= self.get_min_trading_qty(symbol):
                    self.portfolio[base_asset] = self.portfolio.get(base_asset, 0) + ((1 - self.fee) * max_qty)
                    self.portfolio[quote_asset] = 0
                
                    self.trade_hist.loc[len(self.trade_hist)] = \
                        [self._step, last_kline['ts_close'], 'buy', symbol, max_qty, last_kline['price_close']]

                    logger.warning(f"step {self._step}: Bought {max_qty} of {symbol}.")
                else:
                    logger.warning(f"step {self._step}: Ignoring buy order.")
            elif handle == 'ignore':
                logger.warning(f"step {self._step}: Ignoring buy order.")

    def create_sell_order(self, symbol, quantity, handle='max'):
        base_asset, quote_asset = self.split_symbol(symbol)
        last_kline = self.get_last_kline(symbol)
        if self.portfolio.get(base_asset, 0) >= quantity:
            self.portfolio[base_asset] -= quantity
            self.portfolio[quote_asset] = self.portfolio.get(quote_asset, 0) + \
                                             ((1 - self.fee) * quantity) * last_kline['price_close']

            self.trade_hist.loc[len(self.trade_hist)] = \
                [self._step, last_kline['ts_close'], 'sell', symbol, quantity, last_kline['price_close']]

        else:
            logger.warning(f"step {self._step}: Not enough liquidity to sell {quantity} of {symbol}.")
            if handle == 'max':
                max_qty = self.portfolio.get(base_asset, 0)
                if max_qty >= self.get_min_trading_qty(symbol):
                    self.portfolio[base_asset] = 0
                    self.portfolio[quote_asset] = self.portfolio.get(quote_asset, 0) + \
                                                    ((1 - self.fee) * max_qty) * last_kline['price_close']
                
                    self.trade_hist.loc[len(self.trade_hist)] = \
                        [self._step, last_kline['ts_close'], 'sell', symbol, max_qty, last_kline['price_close']]

                    logger.warning(f"step {self._step}: Sold {max_qty} of {symbol}.")
                else:
                    logger.warning(f"step {self._step}: Ignoring sell order.")
            elif handle == 'ignore':
                logger.warning(f"step {self._step}: Ignoring sell order.")

    def tick(self, strategy:TradingStrategy, step:int=1):
        if strategy:
            sell_orders = strategy.sell(self.data, self.portfolio, self.balance)
            for symb, qty in sell_orders.items():
                self.create_sell_order(symb, qty)

            buy_orders = strategy.buy(self.data, self.portfolio, self.balance)
            for symb, qty in buy_orders.items():
                self.create_buy_order(symb, qty)
        else:
            print(f'Step {self._step}')

        self._step += step
        self.update_balance()
        self.balance_hist.loc[self._step] = [self._step, self.balance]
        self.portfolio_hist[self._step] = self.portfolio

    def run(self, strategy:TradingStrategy, step:int=1, offset:int=100, verbose=0):
        i = 0
        while i < offset:
            self.tick(None, step)
            i += step
        while i < self._max_step:
            self.tick(strategy, step)
            i += step
            print(self.balance)
            print(self.portfolio)
        print("End of simuation!")

    def calculate_pnl(self):
        self.balance_hist['prev_balance'] = self.balance_hist['balance'].shift(1)
        self.balance_hist['pnl'] = self.balance_hist['balance'] - self.balance_hist['prev_balance']
        self.balance_hist['cum_pnl'] = self.balance_hist['pnl'].cumsum()
        self.balance_hist['cum_pnl_perc'] = 100 * self.balance_hist['cum_pnl'] / self.balance_hist.loc[0, 'balance']

    def render(self):
        # plot balance hist and pnl hist
        self.calculate_pnl()
        fig, axs = plt.subplots(nrows=2, sharex=True)
        sns.lineplot(data=self.balance_hist, x='step', y='balance', marker='o', ax=axs[0])
        sns.lineplot(data=self.balance_hist, x='step', y='cum_pnl_perc', marker='o', ax=axs[1])

        axs[0].set_title('Balance')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel(f'Balance {self.unit}')

        axs[1].set_title('PnL')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('PnL (%)')

        plt.show()
        # display trades table and pnl per trade (complicated)
        # report of performances
    

            




    