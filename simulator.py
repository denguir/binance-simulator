import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from strategy import TradingStrategy
from datetime import datetime
from binance.client import Client
from order import Order, OrderType, OrderPrice
from joblib import Parallel, delayed, parallel_backend


logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s :: %(levelname)s :: %(name)s :: %(funcName)s :: %(message)s')
logger = logging.getLogger(__name__)


class BinanceSimulator:
    """Simulator for Binance trading platform. Allow the user to apply back-testing
       on his trading strategies.
    """

    def __init__(self, unit: str='USDT', balance: float=10000) -> None:
        self._quotes = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']
        self.unit = unit
        self.balance = balance
        self.portfolio = {unit : balance}

        self._client = Client(None, None)
        self.symbols_info = self.get_symbols_info()
        self._step = 0

        self._data = {}
    
    def init_stats(self, date_from: datetime, resolution: str='1d'):
        self._max_step = max([len(kline) for kline in self._data.values()])
        self._time_step = pd.Timedelta(resolution)
        self._time = date_from

        self.portfolio_hist = {self._step : self.portfolio}
        self.balance_hist = pd.DataFrame(data=[[self._step, self._time, self.balance]],
                                         columns=['step', 'date', 'balance'])
        self.trade_hist = pd.DataFrame(
            columns=['step', 'ts', 'side', 'symbol', 'quantity', 'price'])
        
        self.orders = []
        self.position_hist = pd.DataFrame(
            columns=['symbol', 'date_open', 'price_open', 'date_close', 'price_close', 
                     'quantity', 'closed_quantity', 'pnl', 'pnl_rel', 'close_hist', 'closed']
        )

    @property
    def data(self):
        return {symb : df[:self._step] for symb, df in self._data.items()}

    @property
    def index(self):
        return self._step - 1

    @staticmethod
    def to_timestamp_ms(dt):
        ts = int(1000 * datetime.timestamp(dt))
        return ts

    @staticmethod
    def to_datetime(ts):
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

        base, quote = self.split_symbol(symbol)
        self.portfolio[base] = self.portfolio.get(base, 0)
        self.portfolio[quote] = self.portfolio.get(quote, 0)

    def load_data_from_api(self, date_from:datetime, date_to:datetime, symbols: list='all', resolution: str='1d', n_jobs: int=1):
        # make sure to deal with the case where we dont have the same amount of data for the same time window
        # _step should be an index that is the same for every pair of symbols
        if type(symbols) is list:
            self.symbols = symbols
        else:
            self.symbols = self.symbols_info['symbol']

        with parallel_backend('threading', n_jobs=n_jobs):
            Parallel()(delayed(self.load_symbol_data)(symb, date_from, date_to, resolution) 
                    for symb in self.symbols)
        
        self.init_stats(date_from, resolution)

    def load_data_from_file(self, date_from:datetime, date_to:datetime, filename:str):
        if filename.endswith('.parquet'):
            df = pd.read_parquet(filename)
        else:
            df = pd.read_csv(filename)
        
        df = df[df['open_time'].between(date_from, date_to)]
        
        for symbol, df_symbol in df.groupby('symbol'):
            self._data[symbol] = df_symbol.sort_values('open_time')

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
        if symbol in self.data.keys():
            df = self.data[symbol]
            return df.iloc[self._step - 1]
        else:
            raise Exception(f"{symbol} price is not loaded.")

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
                    logging.warning(f"base {base} seems to have no exchange with one of the supported quotes {self._quotes}.")
                    price = 0.0
        return price

    def update_balance(self):
        balance = 0.0
        for asset, qty in self.portfolio.items():
            price = self.get_price(asset, self.unit)
            balance += (price * qty)
        self.balance = balance

    def open_position(self, symbol, quantity, price):
        ['symbol', 'date_open', 'price_open', 'date_close', 'price_close', 
                     'quantity', 'closed_quantity', 'pnl', 'pnl_rel', 'close_hist', 'closed']
        self.position_hist.loc[len(self.position_hist)] = \
                [symbol, self._time, price, np.nan, np.nan, quantity, 0, np.nan, np.nan, [], False]

    def close_position(self, symbol, quantity, price):
        qty_to_close = quantity
        while qty_to_close:
            open_position = self.position_hist[
                (self.position_hist['symbol'] == symbol) & (self.position_hist['closed'] == False)]\
                    .nsmallest(1, 'date_open')
            if open_position.empty:
                break

            closed_qty = min(qty_to_close, open_position['quantity'].item() - open_position['closed_quantity'].item())
            tot_closed_qty = open_position['closed_quantity'].item() + closed_qty
            closed_ratio = closed_qty / open_position['quantity'].item()
            close_hist = open_position['close_hist'].item() + [(closed_ratio, price)]

            self.position_hist.loc[open_position.index, 'closed_quantity'] = tot_closed_qty
            self.position_hist.at[open_position.index[0], 'close_hist'] = close_hist

            if open_position['quantity'].item() == tot_closed_qty:
                self.position_hist.loc[open_position.index, 'closed'] = True
                self.position_hist.loc[open_position.index, 'date_close'] = self._time
            else:
                base, quote = self.split_symbol(symbol)
                delta_qty = open_position['quantity'].item() - tot_closed_qty
                qty_ratio = delta_qty / open_position['quantity'].item()
                close_hist = close_hist + [(qty_ratio, self.get_price(base, quote))]
            
            price_close = sum(x[0] * x[1] for x in close_hist)
            unit_pnl = price_close - open_position['price_open'].item()
            pnl = open_position['quantity'].item() * unit_pnl
            pnl_rel = unit_pnl / open_position['price_open'].item()

            self.position_hist.loc[open_position.index, 'price_close'] = price_close
            self.position_hist.loc[open_position.index, 'pnl'] = pnl
            self.position_hist.loc[open_position.index, 'pnl_rel'] = pnl_rel

            qty_to_close -= closed_qty

    def get_order_price(self, kline, order: Order):
        if order.price == OrderPrice.Open:
            price = kline['price_open']
        elif order.price == OrderPrice.Close:
            price = kline['price_close']
        elif order.price == OrderPrice.High:
            price = kline['price_high']
        elif order.price == OrderPrice.Low:
            price = kline['price_low']
        elif order.price == OrderPrice.Mean:
            price = (kline['price_low'] + kline['price_high']) / 2
        return price

    def fill_order(self, order: Order, handle=True):
        base_asset, quote_asset = self.split_symbol(order.symbol)
        last_kline = self.get_last_kline(order.symbol)
        price = self.get_order_price(last_kline, order)
        quantity = order.quantity

        if order.side == OrderType.Buy:
            if self.portfolio[quote_asset] >= quantity * price:
                self.portfolio[base_asset] += ((1 - order.fee) * quantity)
                self.portfolio[quote_asset] -= quantity * price
            else:
                logger.warning(
                    f"step {self._step}: Not enough liquidity to buy {quantity} of {order.symbol}.")
                max_qty = self.portfolio[quote_asset] / price
                if handle and max_qty >= self.get_min_trading_qty(order.symbol):
                    self.portfolio[base_asset] += ((1 - order.fee) * max_qty)
                    self.portfolio[quote_asset] = 0
                    quantity = max_qty
                    logger.warning(f"step {self._step}: Bought {quantity} of {order.symbol}.")
                else:
                    quantity = 0
                    logger.warning(f"step {self._step}: Ignoring buy order.")
        
        elif order.side == OrderType.Sell:
            if self.portfolio[base_asset] >= quantity:
                self.portfolio[base_asset] -= quantity
                self.portfolio[quote_asset] += ((1 - order.fee) * quantity) * price
            else:
                logger.warning(
                    f"step {self._step}: Not enough liquidity to sell {quantity} of {order.symbol}.")
                max_qty = self.portfolio[base_asset]
                if handle and max_qty >= self.get_min_trading_qty(order.symbol):
                    self.portfolio[base_asset] = 0
                    self.portfolio[quote_asset] += ((1 - order.fee) * max_qty) * price
                    quantity = max_qty
                    logger.warning(f"step {self._step}: Sold {quantity} of {order.symbol}.")
                else:
                    quantity = 0
                    logger.warning(f"step {self._step}: Ignoring sell order.")
            
        if quantity:
            self.trade_hist.loc[len(self.trade_hist)] = \
                [self._step, self._time, order.side.value, order.symbol, quantity, price]
            
            if order.side == OrderType.Buy:
                self.open_position(order.symbol, quantity, price)
            elif order.side == OrderType.Sell:
                self.close_position(order.symbol, quantity, price)

    def order(self, strategy:TradingStrategy):
        if strategy:
            new_orders = strategy.order()
            self.orders += new_orders
        else:
            logging.info(f'No orders @ step {self._step}')

    def tick(self, strategy:TradingStrategy, step:int=1):
        while len(self.orders):
            next_order = self.orders.pop(0)
            self.fill_order(next_order)

        self.order(strategy)
        self._step += step
        self._time += (step * self._time_step)
        self.update_balance()
        self.balance_hist.loc[self._step] = [self._step, self._time, self.balance]
        self.portfolio_hist[self._step] = self.portfolio

    def run(self, strategy:TradingStrategy, step:int=1, offset:int=100, verbose=0):
        i = 0
        self.strategy = strategy
        while i < offset:
            self.tick(None, step)
            i += step
        while i < self._max_step:
            self.tick(strategy, step)
            strategy._update(self._step, 
                             self.data, 
                             self.portfolio, 
                             self.balance, 
                             self.unit)
            i += step
        logger.info("End of simuation!")

    def calculate_pnl(self):
        self.balance_hist['prev_balance'] = self.balance_hist['balance'].shift(1)
        self.balance_hist['pnl'] = self.balance_hist['balance'] - self.balance_hist['prev_balance']
        self.balance_hist['cum_pnl'] = self.balance_hist['pnl'].cumsum()
        self.balance_hist['cum_pnl_perc'] = 100 * self.balance_hist['cum_pnl'] / self.balance_hist.loc[0, 'balance']

    def _render_trades(self):
        n_symbols = len(self.trade_hist['symbol'].unique())
        fig, axs = plt.subplots(nrows=n_symbols, sharex=True)
        if n_symbols == 1:
            axs = (axs, )
        for i, (symb, trades) in enumerate(self.trade_hist.groupby('symbol')):
            symb_data = self.data[symb]
            symb_data['date'] = symb_data['ts_open'].apply(lambda ts: self.to_datetime(ts))
            sns.lineplot(data=symb_data, x='date', y='price_close', ax=axs[i])
            sns.scatterplot(data=trades,
                            x='ts', 
                            y='price', 
                            hue='side', 
                            style='side',
                            palette={'sell':(1.0, 0.0, 0.0), 'buy':(0.0, 1.0, 0.0)}, 
                            markers={'sell':'v', 'buy':'^'}, 
                            ax=axs[i],
                            legend='brief')
            axs[i].set_title(f'{symb}')
            axs[i].set_xlabel('Timestamp')
            axs[i].set_ylabel(f'Price {symb}')
        plt.show()
    
    def _render_pnl(self):
        self.calculate_pnl()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=self.balance_hist['date'], y=self.balance_hist['balance'], name="balance"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=self.balance_hist['date'], y=self.balance_hist['cum_pnl_perc'], name="cumulative pnl"),
            secondary_y=True,
        )
        fig.update_layout(
            title_text=f"Balance & PnL ({self.unit})"
        )

        fig.update_xaxes(title_text="date")
        fig.update_yaxes(title_text=f"<b>Balance ({self.unit})</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>PnL (%)</b>", secondary_y=True)

        return fig

    def _render_positions(self):
        positions = self.position_hist.drop(columns=['close_hist'])
        positions['price_open'] = positions['price_open'].map('{:,.2f}'.format)
        positions['price_close'] = positions['price_close'].map('{:,.2f}'.format)
        positions['quantity'] = positions['quantity'].map('{:,.2f}'.format)
        positions['closed_quantity'] = positions['closed_quantity'].map('{:,.2f}'.format)
        positions['pnl'] = positions['pnl'].map('{:,.2f}'.format)
        positions['pnl_rel'] = positions['pnl_rel'].map('{:,.2f}%'.format)
        return positions.sort_values(by='date_open', ascending=False)

    def render(self, app):
        pnl = self._render_pnl()
        positions = self._render_positions()
        #trades = self._render_trades()
        app.layout = html.Div(children=[
                        html.H1(f'Simulation report of strategy {str(self.strategy)}'),

                        html.H2(f'Strategy performance'),
                        dcc.Graph(
                            id='pnl',
                            figure=pnl
                        ),

                        html.H2(f'Position history'),
                        dash_table.DataTable(
                                id='positions', 
                                data=positions.to_dict('records'),
                                columns=[{"name": i, "id": i} for i in positions.columns],
                                page_size=20
                            )
                    ])