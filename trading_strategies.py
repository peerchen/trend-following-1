# Not import ... as ...  for parallel computing.

import itertools
import numpy
import pandas
from pandas import Series, DataFrame
from datetime import datetime
import matplotlib
from matplotlib import pyplot
import matplotlib.transforms
from utils import *


class Trading_Strategy:
    """
    Class that performs a trading strategy on a single asset.
    """
    
    def __init__(self, prices, equity=1e6, heat=0.05, tr_stop_loss=0.05,
                 position_step=1, name='', **kwargs):
        """
        Defines the initial state and hyper parameters of the trading strategy.
        
        Parameters
        ----------
        prices: DataFrame with columns Date, Open, High, Low, Close, ...
        equity: Initial equity to invest in the asset.
        heat: Proportion of the equity that one is willing to risk.
        tr_stop_loss: percentage for trailing stop loss.
        position_step: Minimum steps in position' sizes
        name: Name of the asset for plotting purposes.
        """
        self.dates = prices.index
        self.first_trading_day = None
        self.last_trading_day = None
        self.prices_df = prices
        self.prices = {s: prices[s].values for s in list(prices.keys())}
        self.heat = heat
        self.tr_stop_loss = tr_stop_loss
        self.position_step = position_step
        self.name = name
        self.today_prices = None
        
        self.equity = {'Available_Balance': numpy.ones(len(self.dates)) * equity,
                       'Closing_Balance': numpy.ones(len(self.dates)) * equity,
                       'Position': numpy.zeros(len(self.dates)),
                       'Open_Profit': numpy.zeros(len(self.dates)),
                       'Position_Value': numpy.zeros(len(self.dates)),
                       'Equity': numpy.ones(len(self.dates)) * equity}
        
        self.orders = {'buy_stop': numpy.zeros(len(self.dates)) * numpy.nan,
                       'sell_stop': numpy.zeros(len(self.dates)) * numpy.nan,
                       'protective_buy': numpy.zeros(len(self.dates)) * numpy.nan,
                       'protective_sell': numpy.zeros(len(self.dates)) * numpy.nan,
                       'risk_per_lot': numpy.zeros(len(self.dates)) * numpy.nan,
                       'amount': numpy.zeros(len(self.dates)) * numpy.nan}
        
        self.trades = []
        self.max_drawdown = 0
        self.performance = {}
        
        self.init_state(**kwargs)
    
    
    def init_state(self):
        pass
    
    
    def update_state(self, today_i):
        """
        Updates the state as of market close from today_i - 1,
        for determining orders before market openas today_i.
        """
        pass
    
    
    def orders_before_trading_starts(self, today_i):
        """
        Computes the orders for the day (today).
        
        Parameters
        ----------
        prices : a prices data frame with a Date index and columns: Open, High, Low, Close, Volume.
            Prices until today only for accessing current date.
        prev_day_position : position in the asset at the end of yesterday.
        """
        self.entry_order_prices(today_i)
        self.protective_order_prices(today_i)
        self.order_amounts(today_i)
    
    
    def entry_order_prices(self, today_i):
        """
        Computes the entry orders for the day (today) according to a Support and Resistance system.
        """
        pass
    
    
    def protective_order_prices(self, today_i):
        """
        Computes the protective orders for the day (today) according to trailing stop loss.
        """
        state = {k: self.state[k][today_i - 1] for k in self.state.keys()}
        prev_day_position = self.equity['Position'][today_i - 1]
        
        if self.orders['buy_stop'][today_i] > 0:
            self.orders['protective_sell'][today_i] = \
                self.orders['buy_stop'][today_i] * (1 - self.tr_stop_loss)
        elif prev_day_position > 0:
            self.orders['protective_sell'][today_i] = \
                max(self.orders['protective_sell'][today_i - 1], 
                    self.prices['Close'][today_i - 1] * (1 - self.tr_stop_loss))
            
        elif self.orders['sell_stop'][today_i] > 0:
            self.orders['protective_buy'][today_i] = \
                self.orders['sell_stop'][today_i] * (1 + self.tr_stop_loss)
        elif prev_day_position < 0:
            self.orders['protective_buy'][today_i] = \
                min(self.orders['protective_buy'][today_i - 1], 
                    self.prices['Close'][today_i - 1] * (1 + self.tr_stop_loss))
        
        # When the trend changes, close positions at market open...
        elif prev_day_position > 0 and state['Trend'] != 1:
            self.orders['protective_sell'][today_i] = self.prices['Open'][today_i]
        elif prev_day_position < 0 and state['Trend'] != -1:
            self.orders['protective_buy'][today_i] = self.prices['Open'][today_i]
        else:
            pass
    
    
    def order_amounts(self, today_i):
        """
        Sets the order amounts for the day
        """
        equity_to_risk = self.equity['Available_Balance'][today_i - 1] * self.heat

        if self.orders['buy_stop'][today_i] > 0 and self.orders['protective_sell'][today_i] > 0:
            self.orders['risk_per_lot'][today_i] = self.orders['buy_stop'][today_i] - \
                self.orders['protective_sell'][today_i] + 1e-12
            self.orders['amount'][today_i] = self.position_step * \
                    numpy.round((1e-8 + equity_to_risk / self.orders['risk_per_lot'][today_i]) / \
                                self.position_step)
        
        elif self.orders['sell_stop'][today_i] > 0 and self.orders['protective_buy'][today_i] > 0:
            self.orders['risk_per_lot'][today_i] = self.orders['protective_buy'][today_i] - \
                self.orders['sell_stop'][today_i] + 1e-12
            self.orders['amount'][today_i] = - self.position_step * \
                numpy.round((1e-8 + equity_to_risk / self.orders['risk_per_lot'][today_i]) / \
                            self.position_step)
        else:
            pass
    
    
    def excecute_orders(self, today_i, skid=0.5):
        """
        Excecutes orders during the day.
        """
        
        position = self.equity['Position'][today_i - 1]  # previous day's position
        balance = self.equity['Available_Balance'][today_i - 1]  # previous day's balance
        o = {k: self.orders[k][today_i] for k in self.orders.keys()}  # orders for the day
        tp = self.today_prices

        if position == 0:  # Enter the market.

            if not numpy.isnan(o['buy_stop']) and not numpy.isnan(o['protective_sell']) and \
                o['buy_stop'] > o['protective_sell'] and tp['High'] > o['buy_stop']:
                
                buy_price = tp['High'] - skid * (tp['High'] - max(tp['Open'], tp['Low'], o['buy_stop']))
                trade = {'Date': self.dates[today_i],
                         'Price': buy_price,
                         'Amount': min(o['amount'], balance)}
                position += o['amount']
                balance -= o['amount'] * buy_price
                self.trades.append(trade)


            elif not numpy.isnan(o['sell_stop']) and not numpy.isnan(o['protective_buy']) and \
                o['sell_stop'] < o['protective_buy'] and tp['Low'] < o['sell_stop']:
                
                sell_price = tp['Low'] + skid * (min(tp['Open'], tp['High'], o['sell_stop']) - tp['Low'])
                trade = {'Date': self.dates[today_i],
                         'Price': sell_price,
                         'Amount': max(o['amount'], -balance)}  # o['amount'] < 0
                position += o['amount']
                balance -= o['amount'] * sell_price
                self.trades.append(trade)

        if position != 0:  # Close positions.

            if not numpy.isnan(o['protective_buy']) and tp['High'] > o['protective_buy']:
                
                buy_price = tp['High'] - \
                    skid * (tp['High'] - max(tp['Open'], tp['Low'], o['protective_buy']))
                amount = max(-position, 0)
                trade = {'Date': self.dates[today_i],
                         'Price': buy_price,
                         'Amount': amount}
                position += amount
                balance -= amount * buy_price
                self.trades.append(trade)

            if not numpy.isnan(o['protective_sell']) and tp['Low'] < o['protective_sell']:
                
                sell_price = tp['Low'] + \
                    skid * (min(tp['Open'], tp['High'], o['protective_sell']) - tp['Low'])
                amount = min(-position, 0)
                trade = {'Date': self.dates[today_i],
                         'Price': sell_price,
                         'Amount': amount}
                position += amount
                balance -= amount * sell_price
                self.trades.append(trade)

        if self.dates[today_i] == self.dates[-1] and position != 0:  # Last day
            
            if position > 0:
                
                sell_price = tp['Low'] + \
                    skid * (min(tp['Open'], tp['Close'], tp['High']) - tp['Low'])
                amount = min(-position, 0)
                trade = {'Date': self.dates[today_i],
                         'Price': sell_price,
                         'Amount': amount}
                position += amount
                balance -= amount * sell_price
                self.trades.append(trade)
                
            else:  # position < 0
                
                buy_price = tp['High'] - \
                    skid * (tp['High'] - max(tp['Open'], tp['Close'], tp['Low']))
                amount = max(-position, 0)
                trade = {'Date': self.dates[today_i],
                         'Price': buy_price,
                         'Amount': amount}
                position += amount
                balance -= amount * buy_price
                self.trades.append(trade)

        if position != 0:
            open_profit = position * (tp['Close'] - self.trades[-1]['Price'])
        else:
            open_profit = 0

        if self.equity['Position'][today_i - 1] != 0 and position == 0:
            closing_balance = balance
        else:
            closing_balance = self.equity['Closing_Balance'][today_i - 1]


        
        self.equity['Available_Balance'][today_i] = balance
        self.equity['Closing_Balance'][today_i] = closing_balance
        self.equity['Position'][today_i] = position
        self.equity['Open_Profit'][today_i] = open_profit
        self.equity['Position_Value'][today_i] = position * tp['Close']
        self.equity['Equity'][today_i] = balance + position * tp['Close']
    
    
    
    
    def excecute(self, warmup=25, end=None, skid=0.5):
        """
        Excecutes the trading strategy and computes its performance measures.
        
        Parameters
        ----------
        warmup: warmup period in days or a proportion from 0 to 1.
        end: either a datetime or a proportion from 0 to 1 (all sample).
        """
        peak = self.equity['Equity'][0]
        low = peak
        drawdown = 0
        max_drawdown = 0
        
        if end is None:
            end = self.dates[-1]
        elif type(end) == float:
            end = self.dates[int(len(self.dates) * end)]
        
        
        if warmup < 1:
            warmup = int(len(self.dates) * warmup)
            
        self.first_trading_day = self.dates[warmup]
        self.last_trading_day = end
        
        for i in range(1, len(self.dates)):
            self.today_prices = {k: self.prices[k][i] for k in self.prices.keys()}
            self.update_state(i)
            if i >= warmup:
                self.orders_before_trading_starts(i)
                self.excecute_orders(i, skid)
                
                # Max drawdown
                if self.equity['Equity'][i] > peak:
                    peak = self.equity['Equity'][i]
                    low = peak
                if self.equity['Equity'][i] < low:
                    low = self.equity['Equity'][i]
                    drawdown = low / peak - 1
                max_drawdown = min(drawdown, max_drawdown)
                
            if self.dates[i] > end:
                self.equity['Available_Balance'][i] = self.equity['Available_Balance'][i - 1]
                self.equity['Closing_Balance'][i] = self.equity['Closing_Balance'][i - 1]
                self.equity['Position'][i] = self.equity['Position'][i - 1]
                self.equity['Open_Profit'][i] =  self.equity['Position'][i] * \
                    (self.prices['Close'][i] - self.trades[-1]['Price'])
                self.equity['Position_Value'][i] = self.equity['Position'][i] * self.prices['Close'][i]
                self.equity['Equity'][i] = self.equity['Equity'][i - 1]
        
        self.max_drawdown = max_drawdown
        self.compute_performance()
    
    
    def compute_performance(self):
        """
        Computes performance indicators for the trading strategy.
        """
        
        self.performance['Years'] = (self.last_trading_day - self.first_trading_day).days / 364.25
        self.performance['Ratio'] = self.equity['Equity'][-1] / self.equity['Equity'][0]
        # Instantaneously Compounding Annual Gain
        self.performance['ICAGR'] = numpy.log(self.performance['Ratio']) / self.performance['Years']
        self.performance['Max_Drawdown'] = -self.max_drawdown
        # How Often the System Earns Back its Biggest Drawdown
        self.performance['Bliss'] = self.performance['ICAGR'] / self.performance['Max_Drawdown']
        
        eqty = self.equity['Equity'][self.dates > self.first_trading_day]
        self.performance['Volatility'] = numpy.std(numpy.log(eqty[1:] / eqty[:-1])) * numpy.sqrt(364.25)
        
        # Lake Ratio (see http://www.seykota.com/tribe/risk/index.htm)
        lake_bottom = Series(eqty) / self.equity['Equity'][1]
        lake_surface = lake_bottom.cummax()

        earth = numpy.trapz(lake_bottom.values)
        water = numpy.trapz(lake_surface - lake_bottom)
        self.performance['Lake_Ratio'] = water / earth
        
        earth0 = numpy.trapz(lake_bottom.values - 1)
        self.performance['Sea_Ratio'] = water / earth0
    
    
    
    def get_prices(self):
        return self.prices_df
    
    
    def get_state(self):
        if not any([i == 'state_df' for i in dir(self)]):
            self.state_df = DataFrame(self.state, index=self.dates).iloc[:-1]
        return self.state_df
    
    
    def get_orders(self):
        if not any([i == 'orders_df' for i in dir(self)]):
            self.orders_df = DataFrame(self.orders, index=self.dates)
        return self.orders_df
    
    
    def get_trades(self):
        if not any([i == 'trades_df' for i in dir(self)]):
            self.trades_df = dict_list_to_DataFrame(self.trades).set_index('Date')
        return self.trades_df
    
    
    def get_equity(self):
        if not any([i == 'equity_df' for i in dir(self)]):
            self.equity_df =DataFrame(self.equity, index=self.dates)
        return self.equity_df
    
    
    def get_trades_profit(self):
        if not any([i == 'trades_profit' for i in dir(self)]):
            tr = self.get_trades().reset_index()

            entries = tr.iloc[numpy.arange(0, len(tr), 2)]
            entries = entries.reset_index().rename(columns={'index': 'transaction'})
            exits = tr.iloc[numpy.arange(1, len(tr) + 1, 2)]
            exits = exits.reset_index().rename(columns={'index': 'transaction'})
            exits = exits.drop('Amount', axis=1)

            tr_pr = pandas.concat((entries.rename(columns={t: 'Entry ' + t for t in ['Date', 'Price']}),
                                   exits.rename(columns={t: 'Exit ' + t for t in ['Date', 'Price']})),
                                   axis=1)
            tr_pr = tr_pr.assign(PL=tr_pr.Amount * \
                                     (tr_pr['Exit Price'] - tr_pr['Entry Price']),
                                 PL_pct=tr_pr.Amount.apply('sign') * \
                                     (tr_pr['Exit Price']/tr_pr['Entry Price'] - 1))
            self.trades_profit = tr_pr
        return self.trades_profit
    
    
    
    def plot_prices(self):
        fig, ax = pyplot.subplots()
        ax.plot(self.prices_df.Open)
        ax.plot(self.prices_df.High)
        ax.plot(self.prices_df.Low)
        ax.plot(self.prices_df.Close)
        ax.axhline(0, color='grey', lw=2, alpha=0.75)
        ax.set_title(self.name)
        ax.legend()
    
    
    def plot_state(self):
        x = pandas.merge(self.prices_df[['Open', 'High', 'Low', 'Close']],
                         self.get_state(), left_index=True, right_index=True, how='outer')
        tit = self.name + ' State.'
        pal = pyplot.get_cmap('Paired').colors
        fig, ax = pyplot.subplots()
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(x.index, 0, x.High.max(), where= x.Trend > 0, facecolor=pal[0],
                        alpha=0.25, transform=trans, label='Trend up')
        ax.fill_between(x.index, 0, x.High.max(), where= x.Trend < 0, facecolor=pal[4],
                        alpha=0.25, transform=trans, label='Trend down')

        ax.plot(x.drop('Trend', axis=1))
        ax.axhline(0, color='grey', lw=2, alpha=0.75)
        ax.legend()
        ax.set_title(tit)

    
    def plot_trades(self):
        tr_pr = self.get_trades_profit()
        aux = DataFrame({'Long': tr_pr.query('Amount > 0').PL_pct,
                         'Short': tr_pr.query('Amount < 0').PL_pct})
        aux.plot.hist(stacked=True, alpha=0.5, bins=40)

        pyplot.grid(c='grey', alpha=0.5)
        pct_long = (tr_pr.Amount > 0).mean()
        pct_profitable = (tr_pr.PL > 0).mean()
        pr_long = tr_pr.query('Amount > 0').PL.sum()
        pr_short = tr_pr.query('Amount < 0').PL.sum()
        pr = tr_pr.PL.sum()

        tit = self.name + ' - Trades: ' + str(int(10000 * pct_long) / 100) + '% Long, ' 
        tit += str(int(10000 * pct_profitable) / 100) + '% Profitable.\n'
        tit += 'Profit Long: ' + str(int(100 * pr_long) / 100) 
        tit += ', profit short: ' + str(int(100 * pr_short) / 100)
        tit += ', total: ' + str(int(100 * pr) / 100)
        pyplot.title(tit)
        pyplot.xlabel('P&L (%)')
        pyplot.show()
        
    def plot_orders(self):
        x = pandas.merge(self.prices_df[['Open', 'High', 'Low', 'Close']],
                         self.get_orders(), left_index=True, right_index=True, how='outer')
        x = pandas.merge(x, self.get_state(),
                         left_index=True, right_index=True, how = 'outer')
        x = pandas.merge(x, self.get_equity(),
                         left_index=True, right_index=True, how = 'outer')
        tit = self.name + ' - Orders.'
        pal = pyplot.get_cmap('Paired').colors
        pal2 = pyplot.get_cmap('Set1').colors
        fig, ax = pyplot.subplots()
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(x.index, 0, x.High.max(), where= x.Trend > 0, facecolor=pal[0],
                        alpha=0.25, transform=trans, label='Trend up')
        ax.fill_between(x.index, 0, x.High.max(), where= x.Trend < 0, facecolor=pal[4],
                        alpha=0.25, transform=trans, label='Trend down')

        ax.plot(x.Close, label='Close', c=pal2[1])
        ax.plot(x.protective_buy, label='protective_buy', c=pal2[4])
        ax.plot(x.protective_sell, label='protective_sell', c=pal2[4])
        ax.plot(x.buy_stop, 'o', label='buy_stop', alpha=0.5, c=pal2[2])
        ax.plot(x.sell_stop, 'o', label='sell_stop', alpha=0.5, c=pal2[3])
        ax.plot(x.Position.apply('sign') * x.Close.min() / 2,
                label='Position (+/-)',c=pal2[0], alpha=0.5)
        ax.axhline(0, color='grey', lw=2, alpha=0.75)
        ax.legend()
        ax.set_title(tit)
    
    
    def plot_equity(self):
        x = pandas.merge(self.get_equity(), self.get_state(),
                         left_index=True, right_index=True, how = 'outer')
        tit = self.name + ', '
        tit += 'heat: ' + str(self.heat) + '.\n'
        tit += 'Initial Equity: ' + str(int(self.equity['Equity'][0]))
        tit += ', Ending Equity: ' + str(int(self.equity['Equity'][-1])) + ', '
        tit += 'Total Return: '
        tit += str(int(10000 * self.equity['Equity'][-1] / self.equity['Equity'][1] - 1) / 100) + '%.\n'
#         tit += 'ICAGR: ' + str(int(10000 * self.performance['ICAGR']) / 100) + '%.\n'
#         tit += 'Volatility: ' + str(int(10000 * self.performance['Volatility']) / 100) + '%, '
        tit += 'Lake Ratio: ' + str(int(10000 * self.performance['Lake_Ratio']) / 100) + '%, '
        tit += 'Max. Drawdown: ' + str(int(10000* self.performance['Max_Drawdown']) / 100) + '%, '
#         tit += 'Bliss: ' + str(numpy.round(364.25 * self.performance['Bliss'], 1)) + ' days.'
        pal = pyplot.get_cmap('Paired').colors
        fig, ax = pyplot.subplots()
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(x.index, 0, x.Equity.max(), where= x.Trend > 0, facecolor=pal[0],
                        alpha=0.25, transform=trans, label='Trend up')
        ax.fill_between(x.index, 0, x.Equity.max(), where= x.Trend < 0, facecolor=pal[4],
                        alpha=0.25, transform=trans, label='Trend down')
        ax.plot(x.index, x.Equity)
        ax.plot(x.index, x.Closing_Balance)
        ax.axhline(0, color='grey', lw=2, alpha=0.75)
        ax.set_title(tit)
        ax.legend()
    
    
    def plot_lake_ratio(self):
        """
        Illustrates the computation of the lake ratio.
        See http://www.seykota.com/tribe/risk/index.htm
        """
        eqty = self.equity['Equity'][self.dates > self.first_trading_day]
        lake_bottom = Series(eqty) / self.equity['Equity'][1]
        lake_surface = lake_bottom.cummax()
        fig, ax = pyplot.subplots()
        ax.fill_between(lake_bottom.index, y1 = lake_bottom, y2 = lake_surface, alpha=0.5)
        ax.fill_between(lake_bottom.index, y1 = 0, y2 = lake_bottom, alpha=0.5)
        ax.axhline(1, color='grey', lw=2, alpha=0.75)
        ax.axhline(0, color='grey', lw=2, alpha=0.75)
        ax.plot(lake_bottom)
        tit = self.name + ' - Lake Ratio: ' 
        tit += str(int(10000 * self.performance['Lake_Ratio']) / 100) + '%'
        tit += ', Sea Ratio (areas over 1): ' 
        tit += str(int(10000 * self.performance['Sea_Ratio']) / 100) + '%.'
        ax.set_title(tit)








class RS_Trading_Strategy(Trading_Strategy):
    """
    Implements a Support and Resistance trading system, 
    based on http://www.seykota.com/tribe/TSP/SR/index.htm.
    """
    
    def init_state(self, days_fast=20, days_slow=140):
        """
        Parameters
        ----------
        days_fast: days in the moving window for computing the fast support and resistance.
        days_slow: days in the moving window for computing the slow support and resistance.
        """

        self.state = {'Support_slow': numpy.zeros(len(self.dates)) * numpy.nan,
                      'Support_fast': numpy.zeros(len(self.dates)) * numpy.nan,
                      'Resistance_slow': numpy.zeros(len(self.dates)) * numpy.nan,
                      'Resistance_fast': numpy.zeros(len(self.dates)) * numpy.nan,
                      'Trend':  numpy.zeros(len(self.dates))}
        
        self.days_fast = days_fast
        self.days_slow = days_slow
        
    
    
    def update_state(self, today_i):
        """
        Updates the Resistance, Support, and Trend variables.
        """
        
        # State calculations regarding previous day at market closing time.
        past_prices = {k: self.prices[k][:today_i] for k in self.prices.keys()}
        Support_slow = past_prices['Low'][-self.days_slow:].min()
        Support_fast = past_prices['Low'][-self.days_fast:].min()
        Resistance_slow = past_prices['High'][-self.days_slow:].max()
        Resistance_fast = past_prices['High'][-self.days_fast:].max()
        
        if past_prices['High'][-1] >= Resistance_slow and past_prices['Low'][-1] > Support_slow:
            Trend = 1
        elif past_prices['Low'][-1] <= Support_slow and past_prices['High'][-1] < Resistance_slow:
            Trend = -1
        else:
            Trend = self.state['Trend'][today_i - 2]
        
        self.state['Support_slow'][today_i - 1] = Support_slow
        self.state['Support_fast'][today_i - 1] = Support_fast
        self.state['Resistance_slow'][today_i - 1] = Resistance_slow
        self.state['Resistance_fast'][today_i - 1] = Resistance_fast
        self.state['Trend'][today_i - 1] = Trend
    
        
    def entry_order_prices(self, today_i):
        """
        Computes the entry orders for the day (today) according to a Support and Resistance system.
        """
        prev_day_position = self.equity['Position'][today_i - 1]
        state = {k: self.state[k][today_i - 1] for k in self.state.keys()}
        
        if prev_day_position == 0:
            if state['Trend'] == 1:
                self.orders['buy_stop'][today_i] = state['Resistance_fast']
            elif state['Trend'] == -1:
                self.orders['sell_stop'][today_i] = state['Support_fast']
                
        else:
            pass
    
    
    
    def protective_order_prices(self, today_i):
        """
        Computes the protective orders for the day (today) according to a Support and Resistance system.
        """
        state = {k: self.state[k][today_i - 1] for k in self.state.keys()}
        prev_day_position = self.equity['Position'][today_i - 1]
        
        if self.orders['buy_stop'][today_i] > 0 or prev_day_position > 0:
            self.orders['protective_sell'][today_i] = state['Support_fast']
            
        elif self.orders['sell_stop'][today_i] > 0 or prev_day_position < 0:
            self.orders['protective_buy'][today_i] = state['Resistance_fast']
            
        else:
            pass
    
    
        
    def plot_state(self):
        x = pandas.merge(self.prices_df, self.get_state(),
                         left_index=True, right_index=True, how='outer')
        tit = self.name + ' - days_fast: ' + str(self.days_fast) 
        tit += ', days_slow: ' + str(self.days_slow) + '.'
        pal = pyplot.get_cmap('Paired').colors
        fig, ax = pyplot.subplots()
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(x.index, 0, x.High.max(), where= x.Trend > 0, facecolor=pal[0],
                        alpha=0.25, transform=trans, label='Trend up')
        ax.fill_between(x.index, 0, x.High.max(), where= x.Trend < 0, facecolor=pal[4],
                        alpha=0.25, transform=trans, label='Trend down')

        ax.plot(x.index, x.High, c=pal[1])
        ax.plot(x.index, x.Low, c=pal[0])
        ax.plot(x.index, x.Resistance_fast, c=pal[2])
        ax.plot(x.index, x.Resistance_slow, c=pal[3])
        ax.plot(x.index, x.Support_fast, c=pal[4])
        ax.plot(x.index, x.Support_slow, c=pal[5])
        ax.axhline(0, color='grey', lw=2, alpha=0.75)
        ax.legend()
        ax.set_title(tit)
    
    
    def plot_equity(self):
        x = pandas.merge(self.get_equity(), self.get_state(),
                         left_index=True, right_index=True, how = 'outer')
        tit = self.name + ' - days_fast: ' + str(self.days_fast) 
        tit += ', days_slow: ' + str(self.days_slow) + ', '
        tit += 'heat: ' + str(self.heat) + '.\n'
        tit += 'Initial Equity: ' + str(int(self.equity['Equity'][0]))
        tit += ', Ending Equity: ' + str(int(self.equity['Equity'][-1])) + ', '
        tit += 'Total Return: ' 
        tit += str(int(10000 * self.equity['Equity'][-1] / self.equity['Equity'][1] - 1) / 100) + '%, '
        tit += 'ICAGR: ' + str(int(10000 * self.performance['ICAGR']) / 100) + '%.\n'
        tit += 'Volatility: ' + str(int(10000 * self.performance['Volatility']) / 100) + '%, '
        tit += 'Lake Ratio: ' + str(int(10000 * self.performance['Lake_Ratio']) / 100) + '%, '
        tit += 'Max. Drawdown: ' + str(int(10000* self.performance['Max_Drawdown']) / 100) + '%, '
        tit += 'Bliss: ' + str(numpy.round(364.25 * self.performance['Bliss'], 1)) + ' days.'
        pal = pyplot.get_cmap('Paired').colors
        fig, ax = pyplot.subplots()
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(x.index, 0, x.Equity.max(), where= x.Trend > 0, facecolor=pal[0],
                        alpha=0.25, transform=trans, label='Trend up')
        ax.fill_between(x.index, 0, x.Equity.max(), where= x.Trend < 0, facecolor=pal[4],
                        alpha=0.25, transform=trans, label='Trend down')
        ax.plot(x.index, x.Equity)
        ax.plot(x.index, x.Closing_Balance)
        ax.axhline(0, color='grey', lw=2, alpha=0.75)
        ax.set_title(tit)
        ax.legend()



def test_RS_Trading_Strategy():
    """
    Make sure the results match those on http://www.seykota.com/tribe/TSP/SR/index.htm..
    """

    price = pandas.read_csv(filepath_or_buffer='test/Seykota GC----C.csv', header=None,
               names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open_Interest'], 
               index_col=0)

    price.index = [datetime.strptime(str(i), '%Y%m%d') for i in price.index.values]

    rs_tr = RS_Trading_Strategy(price, equity=1e6, position_step=1e2,
                                days_fast=20, days_slow=140,
                                name='Comex Gold')
    rs_tr.excecute(warmup=20)

    # test states
    stts = rs_tr.get_state()
    metrics = pandas.merge(price[['Open', 'High', 'Low', 'Close']],
                           stts, left_index=True, right_index=True)
    metrics_log = pandas.read_excel(io='test/Metrics_Log_1-1.xlsx', index_col=0)
    assert numpy.all(numpy.equal(metrics_log.values, metrics.values))

    # test trades
    trds = rs_tr.get_trades()
    trade_log = pandas.read_excel(io='test/Trade_Log.xlsx')
    trade_log = trade_log.set_index('Date').sort_index()[['Price', 'Amount']]
    assert max(abs(trds.Price.values - trade_log.Price.values)) < 1e-3
    assert trds.Amount.equals(trade_log.Amount.astype('float64'))

    # test equity
    eqty = rs_tr.get_equity()
    equity_log = pandas.read_excel(io='test/Equity_Log.xlsx', index_col=0)
    assert max(abs(eqty.Equity.values - equity_log.Equity.values)) < 1e-3
    assert numpy.max(numpy.abs(equity_log.Clo_Bal - rs_tr.get_equity().Closing_Balance)) < 1e-3
    assert numpy.max(numpy.abs(equity_log.Open_Profit - rs_tr.get_equity().Open_Profit)) < 1e-3

    # test performance indicators
    assert numpy.round(rs_tr.performance['ICAGR'], 4) == 0.0309
    assert numpy.round(rs_tr.performance['Max_Drawdown'], 4) == 0.4077
    assert numpy.round(rs_tr.performance['Bliss'], 4) == 0.0758




class Oracle_Trading_Strategy(Trading_Strategy):
    """
    Implements a trading system that cheats (!!!).
    """
    
    def init_state(self, trend, max_dd):
        """
        Parameters
        ----------
        a: exponential smoothing constant between 0 and 1. S_t = a * P_t + (1 - a) * S_t-1
        """

        self.state = {'Trend': trend}
        self.stop_loss = max_dd
    
    
    def update_state(self, today_i):
        """
        Updates the Resistance, Support, and Trend variables.
        """
        self.tr_stop_loss = self.stop_loss[today_i]
    
    
    def entry_order_prices(self, today_i):
        """
        Computes the entry orders for the day (today).
        """
        prev_day_position = self.equity['Position'][today_i - 1]
        state = {k: self.state[k][today_i - 1] for k in self.state.keys()}
        
        if prev_day_position == 0:
            if state['Trend'] == 1:
                self.orders['buy_stop'][today_i] = self.prices['High'][today_i - 1]
            elif state['Trend'] == -1:
                self.orders['sell_stop'][today_i] = self.prices['Low'][today_i - 1]
            else:
                pass
        else:
            pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class ES1_Trading_Strategy(Trading_Strategy):
    """
    Implements a simple exponential smoothing trading system.
    """
    
    def init_state(self, a=0.1):
        """
        Parameters
        ----------
        a: exponential smoothing constant between 0 and 1. S_t = a * P_t + (1 - a) * S_t-1
        """

        self.state = {'esa': numpy.zeros(len(self.dates)) * numpy.nan,
                      'Trend': numpy.zeros(len(self.dates)) * numpy.nan}
        
        self.a = min(max(a, 0), 1)
        
    
    
    def update_state(self, today_i):
        """
        Updates the Resistance, Support, and Trend variables.
        """
        if today_i == 1:
            self.state['esa'][0] = self.prices['Close'][0]
            self.state['Trend'][0] = 0
        else:        
            self.state['esa'][today_i - 1] = self.a * self.prices['Close'][today_i - 1] + \
                (1 - self.a) * self.state['esa'][today_i - 2]
            self.state['Trend'][today_i - 1] = numpy.sign(self.prices['Close'][today_i - 1] - \
                                                          self.state['esa'][today_i - 1])
    
    
    def entry_order_prices(self, today_i):
        """
        Computes the entry orders for the day (today).
        """
        prev_day_position = self.equity['Position'][today_i - 1]
        state = {k: self.state[k][today_i - 1] for k in self.state.keys()}
        
        if prev_day_position == 0:
            if state['Trend'] == 1:
                self.orders['buy_stop'][today_i] = self.prices['High'][today_i - 1]
            elif state['Trend'] == -1:
                self.orders['sell_stop'][today_i] = self.prices['Low'][today_i - 1]
            else:
                pass
        else:
            pass
    
    
    
    def protective_order_prices(self, today_i):
        """
        Computes the protective orders for the day (today) 
        """
        state = {k: self.state[k][today_i - 1] for k in self.state.keys()}
        prev_day_position = self.equity['Position'][today_i - 1]
        
        if self.orders['buy_stop'][today_i] > 0 or prev_day_position > 0:
            self.orders['protective_sell'][today_i] = state['esa']
            
        elif self.orders['sell_stop'][today_i] > 0 or prev_day_position < 0:
            self.orders['protective_buy'][today_i] = state['esa']
            
        else:
            pass
    


class ES2_Trading_Strategy(Trading_Strategy):
    """
    Implements a simple exponential smoothing trading system.
    """
    
    def init_state(self, a=0.1):
        """
        Parameters
        ----------
        a: exponential smoothing constant between 0 and 1. S_t = a * P_t + (1 - a) * S_t-1
        """

        self.state = {'esa': numpy.zeros(len(self.dates)) * numpy.nan,
                      'Trend': numpy.zeros(len(self.dates)) * numpy.nan}
        
        self.a = min(max(a, 0), 1)
    
    
    def update_state(self, today_i):
        """
        Updates the Resistance, Support, and Trend variables.
        """
        if today_i == 1:
            self.state['esa'][0] = self.prices['Close'][0]
            self.state['Trend'][0] = 0
        else:        
            self.state['esa'][today_i - 1] = self.a * self.prices['Close'][today_i - 1] + \
                (1 - self.a) * self.state['esa'][today_i - 2]
            self.state['Trend'][today_i - 1] = numpy.sign(self.state['esa'][today_i - 1] - \
                                                          self.state['esa'][today_i - 2])
    
    
    def entry_order_prices(self, today_i):
        """
        Computes the entry orders for the day (today).
        """
        prev_day_position = self.equity['Position'][today_i - 1]
        state = {k: self.state[k][today_i - 1] for k in self.state.keys()}
        
        if prev_day_position == 0:
            if state['Trend'] == 1:
                self.orders['buy_stop'][today_i] = self.prices['High'][today_i - 1]
            elif state['Trend'] == -1:
                self.orders['sell_stop'][today_i] = self.prices['Low'][today_i - 1]
            else:
                pass
        else:
            pass
        

        
        

def grid_search(price_df, name="",
                min_days=20, max_days=400, step=100, min_dif=50,
                warmup=None, tr_size=0.5, heat=0.05, equity=1e6,
                return_df=True):
    
    slow = numpy.arange(min_days, max_days, step)
    fast = numpy.arange(min_days, max_days, step)

    res_train = []
    res_val = []
    for s, f in itertools.product(slow, fast):
        if f > (s - 50): 
            continue
        else:
            # Training
            if warmup is None:
                warmup = s
            
            rs_train = RS_Trading_Strategy(price_df, equity=equity, heat=heat,
                                           days_fast=f, 
                                           days_slow=s,
                                           name=name)
            rs_train.excecute(warmup=warmup, end = tr_size)
            res_train.append({'Name': name, 'Slow': s, 'Fast': f,
                              'Type': 'Train', **rs_train.performance})

            # Validation
            rs_val = RS_Trading_Strategy(price_df, equity=equity, heat=heat,
                                         days_fast=f, 
                                         days_slow=s,
                                         name=name)
            rs_val.excecute(warmup=tr_size)
            res_val.append({'Name': name, 'Slow': s, 'Fast': f,
                            'Type': 'Validation', **rs_val.performance})


    if return_df:
        res_train = dict_list_to_DataFrame(res_train)
        res_val = dict_list_to_DataFrame(res_val)
        out = pandas.concat([res_train, res_val])
        return out
    else:
        return res_train, res_val
