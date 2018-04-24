#coding=utf-8
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as tdelta
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
from pyh import *

TO_PCT = 100.0
#set global params
BEN_RATE = 0.04 #risk-free rate

class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y%m'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        """Return the label for time x at position pos"""
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''
        return pd.to_datetime(self.dates[ind], format="%Y-%m-%d").strftime(self.fmt)


class Analyzer(object):
    def __init__(self):
        self._data_path = None
        self._save_path = None

class StockAnalyzer(Analyzer):
    def __init__(self):
        super(StockAnalyzer, self).__init__()
        #Core data: daily stock pnl DataFrame
        #total_capital is 1, in case of 0 dividend. Assert every time when used
        self._total_capital = 1
        self._start_date = pd.Timestamp('1970-1-1')
        self._end_date = pd.Timestamp('1970-1-1')
        self._period = 1
        #daily positions and daily prices have the full list of tickers
        #daily trades may not involve all the stocks on board
        self._pre_positions = None
        self._daily_positions = None
        self._daily_trades = None
        self._daily_prices = None
        self._benchmark = 0
        self._benchmark_price = 0
        self._benchmark_price_list = []
        #pnl counting starts from 1
        self._date_sequence = []
        self._daily_pnl = [0]
        self._accum_pnl = [0]
        self._benchmark_pnl = [0]
        self._accum_benchmark_pnl = [0]
        self._daily_retrace = [0]
        self._max_retrace = 0
        self._max_retrace_start = '1970-1-1'
        self._max_retrace_end = '1970-1-1'
        self._daily_return = [0]
        self._accum_return = [0]
        self._annual_return = 0
        self._benchmark_return = [0]
        self._accum_benchmark_return = [0]
        self._benchmark_annual_return = 0
        self._excess_accum_return = [0]
        self._holding_pnl = None
        self._trading_pnl = None

        #stats
        self._volatility = 0
        self._benchmark_volatility = 0
        self._alpha = 0
        self._beta = 0
        self._sharpe_ratio = 0
        #benchmark prices
        self._irrelevant_position_params = ['long_avail', 'short_total', 'short_avail', \
                                            'short_yestd', 'net_total', 'net_avail', \
                                            'net_yestd']
        self._irrelevant_trade_params = ['order_id', 'exchange_id', 'offset']

    def init_by_config(self, start_date, end_date, capital, benchmark, data_path='Data/test_data/', save_path='Output/'):
        if data_path[-1] == '/':
            self._data_path = data_path
        else:
            self._data_path = data_path + '/'
        if save_path[-1] == '/':
            self._save_path = save_path
        else:
            self._save_path = save_path + '/'
        self._start_date = pd.Timestamp(start_date)
        self._end_date = pd.Timestamp(end_date)
        times = start_date.split('-')
        self._date_sequence.append(pd.Timestamp(dt(int(times[0]), int(times[1]), int(times[2])) + tdelta(days=-1)))
        for date in pd.date_range(self._start_date, self._end_date, freq='D'):
            date_str = str(date).split(' ')[0]
            if self._has_date(date_str):
                self._date_sequence.append(date_str)
        self._period = len(self._date_sequence) - 1
        self._total_capital = capital
        self._benchmark = benchmark

    def _has_date(self, date):
        return os.path.exists(self._data_path + date + '_trades.csv')

    def _get_daily_data(self, date):
        self._daily_positions = pd.read_csv(self._data_path + date + '_position.csv')
        self._daily_trades = pd.read_csv(self._data_path + date + '_trades.csv')
        self._daily_prices = pd.read_csv(self._data_path + date + '_price.csv')

    def _preprocess_daily_data(self):
        #step 1 remove irrelevant columns
        for i in self._irrelevant_position_params:
            self._daily_positions.pop(i)
        self._daily_positions = self._daily_positions.groupby('ticker').sum()
        #print self._daily_positions

        for j in self._irrelevant_trade_params:
            self._daily_trades.pop(j)

        #print self._daily_trades
        #step 2 calculate
        self._daily_trades['volume'] = ((-1)**self._daily_trades['direction']) *\
                                       self._daily_trades['volume']
        self._daily_trades['mul'] = self._daily_trades['price'] * \
                                    self._daily_trades['volume']

        self._daily_trades = self._daily_trades.groupby('ticker').sum()
        #print self._daily_trades

        self._daily_prices = self._daily_prices.groupby('ticker').sum()
        #print self._daily_prices

    #fill 0 to NAN values
    def _fill_null_value(self):
        pass

    def _cal_daily_pnl(self):
        #calculate daily pnl
        #formula is pnl = (v0 + Σvti)p1 - p0v0 - Σvtipti
        daily_pnl = 0
        for ticker in self._daily_positions.index:
            #get v0, p0, p1
            if self._pre_positions is None:
                v0 = 0
            else:
                v0 = self._pre_positions.loc[ticker]['long_total']
            p0 = self._daily_prices.loc[ticker]['pre_close_price']
            p1 = self._daily_prices.loc[ticker]['last_price']
            try:
                Sumv = self._daily_trades.loc[ticker]['volume']
                Sumvp = self._daily_trades.loc[ticker]['mul']
            except:
                Sumv = 0
                Sumvp = 0
            daily_pnl += (v0 + Sumv) * p1 - p0 * v0 - Sumvp
            #print self._daily_prices.loc[indexes]['ticker']
        self._pre_positions = self._daily_positions
        #print type(daily_pnl)
        return daily_pnl

    def _cal_accum_pnl(self, pnl):
        return self._accum_pnl[-1] + pnl

    def _cal_benchmark_pnl(self):
        #daily benchmark pnl formula: pnl = ((p1 - p0) * capital) / p
        return ((self._daily_prices.loc[self._benchmark]['last_price'] - self._daily_prices.loc[self._benchmark]['pre_close_price']) * self._total_capital) / self._benchmark_price

    def _cal_benchmark_accum_pnl(self, pnl):
        return self._accum_benchmark_pnl[-1] + pnl

    def _cal_daily_return(self, pnl):
        return pnl / (self._total_capital + self._accum_pnl[-2])

    def _cal_accum_return(self, ret):
        return self._accum_return[-1] + ret

    def _cal_annual_return(self):
        # portfolio annual return formula: ((1 + p)**(250/n) - 1)
        return (1 + self._accum_return[-1]) ** (250/self._period) - 1

    def _cal_benchmark_return(self, pnl):
        return pnl / (self._total_capital + self._accum_benchmark_pnl[-2])

    def _cal_accum_benchmark_return(self, ret):
        return self._accum_benchmark_return[-1] + ret

    def _cal_benchmark_annual_return(self):
        # benchmark annual return formula: ((1 + p)**(250/n) - 1)
        return (1 + self._accum_benchmark_return[-1]) ** (250 / self._period) - 1

    def _cal_excess_accum_return(self):
        return self._accum_return[-1] - self._accum_benchmark_return[-1]

    def _cal_max_retrace(self):
        pre = 0
        pivot = 0
        start = '1970-1-1'
        end = '1970-1-1'

        for index in range(len(self._daily_pnl)):
            pnl = self._daily_pnl[index]
            sub = pnl - pre
            if sub < pivot:
                pivot = sub
                start = self._date_sequence[index - 1]
                end = self._date_sequence[index]
            self._daily_retrace.append(sub)
            pre = pnl
        #print self._daily_retrace
        #print pivot
        return pivot, start, end

    def _cal_portfolio_volatility(self):
        # portfolio volatility formula: sqrt(250) * stddev(daily_return)
        return np.std(np.array(self._daily_return)) * np.sqrt(250)

    def _cal_benchmark_volatility(self):
        # portfolio volatility formula: sqrt(250) * stddev(benchmark_return)
        return np.std(np.array(self._benchmark_return)) * np.sqrt(250)

    def _cal_beta(self):
        #print np.cov(self._benchmark_return, self._daily_return)[0][1]
        #print np.array(self._benchmark_return).var()
        return np.cov(self._benchmark_return, self._daily_return)[0][1] / np.array(self._benchmark_return).var()

    def _cal_alpha(self):
        return self._annual_return - (BEN_RATE + self._beta * (self._benchmark_annual_return - BEN_RATE))

    def _cal_sharpe_ratio(self):
        return ((self._annual_return - BEN_RATE) / self._cal_portfolio_volatility())

    def _generate_pnl_and_returns_from_data(self):
        first_day = True
        for date in self._date_sequence[1:]:
            self._get_daily_data(date)
            self._preprocess_daily_data()
            if first_day:
                self._benchmark_price = self._daily_prices.loc[self._benchmark]['pre_close_price']
                self._benchmark_price_list.append(self._daily_prices.loc[self._benchmark]['pre_close_price'])
                if (self._benchmark_price == 0):
                    raise ZeroDivisionError
                first_day = False
            self._benchmark_price_list.append(self._daily_prices.loc[self._benchmark]['last_price'])
            pnl = self._cal_daily_pnl()
            self._daily_pnl.append(pnl)
            benchmark_pnl = self._cal_benchmark_pnl()
            self._benchmark_pnl.append(benchmark_pnl)
            # Pre-close total pnl of deposit is at accum_return[-2]
            accum_pnl = self._cal_accum_pnl(pnl)
            self._accum_pnl.append(accum_pnl)
            accum_benchmark_pnl = self._cal_benchmark_accum_pnl(benchmark_pnl)
            self._accum_benchmark_pnl.append(accum_benchmark_pnl)
            ret = self._cal_daily_return(pnl)
            self._daily_return.append(ret)
            accum_return = self._cal_accum_return(ret)
            self._accum_return.append(accum_return)
            benchmark_ret = self._cal_benchmark_return(benchmark_pnl)
            self._benchmark_return.append(benchmark_ret)
            accum_benchmark_return = self._cal_accum_benchmark_return(benchmark_ret)
            self._accum_benchmark_return.append(accum_benchmark_return)
            excess = self._cal_excess_accum_return()
            self._excess_accum_return.append(excess)

    def _generate_pnl_and_returns_from_config(self, pnl, returns):
        # use existing pnl and returns files
        pnl = pd.read_csv(pnl)
        returns = pd.read_csv(returns)
        self._daily_pnl += pnl['strategy_daily_pnl'].tolist()
        self._accum_pnl += pnl['strategy_accumulated_pnl'].tolist()
        self._benchmark_pnl += pnl['benchmark_daily_pnl'].tolist()
        self._accum_benchmark_pnl += pnl['benchmark_accumulated_pnl'].tolist()

        self._daily_return += returns['strategy_daily_return'].tolist()
        self._benchmark_return += returns['benchmark_daily_return'].tolist()
        for index in range(len(self._daily_return)):
            if index == 0:
                self._accum_return[index] = self._daily_return[index]
                self._accum_benchmark_return[index] = self._benchmark_return[index]
                self._excess_accum_return[index] = self._daily_return[index] - self._benchmark_return[index]
            else:
                self._accum_return.append(self._accum_return[index - 1] + self._daily_return[index])
                self._accum_benchmark_return.append(self._accum_benchmark_return[index - 1] + self._benchmark_return[index])
                self._excess_accum_return.append(self._accum_return[index] - self._accum_benchmark_return[index])

    def analyze(self, by_config=False, pnl='', returns=''):

        if by_config:
            self._generate_pnl_and_returns_from_config(pnl, returns)
        else:
            self._generate_pnl_and_returns_from_data()

        #calculate risk matrics
        self._max_retrace, self._max_retrace_start, self._max_retrace_end = self._cal_max_retrace()
        self._annual_return = self._cal_annual_return()
        self._benchmark_annual_return = self._cal_benchmark_annual_return()
        self._volatility = self._cal_portfolio_volatility()
        self._benchmark_volatility = self._cal_benchmark_volatility()
        self._beta = self._cal_beta()
        self._alpha = self._cal_alpha()
        self._sharpe_ratio = self._cal_sharpe_ratio()

        #self._cal_annual_return()

        self.pnl_to_csv()
        self.return_to_csv()

    def pnl_to_csv(self):
        df = pd.DataFrame({'strategy_daily_pnl': self._daily_pnl[1:],\
                           'strategy_accumulated_pnl': self._accum_pnl[1:],\
                          'benchmark_daily_pnl': self._benchmark_pnl[1:],\
                          'benchmark_accumulated_pnl': self._accum_benchmark_pnl[1:]}, self._date_sequence[1:])
        df.to_csv(self._save_path + 'pnl.csv')

    def return_to_csv(self):
        df = pd.DataFrame({'strategy_daily_return': self._daily_return[1:],\
                           'benchmark_daily_return': self._benchmark_return[1:],\
                           'strategy_accumulated_return': self._accum_return[1:],\
                           'benchmark_accumulated_return': self._accum_benchmark_return[1:]}, self._date_sequence[1:])
        df.to_csv(self._save_path + 'return.csv')

    def get_context(self):
        return {'date_sequence': self._date_sequence, 'daily_pnl': self._daily_pnl, 'accum_pnl': self._accum_pnl,\
                'daily_retrace': self._daily_retrace, 'daily_return': self._daily_return,\
                'annual_return': self._annual_return, 'accum_return': self._accum_return,\
                'benchmark_return': self._benchmark_return, 'accum_benchmark_return': self._accum_benchmark_return,\
                'benchmark_annual_return': self._benchmark_annual_return,\
                'excess_accum_return': self._excess_accum_return,\
                'max_retrace': self._max_retrace, 'max_retrace_start': self._max_retrace_start,\
                'max_retrace_end': self._max_retrace_end,\
                'beta': self._beta, 'alpha': self._alpha, 'volatility': self._volatility, 'benchmark_volatility': self._benchmark_volatility,\
                'sharpe_ratio': self._sharpe_ratio}

class Plotter(object):
    def __init__(self):
        self._context = None
        self._save_path = None

class GraphPlotter(Plotter):
    def __init__(self):
        super(GraphPlotter, self).__init__()
        self._pnl = None
        self._return = None

    def init_from_config(self, ctx, path='Report/'):
        self._context = ctx
        if path[-1] == '/':
            self._save_path = path
        else:
            self._save_path = path + '/'

    def plot_pnl(self):
        total = pd.Series(self._context['daily_pnl'][1:], self._context['date_sequence'][1:])
        accum = pd.Series(self._context['accum_pnl'][1:], self._context['date_sequence'][1:])
        #print total
        #print accum
        idx0 = total.index
        n = len(idx0)
        idx = np.arange(n)

        fig, (ax0) = plt.subplots(1, 1, figsize=(16, 13.5), sharex=True)
        ax1 = ax0.twinx()

        bar_width = 0.4
        profit_color, lose_color = '#D63434', '#2DB635'
        curve_color = '#174F67'
        y_label = 'Profit / Loss ($)'
        color_arr_raw = np.array([profit_color] * n)

        color_arr = color_arr_raw.copy()
        color_arr[total < 0] = lose_color
        ax0.bar(idx, total, width=bar_width, color=color_arr)
        ax0.set(title='Daily PnL', ylabel=y_label, xlim=[-2, n + 2], )
        ax0.xaxis.set_major_formatter(MyFormatter(idx0, '%y-%m-%d'))

        ax1.plot(idx, accum, lw=1.5, color=curve_color)
        ax1.set(ylabel='Cum. ' + y_label)
        ax1.yaxis.label.set_color(curve_color)

        plt.savefig(self._save_path + 'pnl.png')
        return fig

    def plot_returns(self):
        """
        Parameters
        ----------
        Series

        """
        portfolio_cum_ret = pd.Series(self._context['accum_return'][1:], self._context['date_sequence'][1:])

        benchmark_cum_ret = pd.Series(self._context['accum_benchmark_return'][1:], self._context['date_sequence'][1:])
        excess_cum_ret = pd.Series(self._context['excess_accum_return'][1:], self._context['date_sequence'][1:])
        max_dd_start = self._context['max_retrace_start']
        max_dd_end = self._context['max_retrace_end']

        n_subplots = 3
        fig, (ax1, ax2, ax3) = plt.subplots(n_subplots, 1, figsize=(16, 4.5 * n_subplots), sharex=True)
        idx_dt = portfolio_cum_ret.index
        idx = np.arange(len(idx_dt))

        y_label_ret = "Cumulative Return (%)"

        ax1.plot(idx, (benchmark_cum_ret - 1) * TO_PCT, label='Benchmark', color='#174F67')
        ax1.plot(idx, (portfolio_cum_ret - 1) * TO_PCT, label='Strategy', color='#198DD6')
        ax1.legend(loc='upper left')
        ax1.set(title="Absolute Return of Portfolio and Benchmark",
                # xlabel="Date",
                ylabel=y_label_ret)
        ax1.grid(axis='y')

        ax2.plot(idx, (excess_cum_ret - 1) * TO_PCT, label='Extra Return', color='#C37051')
        ax2.axvspan(idx_dt.get_loc(max_dd_start), idx_dt.get_loc(max_dd_end), color='lightgreen', alpha=0.5,
                    label='Maximum Drawdown')
        ax2.legend(loc='upper left')
        ax2.set(title="Excess Return Compared to Benchmark", ylabel=y_label_ret
                # xlabel="Date",
                )
        ax2.grid(axis='y')
        ax2.xaxis.set_major_formatter(MyFormatter(idx_dt, '%y-%m-%d'))  # 17-09-31

        ax3.plot(idx, (portfolio_cum_ret) / (benchmark_cum_ret), label='Ratio of NAV', color='#C37051')
        ax3.legend(loc='upper left')
        ax3.set(title="NaV of Portfolio / NaV of Benchmark", ylabel=y_label_ret
                # xlabel="Date",
                )
        ax3.grid(axis='y')
        ax3.xaxis.set_major_formatter(MyFormatter(idx_dt, '%y-%m-%d'))  # 17-09-31

        fig.tight_layout()
        plt.savefig(self._save_path + 'returns.png')
        return fig

    #default plotting all graphs
    def plot(self):
        self.plot_pnl()
        self.plot_returns()

    def get_save_path(self):
        return self._save_path

class HTMLPlotter(Plotter):
    def __init__(self):
        super(HTMLPlotter, self).__init__()
        self._page = None

    #data_path is the path where graphs generated by GraphPlotter are stored
    def init_from_config(self, ctx, save_path='Report/', page_title = 'Analysis Page'):
        if save_path[-1] == '/':
            self._save_path = save_path
        else:
            self._save_path = save_path + '/'
        self._context = ctx
        self._page = PyH(page_title)

    def format(self):
        self._page << h1('PnL Curves', cl='main_header')
        self._pnl_graph = self._page << div(cl='graph_area', id='pnl_area') << img(src='pnl.png', id ='pnl_graph')
        self._page << h1('Return Curves', cl='main_header')
        self._ret_graph = self._page << div(cl='graph_area', id='return_area') << img(src='returns.png', id='return_graph')
        self._page << h1('Performance Matrics ', cl='main_header')
        self._index_table = self._page << div(cl='index_area', id='risk_index_area') << table()
        self._annual_return = self._index_table << tr() << td('Annual Return') + span() + td(self._context['annual_return']) + br()
        self._benchmark_annual_return = self._index_table << tr() << td('Benchmark Annual Return') + span() + td(self._context['benchmark_annual_return']) + br()
        self._alpha = self._index_table << tr() << td('Alpha') + span() + td(self._context['alpha']) + br()
        self._beta = self._index_table << tr() << td('Beta') + span() + td(self._context['beta']) + br()
        self._sharpe_ratio = self._index_table << tr() << td('Sharpe Ratio') + span() + td(self._context['sharpe_ratio']) + br()
        self._max_retrace = self._index_table << tr() << td('Max Retrace') + span() + td(self._context['max_retrace']) + br()
        self._volatility = self._index_table << tr() << td('Portfolio Volatility') + span() + td(self._context['volatility']) + br()
        self._benchmark_volatility = self._index_table << tr() << td('Benchmark Volatility') + span() + td(self._context['benchmark_volatility']) + br()

    def write(self, path=''):
        if path == '':
            path = self._save_path
        elif path[-1] != '/':
            path += '/'
        self._page.printOut(path + 'report.html')

    #default generating html
    def generate(self):
        self.format()
        self.write()
