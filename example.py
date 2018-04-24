# encoding: utf-8

from Analyzer import *

#get data and analyze
sa = StockAnalyzer()
#initialize with test saving path
sa.init_by_config('2018-03-01', '2018-03-22', 1000000, 600395, 'test_data/', 'test_output/')
#sa.analyze(by_config=True, pnl='test_data/pnl.csv', returns='test_data/returns.csv')
sa.analyze()

#get context for plotting
ctx = sa.get_context()

#instantiate GraphPlotter for generating HTML files
plotter = GraphPlotter()
#initialize with test saving path
plotter.init_from_config(ctx, 'test_report/')
plotter.plot()

#instantiate HTMLPlotter for generating HTML files
htmlplotter = HTMLPlotter()
#initialize with test saving path
htmlplotter.init_from_config(ctx, plotter.get_save_path())
htmlplotter.generate()