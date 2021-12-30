####################################################################################################################

import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import ttest_ind

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

import time
from datetime import date
from dateutil.relativedelta import relativedelta

# remove pandas warnings
pd.set_option('mode.chained_assignment', None)

####################################################################################################################

""" STRATEGY INDICATOR VARIABLES """

MA_1__Price_MA_Crossover = 10

MA_1__Price_2MA_Crossover = 10
MA_2__Price_2MA_Crossover = 100

MA_1__2MA_Crossover = 20
MA_2__2MA_Crossover = 50

BB__Bollinger_Bands = 20

RSI__RSI = 14

STOCHASTIC__Stochastic = 14
STOC_K__Stochastic = 3
STOC_D__Stochastic = 3

MACD_FAST__MACD_Crossover = 12
MACD_EMA__MACD_Crossover = 26
MACD_SLOW__MACD_Crossover = 9

####################################################################################################################

PATH_TICKERS = 'BINUS Sem 5/Data Science/FP StockMarket/data/tickers.csv'
PATH_FIGURE_1 = 'BINUS Sem 5/Data Science/FP StockMarket/figure-1.jpeg'
PATH_FIGURE_2 = 'BINUS Sem 5/Data Science/FP StockMarket/figure-2.jpeg'
PATH_DATA_RETURNS = 'BINUS Sem 5/Data Science/FP StockMarket/data/returns.csv'
PATH_DATA_TRADES = 'BINUS Sem 5/Data Science/FP StockMarket/data/trades.csv'

####################################################################################################################

# https://money.kompas.com/read/2021/11/21/153813826/simak-ini-rincian-biaya-jual-beli-saham-yang-perlu-kamu-tahu?page=all#:~:text=Biaya%20komisi%20broker%20biasanya%20sekitar,setiap%20transaksi%20jual%20beli%20saham.

####################################################################################################################
