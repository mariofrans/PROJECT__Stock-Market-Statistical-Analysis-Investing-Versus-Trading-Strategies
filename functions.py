from variables import *

####################################################################################################################

# fetch stock data (already in pandas dataframe format)
def fetch_data(ticker, start_date, end_date):

    # search date is 12 months prior to start date 
    # to get prior data for indicator's calculations by start date
    search_date = start_date - relativedelta(months=12)
    df = yf.Ticker(ticker).history(start=search_date, end=end_date, auto_adjust=False)

    # filter data by OHLCV Columns only
    df = pd.DataFrame(df, columns= ['Open', 'High', 'Low', 'Close', 'Volume'])
    # remove empty data from every column
    df = df.dropna()
    
    return df

####################################################################################################################

def ma(data, period, type):

    # compute moving average of period=period
    if type=='simple': return list(data.rolling(period).mean())
    elif type=='exponential': return list(data.ewm(span=period).mean())
    else: print('MA Type Input Error')

####################################################################################################################

def bollinger_bands(data, period, type):

    # compute bollinger bands of period=period
    data_temp = pd.DataFrame()
    data_temp['BB-Mid'] = ma(data, period, type=type)
    data_temp['BB-STD'] = list(data.rolling(period).std())
    data_temp['BB-U'] = list(data_temp['BB-Mid'] + data_temp['BB-STD']*2)
    data_temp['BB-L'] = list(data_temp['BB-Mid'] - data_temp['BB-STD']*2)
    return [ list(data_temp['BB-Mid']), list(data_temp['BB-U']), list(data_temp['BB-L'])]

####################################################################################################################

def macd(data, macd_slow, macd_fast, macd_ema):

    # compute MACD using a fast and slow exponential moving average
    # return value is emaslow, emafast, macd which are in the form of arrays with len(data)
    data_temp = pd.DataFrame()
    data_temp['EMA-SLOW'] = list(data.ewm(span=macd_slow).mean())
    data_temp['EMA-FAST'] = list(data.ewm(span=macd_fast).mean())
    data_temp['MACD-L'] = data_temp['EMA-FAST'] - data_temp['EMA-SLOW']
    data_temp['MACD-S'] = list(data_temp['MACD-L'].ewm(span=macd_ema).mean())
    data_temp['MACD-H'] = data_temp['MACD-L'] - data_temp['MACD-S']
    return [ list(data_temp['MACD-L']), list(data_temp['MACD-S']), list(data_temp['MACD-H']) ]

####################################################################################################################

def rsi(data, period):

    # compute the relative strength indicator of period=period
    deltas = np.diff(data)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(data)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(data)):
        # cause the diff is 1 shorter
        delta = deltas[i - 1]  
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    return list(rsi)

####################################################################################################################

def stochastic(data, stochastic_period, stochastic_k, stochastic_d):

    # compute the stochastic K & D of period=stochastic_period
    min_val  = data.rolling(stochastic_period).min()
    max_val = data.rolling(stochastic_period).max()   
    stoch = ( (data - min_val) / (max_val - min_val) ) * 100
    K = stoch.rolling(stochastic_k).mean() 
    D = K.rolling(stochastic_d).mean() 
    return [ list(K), list(D) ]

####################################################################################################################

def calculate_indicators(df, periods):

    MA_1, MA_2, BB, RSI, STOCHASTIC, STOC_K, STOC_D, MACD_FAST, MACD_SLOW, MACD_EMA = periods

    # calculate indicator data
    if MA_1!=0: df[f'MA{MA_1}'] = ma(df['Close'], MA_1, 'simple')
    if MA_2!=0: df[f'MA{MA_2}'] = ma(df['Close'], MA_2, 'simple')
    if BB!=0: df[f'BB-M'], df[f'BB-U'], df[f'BB-L'] = bollinger_bands(df['Close'], BB, 'simple')
    if RSI!=0: df['RSI'] = rsi(df['Close'], RSI)
    if STOCHASTIC!=0: df['Stochastic-K'], df['Stochastic-D'] = stochastic(df['Close'], STOCHASTIC, STOC_K, STOC_D)    
    if MACD_FAST!=0:  df['MACD-L'], df['MACD-S'], df['MACD-H'] = macd(df['Close'], MACD_SLOW, MACD_FAST, MACD_EMA)

    return df

####################################################################################################################

def set_indicator_periods(  
                            MA_1=0, MA_2=0, BB=0, RSI=0, STOCHASTIC=0, STOC_K=0, STOC_D=0, 
                            MACD_FAST=0, MACD_SLOW=0, MACD_EMA=0  ):

    return [ MA_1, MA_2, BB, RSI, STOCHASTIC, STOC_K, STOC_D, MACD_FAST, MACD_SLOW, MACD_EMA ]

####################################################################################################################

def strategy__Investing(df, start_date, end_date):

    # compute the percentage change of stock price from the beginning to ending date (buy & hold returns)
    data = df['Close'].loc[start_date:end_date]
    # number of trades (buy + sell)
    trades = 2

    if len(data)!=0: return [round((data.iloc[len(data)-1] - data.iloc[0])/data.iloc[0] * 100, 2), trades]
    else: return [0, 0]
        
####################################################################################################################

def strategy__Price_MA_Crossover(df, start_date, end_date):

    MA_1 = MA_1__Price_MA_Crossover

    periods = set_indicator_periods(MA_1=MA_1)
    data = calculate_indicators(df[['Open', 'Close']], periods)
    data = data.loc[start_date:end_date]
    # print('\n', data)

    # create buying & selling signals based on indicator's conditions
    data_buy = data.loc[data['Close'] > data[f'MA{MA_1}']]
    data_sell = data.loc[data['Close'] < data[f'MA{MA_1}']]

    return trades_and_profit(data, data_buy, data_sell)

####################################################################################################################

def strategy__Price_2MA_Crossover(df, start_date, end_date):

    MA_1 = MA_1__Price_2MA_Crossover
    MA_2 = MA_2__Price_2MA_Crossover 

    periods = set_indicator_periods(MA_1=MA_1, MA_2=MA_2)
    data = calculate_indicators(df[['Open', 'Close']], periods)
    data = data.loc[start_date:end_date]
    # print('\n', data)

    # create buying & selling signals based on indicator's conditions
    data_buy = data.loc[    (data['Close'] > data[f'MA{MA_1}']) & 
                            (data[f'MA{MA_1}'] > data[f'MA{MA_2}'])
                        ]
    data_sell = data.loc[[i for i in list(data.index) if i not in list(data_buy.index)]]

    return trades_and_profit(data, data_buy, data_sell)

####################################################################################################################                

def strategy__2MA_Crossover(df, start_date, end_date):

    MA_1 = MA_1__2MA_Crossover
    MA_2 = MA_2__2MA_Crossover 

    periods = set_indicator_periods(MA_1=MA_1, MA_2=MA_2)
    data = calculate_indicators(df[['Open', 'Close']], periods)
    data = data.loc[start_date:end_date]
    # print('\n', data)

    # create buying & selling signals based on indicator's conditions
    data_buy = data.loc[data[f'MA{MA_1}'] > data[f'MA{MA_2}']]
    data_sell = data.loc[data[f'MA{MA_1}'] < data[f'MA{MA_2}']]

    return trades_and_profit(data, data_buy, data_sell)

####################################################################################################################                

def strategy__Bollinger_Bands(df, start_date, end_date):

    BB = BB__Bollinger_Bands

    periods = set_indicator_periods(BB=BB)
    data = calculate_indicators(df[['Open', 'Close']], periods)
    data = data.loc[start_date:end_date]
    # print('\n', data)

    # create buying & selling signals based on indicator's conditions
    data_buy = data.loc[(data[f'Close'] < data[f'BB-L']) | (data[f'Open'] < data[f'BB-L'])]
    data_sell = data.loc[(data[f'Close'] > data[f'BB-U']) | (data[f'Open'] > data[f'BB-U'])]

    return trades_and_profit(data, data_buy, data_sell)

####################################################################################################################                

def strategy__RSI(df, start_date, end_date):
    
    RSI = RSI__RSI

    periods = set_indicator_periods(RSI=RSI)
    data = calculate_indicators(df[['Open', 'Close']], periods)
    data = data.loc[start_date:end_date]
    # print('\n', data)

    # create buying & selling signals based on indicator's conditions
    data_buy = data.loc[data['RSI'] < 30]
    data_sell = data.loc[data['RSI'] > 70]

    return trades_and_profit(data, data_buy, data_sell)

####################################################################################################################                

def strategy__Stochastic(df, start_date, end_date):
    
    STOCHASTIC = STOCHASTIC__Stochastic
    STOC_K = STOC_K__Stochastic
    STOC_D = STOC_D__Stochastic

    periods = set_indicator_periods(STOCHASTIC=STOCHASTIC, STOC_K=STOC_K, STOC_D=STOC_D)
    data = calculate_indicators(df[['Open', 'Close']], periods)
    data = data.loc[start_date:end_date]
    # print('\n', data)

    # create buying & selling signals based on indicator's conditions
    data_buy = data.loc[(data['Stochastic-K'] < 20) & (data['Stochastic-K'] > data['Stochastic-D'])]
    data_sell = data.loc[(data['Stochastic-K'] > 80) & (data['Stochastic-K'] < data['Stochastic-D'])]

    return trades_and_profit(data, data_buy, data_sell)

####################################################################################################################

def strategy__MACD_Crossover(df, start_date, end_date):

    MACD_FAST = MACD_FAST__MACD_Crossover 
    MACD_EMA = MACD_EMA__MACD_Crossover
    MACD_SLOW = MACD_SLOW__MACD_Crossover

    periods = set_indicator_periods(MACD_FAST=MACD_FAST, MACD_EMA=MACD_EMA, MACD_SLOW=MACD_SLOW)                               
    data = calculate_indicators(df[['Open', 'Close']], periods)
    data = data.loc[start_date:end_date]
    # print('\n', data)

    # create buying & selling signals based on indicator's conditions
    data_buy = data.loc[data['MACD-H'] < 0]
    data_sell = data.loc[data['MACD-H'] > 0]

    return trades_and_profit(data, data_buy, data_sell)

####################################################################################################################

def trades_and_profit(data, data_buy, data_sell):

    # PREPROCESS SIGNALS

    # if both signals (buy + sell) are present for a trading day, make it 0 (neutral/hold)
    for i in list(data_buy.index): 
        if i in list(data_sell.index): 
            data_buy = data_buy.drop([i])
            data_sell = data_sell.drop([i])

    # set, concatenate, & sort signals (1 = Buy, -1 = Sell)
    data_buy['Signal'], data_sell['Signal'] = 1, -1
    data_signals = pd.concat([data_sell, data_buy])
    data_signals = data_signals.sort_index()
    # print('\n', data_signals)
    
    # add the signals to data
    # 0 if no signals are present (neutral/hold)
    data['Signal'] = 0
    for i in list(data_signals.index): data['Signal'][i] = data_signals['Signal'][i]
    data = data[['Open', 'Close', 'Signal']]
    # print('\n', data)

    # create & fill temporaty lists to store signals, its date, and current stock price
    dates, prices, signals = [], [], []
    datetimeindexes = list(data.index.date)

    # create trade signals
    for i in range(len(data)-2):
        signal = None
        if data['Signal'][i]==-1 and data['Signal'][i+1]==1: signal = 'Buy'
        elif data['Signal'][i]==1 and data['Signal'][i+1]==-1: signal = 'Sell'
        elif data['Signal'][i]==0 and data['Signal'][i+1]==1: signal = 'Buy'
        elif data['Signal'][i]==0 and data['Signal'][i+1]==-1: signal = 'Sell'

        if signal=='Buy' or signal=='Sell':
            dates.append(datetimeindexes[i+2])
            prices.append(data['Open'][i+2])
            signals.append(signal)

    # make sure that there are no same consecutive signal/s in a row
    if len(signals)>1:
        delete_indexes = []
        for i in range(len(signals)-1): 
            if signals[i]==signals[i+1]: 
                delete_indexes.append(i+1)

        # delete next signal if it is the same as the current signal
        if len(delete_indexes)>0:
            count = 0
            for i in delete_indexes:
                del dates[i-count]
                del prices[i-count]
                del signals[i-count]
                count += 1

    # remove first signal data if it is 'Sell'
    if len(signals)>0 and signals[0]=='Sell':
        del dates[0]
        del prices[0]
        del signals[0]

    # remove last signal data if it is 'Buy'
    if len(signals)>0 and signals[len(signals)-1]=='Buy':
        del dates[len(signals)-1]
        del prices[len(signals)-1]
        del signals[len(signals)-1]

    # create new df to store signals (display)
    df_signals = pd.DataFrame([dates, prices, signals])
    df_signals = df_signals.transpose()
    df_signals.columns = ['Date', 'Price', 'Signal']
    # print(df_signals)

    # create trades based on signals
    if len(df_signals)>0:
        
        entry_dates, exit_dates, entry_prices, exit_prices = [], [], [], []
        profits, profits_percentages, results = [], [], []

        for index in range(0, len(df_signals)-1, 2):

            entry_dates.append(df_signals['Date'][index])
            exit_dates.append(df_signals['Date'][index+1])
            entry_prices.append(round(df_signals['Price'][index], 2))
            exit_prices.append(round(df_signals['Price'][index+1], 2))

            profit = df_signals['Price'][index+1] - df_signals['Price'][index]
            profit_percentage = (profit * 100) / df_signals['Price'][index]

            profits.append(round(profit, 2))
            profits_percentages.append(round(profit_percentage, 2))

            if profit>0: results.append('Profit')
            elif profit<0: results.append('Loss')
            elif profit==0: results.append('Draw')

        # create new df to store trades (display)
        df_trades = pd.DataFrame([  entry_dates, exit_dates, entry_prices, exit_prices, 
                                    profits, profits_percentages, results   ])
        df_trades = df_trades.transpose()
        df_trades.columns = [   'Entry Date', 'Exit Date', 'Entry Price', 'Exit Price', 
                                'Profit', 'Profit (%)', 'Result'    ]
        # print(df_trades)

        # number of trades (buy + sell)
        trades = len(df_trades) * 2
        # print(trades)

        # calculate total profit for strategy in chosen trading period
        total_profit = round(sum(df_trades['Profit']), 2)
        total_profit_percentage = round( (total_profit / df_signals['Price'][0]) * 100, 2 )
        return [total_profit_percentage, trades]
    else: 
        return [0, 0]

####################################################################################################################