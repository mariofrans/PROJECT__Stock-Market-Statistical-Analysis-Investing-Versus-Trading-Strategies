from numpy.core.numeric import NaN
from variables import *
from functions import *

####################################################################################################################

# start timer (to calculate execution time)
start_time = time.time()

# list of all Indonesian Blue-Chip stocks
df_tickers_indo_bluechip = pd.read_csv(PATH_TICKERS)
tickers_indo_bluechip = list(df_tickers_indo_bluechip['Ticker-YF'])

####################################################################################################################

""" COLLECT HISTORICAL DATA & COMPUTE TRADE RESULTS, FOR EACH STRATEGY ON EACH TICKER'S DATA """

def backtest_strategies(tickers, backtest_years):

    months = backtest_years * 12
    end_date = date.today()
    start_date = end_date - relativedelta(months=months)

    # collect market data for each ticker
    dfs = [fetch_data(ticker, start_date, end_date) for ticker in tickers]

    # columns of final dataframe
    columns = [ 'Months', 'Start Dates', 'End Dates', 'Investing', 'Price-MA Cross', 'Price-2MA Cross', 
                '2MA Cross', 'Bollinger Bands', 'RSI Strategy', 'Stochastic', 'MACD Cross']
    # create empty summary dataframes
    df_summary_returns, df_summary_trades = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    for i in range(1, months+1):

        start_date_temp = end_date - relativedelta(months=i)
    
        # create & display new df to store strategy results (returns & number of trades) for each ticker
        df_results = pd.DataFrame()
        df_results['Ticker'] = tickers
        df_results['Investing'] = [strategy__Investing(df, start_date_temp, end_date) for df in dfs]
        df_results['Price-MA Cross'] = [strategy__Price_MA_Crossover(df, start_date_temp, end_date) for df in dfs]
        df_results['Price-2MA Cross'] = [strategy__Price_2MA_Crossover(df, start_date_temp, end_date) for df in dfs]
        df_results['2MA Cross'] = [strategy__2MA_Crossover(df, start_date_temp, end_date) for df in dfs]
        df_results['Bollinger Bands'] = [strategy__Bollinger_Bands(df, start_date_temp, end_date) for df in dfs]
        df_results['RSI Strategy'] = [strategy__RSI(df, start_date_temp, end_date) for df in dfs]
        df_results['Stochastic'] = [strategy__Stochastic(df, start_date_temp, end_date) for df in dfs]
        df_results['MACD Cross'] = [strategy__MACD_Crossover(df, start_date_temp, end_date) for df in dfs]
        df_results = df_results.set_index('Ticker')
        # print(df_results)

        # split df_results into trade returns, and number of trades dataframes respectively
        df_returns, df_trades = pd.DataFrame(), pd.DataFrame()
        for column in list(df_results.columns):
            df_temp = pd.DataFrame(df_results[column].tolist(), index=df_results.index)
            df_returns[column], df_trades[column] = df_temp[0], df_temp[1]
        # print(df_returns)
        # print(df_trades)

        # create lists of summarised data for each trading strategy within the time frame
        summary_returns = [i, start_date_temp.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y')]
        summary_trades = [i, start_date_temp.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y')]
        for column in [i for i in columns if i not in ['Months', 'Start Dates', 'End Dates']]:
            summary_returns.append(round(df_returns[column].mean(), 2))
            summary_trades.append(round(df_trades[column].mean(), 2))

        # convert list into series
        summary_returns = pd.Series(summary_returns, index=df_summary_returns.columns)
        summary_trades = pd.Series(summary_trades, index=df_summary_trades.columns)

        # append returns and results data to main summary dataframes
        df_summary_returns = df_summary_returns.append(summary_returns, ignore_index=True)
        df_summary_trades = df_summary_trades.append(summary_trades, ignore_index=True)
        print(df_summary_returns)
        print(df_summary_trades)
        
        # save summary dataframe into their respective csv files
        df_summary_returns.set_index('Months').to_csv(PATH_DATA_RETURNS)
        df_summary_trades.set_index('Months').to_csv(PATH_DATA_TRADES)

# backtest_strategies(tickers_indo_bluechip, 10)

####################################################################################################################

""" PREPROCESS COLLECTED DATA """

def preprocess_data(df_returns, df_trades):

    df = pd.DataFrame()

    # list of all trading strategies
    strategies = [col for col in list(df_returns.columns) if col not in ['Months', 'Start Dates', 'End Dates']]

    for strategy in strategies:

        # create empty temp df
        df_strategy = pd.DataFrame()
        # add strategy's detail's columns
        df_strategy['Months'] = df_returns['Months']
        df_strategy['Trades'] = df_trades[strategy]
        df_strategy['Returns'] = df_returns[strategy]
        df_strategy['Strategy'] = strategy
        # add strategy df to main df
        df = pd.concat([df, df_strategy])
        # sort df values
        df = df.sort_values(by=['Strategy', 'Months'])

    # correct indexes
    df.index = [i for i in range(len(df))]

    return df

def minus_brokerage_fees(df, fee_buy=0.16, fee_sell=0.26):

    brokerage_fee = fee_buy + fee_sell
    # calculate returns after fees
    df['Returns Including Fees'] = df['Returns'] - ( (df['Trades'] / 2) * brokerage_fee )
    # rearrange columns
    df = df[['Months', 'Trades', 'Returns', 'Returns Including Fees', 'Strategy']]

    return df

def get_df_average_returns(df):

    df_average_returns = pd.DataFrame()
    strategies = df['Strategy'].unique()
    list_average_returns = []

    for strategy in strategies:
        df_strategy = df.loc[df['Strategy']==strategy]
        df_strategy = df_strategy.loc[df_strategy['Months']%12 == 0]
        df_strategy['Returns Including Fees Changes'] = df_strategy['Returns Including Fees'].diff()
        df_strategy['Returns Including Fees Changes'].iloc[0] = df_strategy['Returns Including Fees'].iloc[0]
        print(df_strategy)
        average_annual_return_including_fees = df_strategy['Returns Including Fees Changes'].mean()
        list_average_returns.append(round(average_annual_return_including_fees, 2))

    df_average_returns['Average Annual Returns Including Fees'] = list_average_returns
    df_average_returns.index = strategies
    df_average_returns = df_average_returns.sort_values('Average Annual Returns Including Fees', ascending=False)

    return df_average_returns

df_returns = pd.read_csv(PATH_DATA_RETURNS)
df_trades = pd.read_csv(PATH_DATA_TRADES)
# print(df_returns)
# print(df_trades)

df = preprocess_data(df_returns, df_trades)
print(df)
df = minus_brokerage_fees(df)
print(df)
df_average_returns = get_df_average_returns(df)
print(df_average_returns)

####################################################################################################################

def visualise_data(df, path_1, path_2):

    strategies = [  'Investing', 'Price-MA Cross', 'Price-2MA Cross', 'Bollinger Bands', 
                    '2MA Cross', 'RSI Strategy', 'Stochastic', 'MACD Cross'   ]
    dfs = [df.loc[df['Strategy']==strategy] for strategy in strategies]

    figure_title = 'Average Returns of all Indonesian Blue Chip Stock\nbased on Trading/Investing Strategy\n'

    """ PLOT ALL STRATEGIES RETURNS IN A SINGLE CHART ON THE FIRST GRAPH """

    # set figure & subplots
    fig, axs = plt.subplots(figsize=(10, 7))

    for strategy in strategies: 
        
        strategy_index = strategies.index(strategy)
        df = dfs[strategy_index]
        x_label = 'Months'
        y_label = 'Returns Including Fees'

        x = np.array(df[x_label]).reshape(-1, 1)
        y = np.array(df[y_label])

        axs.plot(x, y, label=strategy)
        axs.legend()
    
    fig.suptitle(f'{figure_title}({y_label})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path_1)

    """ PLOT ALL STRATEGIES RETURNS IN A RESPECTIVE CHART ON THE SECOND GRAPH """

    # set figure & subplots
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))

    # create empty lists to store linear regression data
    coefficients, intercepts = [], []

    for strategy in strategies: 

        strategy_index = strategies.index(strategy)
        df = dfs[strategy_index]
        x_label = 'Months'
        y_labels = ['Returns', 'Returns Including Fees']

        if strategy_index>3: row, col = (strategy_index - 4), 1
        else: row, col = strategy_index, 0

        for y_label in y_labels:

            x = np.array(df[x_label]).reshape(-1, 1)
            y = np.array(df[y_label])
            
            axs[row, col].set_title(strategy)
            axs[row, col].plot(x, y, label=y_label)
            axs[row, col].axhline(0, color='red', linestyle='--')

            if y_label=='Returns Including Fees':

                model = LinearRegression()
                model.fit(x, y)
                axs[row, col].plot(x, model.predict(x), label='Linear Regression')

                coefficients.append(float(model.coef_))
                intercepts.append(float(model.intercept_))

            axs[row, col].legend()

    fig.suptitle(figure_title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path_2)

    # display collected linear regression data
    df_strategies = pd.DataFrame([strategies, coefficients, intercepts]).transpose()
    df_strategies.columns = ['Strategies', 'Coeffecients', 'Intercepts']
    print(df_strategies)

# visualise_data(df, PATH_FIGURE_1, PATH_FIGURE_2)

####################################################################################################################

def get_t_test(df):

    strategies = df['Strategy'].unique()
    df_t_statistics, df_p_values = pd.DataFrame(), pd.DataFrame()
    
    for x in strategies:

        list_t, list_p = [], []

        for y in strategies:

            df_1 = df.loc[df['Strategy']==x]['Returns']
            df_2 = df.loc[df['Strategy']==y]['Returns']
            t_statistic, p_value = ttest_ind(df_1, df_2)
            list_t.append(t_statistic)
            list_p.append(p_value)

        df_t_statistics[x] = list_t
        df_p_values[x] = list_p

    df_t_statistics.index = strategies
    df_p_values.index = strategies

    return df_t_statistics, df_p_values

# print('\n1 Month:')
# df_temp = df.loc[df['Months']==1]
# # print(df_temp)
# df_t_statistics, df_p_values = get_t_test(df_temp)
# print(df_t_statistics)
# print(df_p_values)

# print('\n3 Month:')
# df_temp = df.loc[df['Months']==3]
# # print(df_temp)
# df_t_statistics, df_p_values = get_t_test(df_temp)
# print(df_t_statistics)
# print(df_p_values)

# print('\n6 Month:')
# df_temp = df.loc[df['Months']==6]
# # print(df_temp)
# df_t_statistics, df_p_values = get_t_test(df_temp)
# print(df_t_statistics)
# print(df_p_values)

# print('\n9 Month:')
# df_temp = df.loc[df['Months']==9]
# # print(df_temp)
# df_t_statistics, df_p_values = get_t_test(df_temp)
# print(df_t_statistics)
# print(df_p_values)

# print('\n12 Month:')
# df_temp = df.loc[df['Months']==12]
# # print(df_temp)
# df_t_statistics, df_p_values = get_t_test(df_temp)
# print(df_t_statistics)
# print(df_p_values)

# print('\n15 Month:')
# df_temp = df.loc[df['Months']==15]
# # print(df_temp)
# df_t_statistics, df_p_values = get_t_test(df_temp)
# print(df_t_statistics)
# print(df_p_values)

####################################################################################################################

""" IN-SAMPLE & OUT-OF-SAMPLE PREDICTIONS """

# def visualise_data(df):

#     strategies = [i for i in df.columns if i not in ['Start Dates', 'End Dates']]

#     # set figure & subplots
#     plt.figure(figsize=(10, 20))
#     gs = gridspec.GridSpec(9, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1, 1, 1])
#     ax = [plt.subplot(gs[0])]
#     ax.extend([plt.subplot(gs[i+1], sharex=ax[0]) for i in range(8)])

#     for column in strategies: 

#         x = np.array(df.index).reshape(-1, 1)
#         y = np.array(df[column])

#         # plot strategy data on main chart
#         ax[0].plot(x, y, label=column)
#         # plot strategy data on respective chart
#         ax[strategies.index(column) + 1].plot(x, y, label=column)
#         # plot zero line on respective chart (to determine positive & negative)
#         ax[strategies.index(column) + 1].axhline(0, color='red', linestyle='--')

#         # train test split
#         x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

#         # generate & fit model
#         model = SVR(kernel='linear')
#         # model = LinearRegression()
#         model.fit(x_train, y_train)

#         y_pred_train = model.predict(x_train)
#         y_pred_train = list(y_pred_train) + [None for i in x_test]
#         ax[strategies.index(column) + 1].plot(x, y_pred_train, label='In-Sample Forecast')

#         y_pred_test = model.predict(x_test)
#         y_pred_test = [None for i in y_train] + list(y_pred_test)
#         ax[strategies.index(column) + 1].plot(x, y_pred_test, label='Out-of-Sample Forecast')
        
#     for i in range(9): ax[i].legend()
#     ax[0].set_title('Average Returns of all Indonesian Blue Chip Stock\nbased on Trading/Investing Strategy\n', 
#                     fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(PATH_FIGURE)

####################################################################################################################

# df_summary = pd.read_csv(PATH_CSV)
# df_summary = df_summary.set_index('Months')
# df_summary = df_summary.fillna(0)
# print(df_summary)

# visualise_data(df_summary)

####################################################################################################################

# calculate execution time
print("Execution time: %s seconds" % round((time.time() - start_time), 2), "\n")

####################################################################################################################

