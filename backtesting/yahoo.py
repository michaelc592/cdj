import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored as cl
from math import floor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ticker = "GME"
stock = yf.Ticker(ticker)
hist = stock.history(period = "2y")
# apple = yf.download("AAPL", period = "1y")
# hist.to_csv("tesla.csv")
# apple.to_csv("apple.csv")

# tickerStrings = ['AAPL', 'MSFT']
# df_list = []
# for ticker in tickerStrings:
#     data = yf.download(ticker, group_by="Ticker", period='d')
#     data['ticker'] = ticker
#     df_list.append(data)

# # Combine all dataframes into a single dataframe
# df = pd.concat(df_list)
# df.to_csv('ticker.csv')

def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret.iloc[i] < 0:
            up.append(0)
            down.append(ret.iloc[i])
        else:
            up.append(ret.iloc[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]

def implement_rsi_strategy(prices, rsi):    
    buy_price = []
    sell_price = []
    rsi_signal = []
    signal = 0

    for i in range(len(rsi)):
        if rsi.iloc[i-1] > 30 and rsi.iloc[i] < 30:
            if signal != 1:
                buy_price.append(prices.iloc[i])
                sell_price.append(np.nan)
                signal = 1
                rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
        elif rsi.iloc[i-1] < 70 and rsi.iloc[i] > 70:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices.iloc[i])
                signal = -1
                rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            rsi_signal.append(0)
            
    return buy_price, sell_price, rsi_signal

hist['rsi_14'] = get_rsi(hist['Close'], 14)
hist = hist.dropna()
hist.tail()

ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
ax1.plot(hist['Close'], linewidth = 2.5)
ax1.set_title(ticker + ' CLOSE PRICE')
ax2.plot(hist['rsi_14'], color = 'orange', linewidth = 2.5)
ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
ax2.set_title(ticker + ' RELATIVE STRENGTH INDEX')
plt.show()

buy_price, sell_price, rsi_signal = implement_rsi_strategy(hist['Close'], hist['rsi_14'])

ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
ax1.plot(hist['Close'], linewidth = 2.5, color = 'skyblue', label = 'ibm')
ax1.plot(hist.index, buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
ax1.plot(hist.index, sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
ax1.set_title(ticker + ' RSI TRADE SIGNALS')
ax2.plot(hist['rsi_14'], color = 'orange', linewidth = 2.5)
ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
plt.show()

position = []
for i in range(len(rsi_signal)):
    if rsi_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(hist['Close'])):
    if rsi_signal[i] == 1:
        position[i] = 1
    elif rsi_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
rsi = hist['rsi_14']
close_price = hist['Close']
rsi_signal = pd.DataFrame(rsi_signal).rename(columns = {0:'rsi_signal'}).set_index(hist.index)
position = pd.DataFrame(position).rename(columns = {0:'rsi_position'}).set_index(hist.index)

frames = [close_price, rsi, rsi_signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)

ret = pd.DataFrame(np.diff(hist['Close'])).rename(columns = {0:'returns'})
rsi_strategy_ret = []
ret_percent = [None]

for i in range(len(ret)):
    returns = ret['returns'][i]*strategy['rsi_position'][i]
    rsi_strategy_ret.append(returns)
    if (i != 0):
        ret_percent.append(returns / hist['Close'][i-1])
    
rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns = {0:'rsi_returns'})
rsi_strategy_ret_df['returns_percent'] = ret_percent

investment_value = 100000
investment_value2 = 100000
rsi_investment_ret = []
rsi_investment_ret2 = []
gained_returns = 0

for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
    number_of_stocks = floor(investment_value/hist['Close'][i])
    returns = number_of_stocks*rsi_strategy_ret_df['rsi_returns'][i]
    rsi_investment_ret.append(returns)

total_values = [100000]
total_value = 100000
#strategy 2
for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
    if i != 0 and strategy['rsi_position'][i] == 0 and strategy['rsi_position'][i-1] == 1:
        investment_value2 += gained_returns
        gained_returns = 0
    number_of_stocks = floor(investment_value2/hist['Close'][i])
    returns = number_of_stocks*rsi_strategy_ret_df['rsi_returns'][i]
    total_value += returns
    if strategy['rsi_position'][i] == 1:
        gained_returns += returns
    rsi_investment_ret2.append(returns)
    total_values.append(total_value)

rsi_strategy_ret_df['rsi_returns'].dropna()
rsi_strategy_ret_df = rsi_strategy_ret_df[rsi_strategy_ret_df['rsi_returns'] != 0]

dates = hist.index

if len(dates) > len(total_values):
    dates = dates[:len(total_values)]

plt.figure(figsize=(10, 6))
plt.plot(dates, total_values, linewidth=2.5, color='lightgreen', label='Total value of investment')

plt.title('Total Value Of Investment Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Value', fontsize=14)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(rsi_strategy_ret_df['returns_percent'], bins=40, color='green', edgecolor = 'white', alpha=0.7)
plt.title('Return Distribution Histogram')
plt.xlabel('Percent Returns')
plt.ylabel('Frequency')
plt.show()

rsi_investment_ret_df = pd.DataFrame(rsi_investment_ret).rename(columns = {0:'investment_returns'})
total_investment_ret = round(sum(rsi_investment_ret_df['investment_returns']), 2)
profit_percentage = floor((total_investment_ret/investment_value)*100)
print(cl('Profit gained from the RSI strategy by investing $100k in ' + ticker + ' : {}'.format(total_investment_ret), attrs = ['bold']))
print(cl('Profit percentage of the RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))

rsi_investment_ret_df2 = pd.DataFrame(rsi_investment_ret2).rename(columns = {0:'investment_returns'})
total_investment_ret2 = round(sum(rsi_investment_ret_df2['investment_returns']), 2)
profit_percentage2 = floor((total_investment_ret2/investment_value)*100)
print(cl('Profit gained from the RSI strategy 2 by investing $100k in ' + ticker + ' : {}'.format(total_investment_ret2), attrs = ['bold']))
print(cl('Profit percentage of the RSI strategy 2 : {}%'.format(profit_percentage2), attrs = ['bold']))

first_value = hist['Close'].iloc[0]
last_value = hist['Close'].iloc[-1]
percent_change = (last_value - first_value) / first_value * 100
print('Percentage change in ' + ticker + ' over the same time period: ' + str(percent_change) + '%')