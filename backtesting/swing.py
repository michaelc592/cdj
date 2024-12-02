import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Possible tickers to use: TSLA, ^GSPC, F, PFE, ^DJI, NSRGY, NDX
ticker = 'TSLA'
data = yf.download(ticker, start='2022-10-01',end='2024-10-01',interval='1d')
smp = yf.download('^GSPC', start='2022-10-01',end='2024-10-01',interval='1d')

# Define short-term and long-term moving averages
short_ma = 5
long_ma = 40


# Calculate moving averages
data['short_ma'] = data['Close'].rolling(window=short_ma).mean()
data['long_ma'] = data['Close'].rolling(window=long_ma).mean()

# Create a new column to signal buy/sell orders based on moving averages crossover
data['Position'] = np.where(data['short_ma'] > data['long_ma'], 1, 0)
data['Signal'] = data['Position'].diff()

# Array of buy dates
buy = []
for i in range(len(data)):  # Start from 1 to avoid checking i > 0 separately
    if i > 0 and data['Signal'].iloc[i] == 1:
        buy.append(data.iloc[i, 2])
    else:
      buy.append(np.nan)
data["Buy"] = buy

# Array of sell dates
sell = []
for i in range(len(data)):  # Start from 1 to avoid checking i > 0 separately
    if i > 0 and data['Signal'].iloc[i] == -1:
        sell.append(data.iloc[i, 2])
    else:
      sell.append(np.nan)
data["Sell"] = sell

# Set up initial capital and calculate position size based on daily data points
initial_capital = 100000.0
position_size = initial_capital / len(data)

# Calculate daily returns and strategy returns
data['Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Signal'] * data['Returns'] * position_size

# Calculate cumulative returns over the period
cumulative_returns = []
capital = initial_capital
position_active = False
buy_price = 0

for i in range(len(data)):
    if data['Signal'].iloc[i] == 1 and not position_active:
        buy_price = data['Close'].iloc[i]
        position_active = True
        cumulative_returns.append(capital)

    elif data['Position'].iloc[i] == 1 and position_active:
        cumulative_returns.append(capital * (data['Close'].iloc[i] / buy_price))

    elif data['Signal'].iloc[i] == -1 and position_active:
        capital = capital * (data['Close'].iloc[i] / buy_price)
        position_active = False
        cumulative_returns.append(capital)
    else:
        cumulative_returns.append(capital)

for i in range(len(cumulative_returns)):
  try:
    cumulative_returns[i] = float(cumulative_returns[i].iloc[0])
  except:
    pass

data['Cumulative Returns'] = cumulative_returns

# Column of returns if we just invested in the market (control)
data['Control'] = initial_capital * data['Close'] / data['Close'].iloc[0]

#Column of returns if we just invested in the S&P 500 (second control)
smp['Control'] = initial_capital * smp['Close'] / smp['Close'].iloc[0]
#-------------------------------------------------------------------------------

# Print out key statistics
final = cumulative_returns[-1]
print()
print('-'*100)
print('Starting capital: $' + str(round(initial_capital,2)))
print("Total value after strategy period: $" + str(round(final,2)))
print("Percent gain from investing: " + str(round(final/initial_capital*100,2)) + "%")
print("Gain of stock over 2 year period: " + str(round(data['Close'].iloc[-1].iloc[0] / data['Close'].iloc[0].iloc[0]*100,2)) + "%")
print('-'*100)
print()

# Plotting the close price with buy/sell signals
plt.figure(figsize=(20, 4))
plt.plot(data['Close'], label='Close Price', color='deepskyblue', alpha=0.5, linewidth=2.5)
plt.plot(data['short_ma'], label=f'Short ({short_ma}-Day) MA', color='orange', linestyle='--', alpha=0.7)
plt.plot(data['long_ma'], label=f'Long ({long_ma}-Day) MA', color='red', linestyle='--', alpha=0.7)
plt.scatter(data.index, buy, marker='^', color='green', label='Buy Signal', alpha=1, s=100)
plt.scatter(data.index, sell, marker='v', color='red', label='Sell Signal', alpha=1, s=100)
plt.title('Close Price with Buy and Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Plotting cumulative returns over time
plt.figure(figsize=(10, 6))
plt.plot(data['Cumulative Returns'], label='Swing Strategy',color='green', linewidth=2.5)
plt.plot(data['Control'],label='No Trading Strategy',color='purple',linewidth=2.5)
plt.plot(smp['Control'], label='S&P 500 Investment', color='orange', linestyle='--', linewidth=2.5)
plt.title('Total Value of Investment in ' + ticker + ' Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Portfolio Value (USD)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
