import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

apple_data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")
tesla_data = yf.download("TSLA", start="2023-01-01", end="2023-12-31")
pfizer_data = yf.download("PFE", start="2023-01-01", end="2023-12-31")
apple_prices = apple_data['Close']
tesla_prices = tesla_data['Close']
pfizer_prices = pfizer_data['Close']

#code for mean revision
def mean_reversion(prices, moving_average):
   if prices[-1] > moving_average[-1]:
       return "sell"
   elif prices[-1] < moving_average[-1]:
       return "buy"
   else:
       return "hold"

# function for backtesting
def backtest(prices, moving_average, initial_capital=10000, window_size=20):
   cash = initial_capital
   shares = 0
   portfolio_values = []


   for i in range(window_size, len(prices)):
       signal = mean_reversion(prices[:i+1], moving_average[:i+1])
       current_price = prices[i]
      
       if signal == "buy" and cash > current_price:
           # buy as much stock as possible with the cash
           shares_to_buy = cash // current_price
           cash -= shares_to_buy * current_price
           shares += shares_to_buy
       elif signal == "sell" and shares > 0:
           # Sell all
           cash += shares * current_price
           shares = 0
      
       # calculate total portfolio value
       portfolio_value = cash + shares * current_price
       portfolio_values.append(portfolio_value)


   # put results in a DataFrame
   backtest_results = pd.DataFrame({
       'Date': prices.index[window_size:],
       'Portfolio Value': portfolio_values
   })
   backtest_results.set_index('Date', inplace=True)
  
   return backtest_results

apple_backtest = backtest(apple_prices, moving_average_apple)
tesla_backtest = backtest(tesla_prices, moving_average_apple)
pfizer_backtest = backtest(pfizer_prices, moving_average_pfizer)

def create_plot(stock, ticker):
   plt.figure(figsize=(12, 6))
   plt.plot(stock['Portfolio Value'], label='Portfolio Value')
   plt.title('Mean Reversion Strategy Backtest with ' + ticker +' stock')
   plt.xlabel('Date')
   plt.ylabel('Portfolio Value')
   plt.legend()
   plt.show()


create_plot(apple_backtest, "APPL")
create_plot(tesla_backtest, "TSLA")
create_plot(pfizer_backtest, "PFE")

def plot_strategy_vs_sp500(backtest_results, sp500_data, initial_capital=10000):
   """
   Overlays the strategy portfolio value with S&P 500 performance.
  
   Parameters:
   - backtest_results: DataFrame containing the portfolio value over time.
   - sp500_data: Series of S&P 500 closing prices over time.
   - initial_capital: Starting amount of money for normalization.
  
   """
   # Step 1: Normalize Portfolio Value
   backtest_results['Normalized Portfolio Value'] = (
       backtest_results['Portfolio Value'] / backtest_results['Portfolio Value'].iloc[0] * initial_capital
   )


   # Step 2: Normalize S&P 500 Data
   sp500_normalized = sp500_data / sp500_data.iloc[0] * initial_capital


   # Step 3: Plot both normalized values
   plt.figure(figsize=(14, 7))
   plt.plot(backtest_results.index, backtest_results['Normalized Portfolio Value'], label='Strategy Portfolio', color='blue')
   plt.plot(sp500_normalized.index, sp500_normalized, label='S&P 500 Index', color='orange')


   # Add labels, title, and legend
   plt.title('Strategy vs. S&P 500 Performance (Normalized to Initial Capital)')
   plt.xlabel('Date')
   plt.ylabel('Portfolio Value (Starting with ${})'.format(initial_capital))
   plt.legend()


   plt.show()
