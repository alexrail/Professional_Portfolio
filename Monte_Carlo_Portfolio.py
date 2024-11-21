# This script is an implementation of a monte carlos simulation to
# determine the possible returns of a stock portfolio over time.

#importing relevant modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

#function for importing data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

#selecting stocks and date ranges
stockList = ['AAPL','MSFT','GOOG','AMZN','NVDA','META']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

#running function to pull data on selected stocks
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

#evenly weighing each stock in portfolio
weights = [1/(len(stockList))] * len(stockList)

# monte carlo simulation
mc_sims = 400 # number of simulations
T = 100 #timeframe in days


meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
initialPortfolio = 10000
for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))#uncorrelated RV's
    L = np.linalg.cholesky(covMatrix) #cholesky decomposition
    dailyReturns = meanM + np.inner(L, Z) #correlated daily returns for individual stocks
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio


#plotting the simulation
plt.plot(portfolio_sims)
plt.ylabel('Predicted Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of Stock Portfolio')
plt.show()
