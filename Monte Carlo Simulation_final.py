
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm, t
import matplotlib.pyplot as plt

# === 1. Fetch Stock Data ===
def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change().dropna()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

# === 2. Portfolio Performance ===
def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns * weights) * Time
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
    return returns, std

# === 3. Historical VaR & CVaR ===
def historicalVaR(returns, alpha=5):
    return np.percentile(returns, alpha)

def historicalCVaR(returns, alpha=5):
    var = historicalVaR(returns, alpha)
    return returns[returns <= var].mean()

# === 4. Parametric VaR & CVaR ===
def var_parametric(ret, std, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        VaR = norm.ppf(1 - alpha / 100) * std - ret
    elif distribution == 't-distribution':
        nu = dof
        VaR = np.sqrt((nu - 2) / nu) * t.ppf(1 - alpha / 100, nu) * std - ret
    else:
        raise ValueError("distribution must be 'normal' or 't-distribution'")
    return VaR

def cvar_parametric(ret, std, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        CVaR = (alpha / 100) ** -1 * norm.pdf(norm.ppf(alpha / 100)) * std - ret
    elif distribution == 't-distribution':
        nu = dof
        xanu = t.ppf(alpha / 100, nu)
        CVaR = -1 / (alpha / 100) * (1 - nu) ** -1 * (nu - 2 + xanu ** 2) * t.pdf(xanu, nu) * std - ret
    else:
        raise ValueError("distribution must be 'normal' or 't-distribution'")
    return CVaR

# === 5. Monte Carlo Simulation (VaR & CVaR by portfolio value paths) ===
def simulate_portfolio_paths(meanReturns, covMatrix, weights, T, mc_sims, initialPortfolio):
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanM = meanM.T
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    for m in range(mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

    return portfolio_sims

def mcVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

def mcCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")

# === 6. Setup ===
stockList = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA']
stocks = stockList
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)
returns, meanReturns, covMatrix = getData(stocks, start=startDate, end=endDate)

weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)
returns['portfolio'] = returns.dot(weights)

Time = 100
InitialInvestment = 10000
alpha = 5

# === 7. Calculate Risk Measures ===
hVaR = -historicalVaR(returns['portfolio'], alpha=alpha) * np.sqrt(Time)
hCVaR = -historicalCVaR(returns['portfolio'], alpha=alpha) * np.sqrt(Time)

pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)
normVaR = var_parametric(pRet, pStd, 'normal', alpha)
normCVaR = cvar_parametric(pRet, pStd, 'normal', alpha)
tVaR = var_parametric(pRet, pStd, 't-distribution', alpha, dof=6)
tCVaR = cvar_parametric(pRet, pStd, 't-distribution', alpha, dof=6)

# === 8. Monte Carlo Simulation + Graph ===
mc_sims = 400
portfolio_sims = simulate_portfolio_paths(meanReturns, covMatrix, weights, Time, mc_sims, InitialInvestment)

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()

# Get last-day values from simulations
portResults = pd.Series(portfolio_sims[-1, :])
mcVaR_final = InitialInvestment - mcVaR(portResults, alpha=alpha)
mcCVaR_final = InitialInvestment - mcCVaR(portResults, alpha=alpha)

# === 9. Print Final Results ===
print(f"\n--- Portfolio Risk Measures on ${InitialInvestment} over {Time} days ---\n")
print("Expected Portfolio Return:", round(pRet * InitialInvestment, 2))
print("Portfolio Volatility:", round(pStd * InitialInvestment, 2))
print()

print("🔹 VaR Comparison (95% CI):")
print("Historical:", round(InitialInvestment * hVaR, 2))
print("Parametric Normal:", round(InitialInvestment * normVaR, 2))
print("Parametric t-dist:", round(InitialInvestment * tVaR, 2))
print("Monte Carlo:", round(mcVaR_final, 2))
print()

print("🔹 CVaR Comparison (95% CI):")
print("Historical:", round(InitialInvestment * hCVaR, 2))
print("Parametric Normal:", round(InitialInvestment * normCVaR, 2))
print("Parametric t-dist:", round(InitialInvestment * tCVaR, 2))
print("Monte Carlo:", round(mcCVaR_final, 2))
