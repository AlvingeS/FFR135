import yfinance as yf
import matplotlib.pyplot as plt

# Download historical data as dataframe
data = yf.download("AAPL", start="2020-01-01", end="2020-12-31")
data = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
data.to_csv('aapl.csv', index=False, header=False)
