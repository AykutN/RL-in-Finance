import yfinance as yf
import pandas as pd

symbols = ['AAPL', 'GOOGL']
data = yf.download(symbols, start='2020-01-01', end='2024-01-01')['Adj Close']


# Tamamen NaN olan sütunları kaldır
data = data.dropna(axis=1, how="all")

# Günlük getirileri hesapla
returns = data.pct_change(fill_method=None).dropna(how="any")



