import yfinance as yf

ticker = "AAPL"
print(f"Downloading {ticker} data...")

df = yf.download(ticker, period="30d", interval="1d", progress=False, auto_adjust=True)

print(f"\nDataFrame type: {type(df)}")
print(f"DataFrame shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nColumn types: {df.dtypes}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nAccessing Close column:")
try:
    close = df["Close"]
    print(f"  Type: {type(close)}")
    print(f"  Has tolist: {hasattr(close, 'tolist')}")
    if hasattr(close, 'tolist'):
        print(f"  Values: {close.tolist()[:5]}")
except Exception as e:
    print(f"  Error: {e}")
