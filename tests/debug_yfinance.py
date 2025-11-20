import yfinance as yf
import json

tickers = ["AAPL", "MSFT", "NVDA"]
for t in tickers:
    print(f"--- {t} ---")
    try:
        tk = yf.Ticker(t)
        news = tk.news
        if news:
            print(json.dumps(news[0], indent=2))
        else:
            print("No news found.")
    except Exception as e:
        print(f"Error: {e}")
