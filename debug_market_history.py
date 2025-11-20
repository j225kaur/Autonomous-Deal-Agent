from src.data_ingestion.yahoo_sec import fetch_price_history
import json

tickers = ["AAPL", "MSFT", "NVDA"]
print("Fetching price history for:", tickers)

try:
    market_history = fetch_price_history(tickers, days=30)
    print(f"\nResult: {json.dumps(market_history, indent=2)}")
    
    for ticker, data in market_history.items():
        print(f"\n{ticker}:")
        print(f"  Close prices: {len(data.get('close', []))} values")
        print(f"  Volume data: {len(data.get('volume', []))} values")
        if data.get('close'):
            print(f"  Latest close: {data['close'][-1]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
