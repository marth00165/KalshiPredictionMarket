#!/usr/bin/env python3
"""Quick script to check trending Kalshi markets by fetching interesting events"""
import urllib.request
import json
import time

print("=== FETCHING ALL EVENTS ===\n")

url = 'https://api.elections.kalshi.com/trade-api/v2/events?limit=200&status=open'
try:
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read())
    events = data.get('events', [])
    print(f"Found {len(events)} open events total\n")
    
    # Show all unique series tickers
    series_set = set()
    for e in events:
        series = e.get('series_ticker', '')
        if series:
            series_set.add(series)
    
    print(f"Unique series: {len(series_set)}")
    for s in sorted(series_set)[:30]:
        print(f"  {s}")
        
except Exception as ex:
    print(f"Error fetching events: {ex}")

print("\n" + "="*60)
time.sleep(7)

# Fetch markets for an event with full ticker
print("\n=== MARKETS FOR POPE EVENT (KXNEWPOPE-70) ===\n")

url = 'https://api.elections.kalshi.com/trade-api/v2/markets?limit=50&status=open&event_ticker=KXNEWPOPE-70'
try:
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read())
    
    markets = data.get('markets', [])
    print(f"Markets found: {len(markets)}\n")
    
    for m in markets[:10]:
        yes_bid = m.get('yes_bid', 0)
        yes_ask = m.get('yes_ask', 0)
        vol = m.get('volume', 0)
        print(f"{m.get('ticker')}")
        print(f"  {m.get('title')[:65]}")
        print(f"  Bid: {yes_bid}¢ | Ask: {yes_ask}¢ | Vol: ${vol:,}")
        print()

except Exception as ex:
    print(f"Error: {ex}")

time.sleep(7)

# Try fetching by series_ticker instead
print("\n=== MARKETS BY SERIES (KXNEWPOPE) ===\n")

url = 'https://api.elections.kalshi.com/trade-api/v2/markets?limit=50&status=open&series_ticker=KXNEWPOPE'
try:
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read())
    
    markets = data.get('markets', [])
    print(f"Markets found: {len(markets)}\n")
    
    for m in markets[:10]:
        yes_bid = m.get('yes_bid', 0)
        yes_ask = m.get('yes_ask', 0)
        vol = m.get('volume', 0)
        print(f"{m.get('ticker')}")
        print(f"  {m.get('title')[:65]}")
        print(f"  Bid: {yes_bid}¢ | Ask: {yes_ask}¢ | Vol: ${vol:,}")
        print()

except Exception as ex:
    print(f"Error: {ex}")
