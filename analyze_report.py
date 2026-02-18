#!/usr/bin/env python3
"""Analyze the latest cycle report"""
import json
import glob
import os

# Find the latest report
reports = glob.glob('reports/cycle_*.json')
latest = max(reports, key=os.path.getmtime)
print(f"Analyzing: {latest}\n")

with open(latest) as f:
    data = json.load(f)

print('=== MARKETS FETCHED ===')
print(f"Total scanned: {data['counts']['scanned']}")
print(f"Passed filters: {data['counts']['passed_filters']}")
print(f"Analyzed: {data['counts']['analyzed']}")
print(f"Opportunities (>8% edge): {data['counts']['opportunities']}")
print(f"Trade signals: {data['counts']['signals']}")
print()

# Group by series
series_count = {}
for m in data['markets']:
    ticker = m['market_id']
    # Extract series from ticker (e.g., KXCABOUT-29JAN-PBOD -> KXCABOUT)
    series = ticker.split('-')[0]
    if series not in series_count:
        series_count[series] = 0
    series_count[series] += 1

print('=== SERIES BREAKDOWN ===')
for series, count in sorted(series_count.items(), key=lambda x: -x[1]):
    print(f"  {series}: {count} markets")

print()
print('=== ALL MARKETS WITH PRICES ===')
print(f"{'Price':>6} | {'Volume':>12} | Title")
print("-" * 80)

for m in data['markets']:
    title = m['title'][:55]
    yes = m['prices']['yes']
    vol = m['stats']['volume']
    passed = "✓" if m['filters']['passed'] else "✗"
    print(f"{yes*100:5.1f}% | ${vol:>10,.0f} | {passed} {title}")

print()
print('=== API COST ===')
cost = data['api_cost']
print(f"Total cost: ${cost['total_cost']:.4f}")
print(f"Total requests: {cost['total_requests']}")
print(f"Avg per request: ${cost['avg_cost_per_request']:.6f}")
