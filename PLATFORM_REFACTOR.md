## Plan: Refactor Trading Bot for Reusability & Maintainability

**TL;DR**: The code has significant duplication across API clients, mixed responsibilities in classes, scattered configuration access, and no retry logic. Restructure into modular components with a base API client, separate platform-specific implementations, and a cleaner configuration system.

### Steps

1. **Extract base API client** — Create [api_clients/base_client.py](api_clients/base_client.py) with shared HTTP logic (`_build_headers()`, `_get_paginated()`, `_call_with_retry()`, timeout handling) to eliminate duplicate patterns across platform fetchers.

2. **Separate platform clients** — Split [MarketScanner](main/ai_trading_bot.py#L281) into [api_clients/polymarket_client.py](api_clients/polymarket_client.py) and [api_clients/kalshi_client.py](api_clients/kalshi_client.py), inheriting from base client and using platform-specific field mappings instead of inline JSON parsing.

3. **Consolidate configuration** — Create [utils/config_manager.py](utils/config_manager.py) with typed property accessors (`claude_api_key`, `min_edge`, `polymarket_api_url`, etc.) to eliminate fragmented config access patterns and add validation.

4. **Extract data models** — Move `MarketData`, `FairValueEstimate`, `TradeSignal` to [models/](models/) and create a [utils/response_parser.py](utils/response_parser.py) for JSON extraction logic (fixes hardcoded markdown parsing).

5. **Break up AdvancedTradingBot** — Extract [trading/position_manager.py](trading/position_manager.py), [trading/strategy.py](trading/strategy.py) (filtering, opportunity finding, signal generation), and [trading/executor.py](trading/executor.py) (trade execution placeholder) to reduce monolithic class.

6. **Add error handling & retry logic** — Create [utils/api_error.py](utils/api_error.py) with custom exception types and implement exponential backoff in base client for robustness.

### Further Considerations

1. **Async file operations** — Replace blocking `open()` calls with `aiofiles` in config loading and main setup to prevent event loop blocking?

2. **Testing approach** — Should we add dependency injection to classes (e.g., `ClaudeAnalyzer(api_key, model)` instead of reading config) to make them unit testable without real files/APIs?

3. **Trade execution gap** — The [\_execute_single_trade()](main/ai_trading_bot.py#L538) is stubbed and logs "DRY RUN MODE". Should we implement actual Polymarket/Kalshi trade execution, or keep it as a template for later?
