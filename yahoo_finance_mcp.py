"""
Yahoo Finance MCP Server

A comprehensive MCP server for accessing Yahoo Finance data including:
- Real-time and historical stock quotes
- Company information and financials
- Technical indicators
- Options data
- News and analysis

Requirements:
    pip install mcp yfinance httpx pydantic
"""

import json
import asyncio
import copy
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Optional, List, Dict, Any
from threading import Lock

import yfinance as yf
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Initialize the MCP server
mcp = FastMCP("yahoo_finance_mcp")

# ============================================================================
# Constants
# ============================================================================

DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "1d"
MAX_SYMBOLS = 10
YAHOO_TIMEOUT = float(os.getenv("YF_TIMEOUT", "10"))
YAHOO_INFO_CACHE_TTL = int(os.getenv("YF_INFO_CACHE_TTL", "60"))
YAHOO_BATCH_WORKERS = int(os.getenv("YF_BATCH_WORKERS", "5"))
_YAHOO_INFO_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_YAHOO_INFO_CACHE_LOCK = Lock()

# ============================================================================
# Enums
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"

class Period(str, Enum):
    """Valid periods for historical data."""
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    YTD = "ytd"
    MAX = "max"

class Interval(str, Enum):
    """Valid intervals for historical data."""
    ONE_MINUTE = "1m"
    TWO_MINUTES = "2m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    SIXTY_MINUTES = "60m"
    NINETY_MINUTES = "90m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"

# ============================================================================
# Pydantic Input Models
# ============================================================================

class TickerInput(BaseModel):
    """Input model for single ticker operations."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    symbol: str = Field(
        ..., 
        description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')",
        min_length=1,
        max_length=10
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.upper().strip()

class MultiTickerInput(BaseModel):
    """Input model for multiple ticker operations."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    symbols: List[str] = Field(
        ...,
        description="List of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])",
        min_length=1,
        max_length=MAX_SYMBOLS
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
    
    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v: List[str]) -> List[str]:
        return [s.upper().strip() for s in v]

class HistoricalDataInput(BaseModel):
    """Input model for historical data retrieval."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    symbol: str = Field(
        ...,
        description="Stock ticker symbol (e.g., 'AAPL')",
        min_length=1,
        max_length=10
    )
    period: Period = Field(
        default=Period.ONE_MONTH,
        description="Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"
    )
    interval: Interval = Field(
        default=Interval.ONE_DAY,
        description="Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.upper().strip()

class OptionsInput(BaseModel):
    """Input model for options data retrieval."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    symbol: str = Field(
        ...,
        description="Stock ticker symbol (e.g., 'AAPL')",
        min_length=1,
        max_length=10
    )
    expiration_date: Optional[str] = Field(
        default=None,
        description="Options expiration date in YYYY-MM-DD format. If not provided, returns nearest expiration."
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.upper().strip()

class SearchInput(BaseModel):
    """Input model for ticker search."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    query: str = Field(
        ...,
        description="Search query for finding tickers (company name or partial symbol)",
        min_length=1,
        max_length=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )

class ScreenerInput(BaseModel):
    """Input model for stock screening."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    market_cap_min: Optional[float] = Field(
        default=None,
        description="Minimum market cap in USD (e.g., 1000000000 for $1B)"
    )
    market_cap_max: Optional[float] = Field(
        default=None,
        description="Maximum market cap in USD"
    )
    pe_ratio_max: Optional[float] = Field(
        default=None,
        description="Maximum P/E ratio"
    )
    dividend_yield_min: Optional[float] = Field(
        default=None,
        description="Minimum dividend yield as percentage (e.g., 2.0 for 2%)"
    )
    sector: Optional[str] = Field(
        default=None,
        description="Sector filter (e.g., 'Technology', 'Healthcare', 'Financial Services')"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format"
    )

class SectorKeyInput(BaseModel):
    """Input model for sector lookups."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    key: str = Field(
        ...,
        description="Sector key (e.g., 'technology', 'healthcare', 'financial-services')",
        min_length=1,
        max_length=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format"
    )

    @field_validator('key')
    @classmethod
    def validate_key(cls, v: str) -> str:
        return v.strip().lower()

class IndustryKeyInput(BaseModel):
    """Input model for industry lookups."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    key: str = Field(
        ...,
        description="Industry key (e.g., 'software-infrastructure', 'semiconductors')",
        min_length=1,
        max_length=100
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Output format"
    )

    @field_validator('key')
    @classmethod
    def validate_key(cls, v: str) -> str:
        return v.strip().lower()

# ============================================================================
# Helper Functions
# ============================================================================

def _handle_error(e: Exception, context: str = "") -> str:
    """Consistent error formatting across all tools."""
    error_msg = str(e)
    if "No data found" in error_msg or "symbol may be delisted" in error_msg:
        return f"Error: Symbol not found or no data available. Please verify the ticker symbol is correct. {context}"
    if "404" in error_msg:
        return f"Error: Resource not found. {context}"
    if "rate limit" in error_msg.lower():
        return "Error: Rate limit exceeded. Please wait before making more requests."
    return f"Error: {error_msg}. {context}"

def with_timeout(timeout: float = YAHOO_TIMEOUT):
    """Run blocking Yahoo Finance work in a worker thread with a timeout."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return f"Error: Yahoo Finance request timed out after {timeout:.0f}s."
        return wrapper
    return decorator


def _get_cached_ticker_data(symbol: str) -> Optional[Dict[str, Any]]:
    cache_key = symbol.upper()
    with _YAHOO_INFO_CACHE_LOCK:
        cached = _YAHOO_INFO_CACHE.get(cache_key)
        if cached is None:
            return None
        expires_at, payload = cached
        if time.time() >= expires_at:
            _YAHOO_INFO_CACHE.pop(cache_key, None)
            return None
        return copy.deepcopy(payload)


def _set_cached_ticker_data(symbol: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cache_key = symbol.upper()
    stored = copy.deepcopy(payload)
    with _YAHOO_INFO_CACHE_LOCK:
        _YAHOO_INFO_CACHE[cache_key] = (time.time() + max(YAHOO_INFO_CACHE_TTL, 1), stored)
    return copy.deepcopy(payload)


def _get_ticker_data(symbol: str, *, use_cache: bool = True) -> Dict[str, Any]:
    if use_cache:
        cached = _get_cached_ticker_data(symbol)
        if cached is not None:
            return cached
    ticker = yf.Ticker(symbol)
    data = _ticker_to_dict(ticker)
    return _set_cached_ticker_data(symbol, data)


def _get_multiple_ticker_data(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    def _load_symbol(symbol: str) -> Dict[str, Any]:
        return _get_ticker_data(symbol)

    if not symbols:
        return {}

    max_workers = max(1, min(YAHOO_BATCH_WORKERS, len(symbols)))
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_load_symbol, symbol): symbol for symbol in symbols}
        for future, symbol in future_map.items():
            try:
                results[symbol] = future.result()
            except Exception as exc:
                results[symbol] = {"symbol": symbol, "error": str(exc)}
    return results

def _format_number(value: Any, decimals: int = 2) -> str:
    """Format numbers for display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1e12:
            return f"${value/1e12:.{decimals}f}T"
        if abs(value) >= 1e9:
            return f"${value/1e9:.{decimals}f}B"
        if abs(value) >= 1e6:
            return f"${value/1e6:.{decimals}f}M"
        if abs(value) >= 1e3:
            return f"${value/1e3:.{decimals}f}K"
        return f"${value:.{decimals}f}"
    return str(value)

def _format_percentage(value: Any, decimals: int = 2) -> str:
    """Format percentages for display."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value * 100:.{decimals}f}%"
    return str(value)

def _safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary."""
    return data.get(key, default) if data else default

def _first_valid(*values: Any) -> Any:
    """Return the first non-null/non-NaN value."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and value != value:
            continue
        return value
    return None

def _resolve_quote_price(info: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve regular, pre-market, and after-hours pricing into one payload."""
    previous_close = _first_valid(
        _safe_get(info, "previousClose"),
        _safe_get(info, "regularMarketPreviousClose"),
    )
    regular_price = _first_valid(
        _safe_get(info, "currentPrice"),
        _safe_get(info, "regularMarketPrice"),
    )
    pre_market_price = _safe_get(info, "preMarketPrice")
    post_market_price = _safe_get(info, "postMarketPrice")

    current_price = regular_price
    current_price_source = "regular_market"
    current_price_label = "Regular Market"
    change_value = _first_valid(
        _safe_get(info, "regularMarketChange"),
        regular_price - previous_close
        if regular_price is not None and previous_close not in (None, 0)
        else None,
    )
    change_percent = _first_valid(
        _safe_get(info, "regularMarketChangePercent"),
        (change_value / previous_close) * 100
        if change_value is not None and previous_close not in (None, 0)
        else None,
    )

    if post_market_price is not None:
        current_price = post_market_price
        current_price_source = "post_market"
        current_price_label = "After Hours"
        change_value = _first_valid(
            _safe_get(info, "postMarketChange"),
            post_market_price - regular_price if regular_price is not None else None,
        )
        change_percent = _first_valid(
            _safe_get(info, "postMarketChangePercent"),
            (change_value / regular_price) * 100
            if change_value is not None and regular_price not in (None, 0)
            else None,
        )
    elif pre_market_price is not None:
        current_price = pre_market_price
        current_price_source = "pre_market"
        current_price_label = "Pre-Market"
        change_value = _first_valid(
            _safe_get(info, "preMarketChange"),
            pre_market_price - previous_close
            if previous_close is not None
            else None,
        )
        change_percent = _first_valid(
            _safe_get(info, "preMarketChangePercent"),
            (change_value / previous_close) * 100
            if change_value is not None and previous_close not in (None, 0)
            else None,
        )

    return {
        "current_price": current_price,
        "current_price_source": current_price_source,
        "current_price_label": current_price_label,
        "market_state": _safe_get(info, "marketState"),
        "regular_market_price": regular_price,
        "pre_market_price": pre_market_price,
        "post_market_price": post_market_price,
        "price_change": change_value,
        "price_change_percent": change_percent,
        "previous_close": previous_close,
    }

def _normalize_value(value: Any) -> Any:
    """Normalize values for JSON/markdown output (dates, NaN, nested structures)."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    # Handle pandas-like objects early to avoid ambiguous truthiness in callers.
    if hasattr(value, "to_dict") and hasattr(value, "empty") and not isinstance(value, dict):
        try:
            if value.empty:
                return [] if hasattr(value, "columns") else {}
        except Exception:
            pass
        if hasattr(value, "reset_index"):
            try:
                records = value.reset_index().to_dict(orient="records")
                return [_normalize_value(r) for r in records]
            except Exception:
                pass
        try:
            return _normalize_value(value.to_dict())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().isoformat()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value

def _calendar_to_dict(calendar: Any) -> Dict[str, Any]:
    """Convert the Yahoo Finance calendar to a simple dict."""
    if calendar is None:
        return {}
    try:
        if hasattr(calendar, "empty") and calendar.empty:
            return {}
        if hasattr(calendar, "to_dict"):
            data = calendar.to_dict()
            if isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))
            return _normalize_value(data) if isinstance(data, dict) else {}
        if isinstance(calendar, dict):
            return _normalize_value(calendar)
    except Exception:
        return {}
    return {}

def _df_to_records(df: Any) -> List[Dict[str, Any]]:
    """Convert a DataFrame-like object to list of records, normalized."""
    if df is None:
        return []
    try:
        if hasattr(df, "empty") and df.empty:
            return []
        if hasattr(df, "reset_index") and hasattr(df, "to_dict"):
            records = df.reset_index().to_dict(orient="records")
            return [_normalize_value(r) for r in records]
    except Exception:
        return []
    return []

def _format_value_for_table(value: Any) -> str:
    """Format values for markdown tables."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return str(value)

def _records_to_markdown_table(records: List[Dict[str, Any]], max_rows: int = 8) -> str:
    """Render a list of dicts as a markdown table."""
    if not records:
        return ""
    keys = list(records[0].keys())[:6]
    header = "| " + " | ".join(keys) + " |"
    separator = "|" + "|".join(["---"] * len(keys)) + "|"
    rows = []
    for row in records[:max_rows]:
        rows.append("| " + " | ".join(_format_value_for_table(row.get(k)) for k in keys) + " |")
    table = "\n".join([header, separator] + rows)
    if len(records) > max_rows:
        table += f"\n\n*Showing {max_rows} of {len(records)} rows*"
    return table

def _ticker_to_dict(ticker: yf.Ticker) -> Dict[str, Any]:
    """Convert ticker info to a clean dictionary."""
    info = ticker.info
    price_data = _resolve_quote_price(info)
    return {
        "symbol": _safe_get(info, "symbol"),
        "name": _safe_get(info, "longName") or _safe_get(info, "shortName"),
        "currency": _safe_get(info, "currency"),
        "exchange": _safe_get(info, "exchange"),
        "market_cap": _safe_get(info, "marketCap"),
        "current_price": price_data["current_price"],
        "current_price_source": price_data["current_price_source"],
        "current_price_label": price_data["current_price_label"],
        "market_state": price_data["market_state"],
        "regular_market_price": price_data["regular_market_price"],
        "pre_market_price": price_data["pre_market_price"],
        "post_market_price": price_data["post_market_price"],
        "price_change": price_data["price_change"],
        "price_change_percent": price_data["price_change_percent"],
        "previous_close": price_data["previous_close"],
        "open": _safe_get(info, "open") or _safe_get(info, "regularMarketOpen"),
        "day_high": _safe_get(info, "dayHigh") or _safe_get(info, "regularMarketDayHigh"),
        "day_low": _safe_get(info, "dayLow") or _safe_get(info, "regularMarketDayLow"),
        "volume": _safe_get(info, "volume") or _safe_get(info, "regularMarketVolume"),
        "avg_volume": _safe_get(info, "averageVolume"),
        "52_week_high": _safe_get(info, "fiftyTwoWeekHigh"),
        "52_week_low": _safe_get(info, "fiftyTwoWeekLow"),
        "pe_ratio": _safe_get(info, "trailingPE"),
        "forward_pe": _safe_get(info, "forwardPE"),
        "peg_ratio": _safe_get(info, "pegRatio"),
        "price_to_book": _safe_get(info, "priceToBook"),
        "eps": _safe_get(info, "trailingEps"),
        "dividend_yield": _safe_get(info, "dividendYield"),
        "dividend_rate": _safe_get(info, "dividendRate"),
        "profit_margin": _safe_get(info, "profitMargins"),
        "revenue_growth": _safe_get(info, "revenueGrowth"),
        "beta": _safe_get(info, "beta"),
        "52_week_change": _safe_get(info, "52WeekChange"),
        "sector": _safe_get(info, "sector"),
        "industry": _safe_get(info, "industry"),
        "description": _safe_get(info, "longBusinessSummary"),
        "website": _safe_get(info, "website"),
        "employees": _safe_get(info, "fullTimeEmployees"),
        "country": _safe_get(info, "country"),
        "city": _safe_get(info, "city"),
    }

def _format_quote_markdown(data: Dict[str, Any]) -> str:
    """Format quote data as markdown."""
    price = data.get("current_price", "N/A")
    price_change = data.get("price_change")
    pct_change = data.get("price_change_percent")
    price_label = data.get("current_price_label") or "Price"
    volume = data.get("volume")
    volume_display = f"{volume:,}" if isinstance(volume, (int, float)) else "N/A"
    regular_market_price = data.get("regular_market_price")
    regular_market_display = (
        f"${regular_market_price}" if isinstance(regular_market_price, (int, float)) else "N/A"
    )

    change = ""
    if isinstance(price, (int, float)) and isinstance(price_change, (int, float)) and isinstance(
        pct_change, (int, float)
    ):
        direction = "🟢" if price_change >= 0 else "🔴"
        change = f" {direction} {price_change:+.2f} ({pct_change:+.2f}%)"

    session_line = ""
    if data.get("current_price_source") != "regular_market":
        session_line = (
            f"\n**Session:** {price_label}"
            f" | Regular Market: {regular_market_display}"
        )

    return f"""## {data.get('name', 'Unknown')} ({data.get('symbol', 'N/A')})

**{price_label}:** ${price}{change}{session_line}

| Metric | Value |
|--------|-------|
| Open | ${data.get('open', 'N/A')} |
| Day High | ${data.get('day_high', 'N/A')} |
| Day Low | ${data.get('day_low', 'N/A')} |
| Previous Close | ${data.get('previous_close', 'N/A')} |
| Volume | {volume_display} |
| Market Cap | {_format_number(data.get('market_cap'))} |
| P/E Ratio | {data.get('pe_ratio', 'N/A')} |
| 52W High | ${data.get('52_week_high', 'N/A')} |
| 52W Low | ${data.get('52_week_low', 'N/A')} |

**Sector:** {data.get('sector', 'N/A')} | **Industry:** {data.get('industry', 'N/A')}
"""

# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool(
    name="yf_get_quote",
    annotations={
        "title": "Get Stock Quote",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_quote(params: TickerInput) -> str:
    """Get real-time stock quote and basic information for a ticker.
    
    Retrieves current price, daily statistics, and key metrics for a stock.
    
    Args:
        params (TickerInput): Input containing:
            - symbol (str): Stock ticker symbol (e.g., 'AAPL')
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Quote data in requested format (JSON or Markdown)
    """
    try:
        data = _get_ticker_data(params.symbol)
        
        if not data.get("name") and not data.get("current_price"):
            return f"Error: No data found for symbol '{params.symbol}'. Please verify the ticker is correct."
        
        if params.response_format == ResponseFormat.MARKDOWN:
            return _format_quote_markdown(data)
        return json.dumps(data, indent=2, default=str)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_multiple_quotes",
    annotations={
        "title": "Get Multiple Stock Quotes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_multiple_quotes(params: MultiTickerInput) -> str:
    """Get real-time quotes for multiple stocks at once.
    
    Efficiently retrieves current prices and key metrics for up to 10 stocks.
    
    Args:
        params (MultiTickerInput): Input containing:
            - symbols (List[str]): List of ticker symbols
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Quotes data for all requested symbols
    """
    try:
        results = _get_multiple_ticker_data(params.symbols)
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md_output = "# Stock Quotes\n\n"
            for symbol, data in results.items():
                if "error" in data:
                    md_output += f"## {symbol}\n\n⚠️ Error: {data['error']}\n\n"
                else:
                    md_output += _format_quote_markdown(data) + "\n---\n\n"
            return md_output
        
        return json.dumps(results, indent=2, default=str)
        
    except Exception as e:
        return _handle_error(e)

@mcp.tool(
    name="yf_get_historical_data",
    annotations={
        "title": "Get Historical Price Data",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_historical_data(params: HistoricalDataInput) -> str:
    """Get historical OHLCV (Open, High, Low, Close, Volume) data for a stock.
    
    Retrieves historical price data with customizable period and interval.
    Useful for backtesting, technical analysis, and charting.
    
    Args:
        params (HistoricalDataInput): Input containing:
            - symbol (str): Stock ticker symbol
            - period (Period): Data period (1d to max)
            - interval (Interval): Data interval (1m to 3mo)
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Historical OHLCV data with timestamps
    """
    try:
        ticker = yf.Ticker(params.symbol)
        hist = ticker.history(period=params.period.value, interval=params.interval.value)
        
        if hist.empty:
            return f"Error: No historical data found for '{params.symbol}' with period={params.period.value}, interval={params.interval.value}"
        
        # Convert to list of records
        records = []
        for date, row in hist.iterrows():
            records.append({
                "date": date.isoformat(),
                "open": round(row["Open"], 4),
                "high": round(row["High"], 4),
                "low": round(row["Low"], 4),
                "close": round(row["Close"], 4),
                "volume": int(row["Volume"]),
            })
        
        result = {
            "symbol": params.symbol,
            "period": params.period.value,
            "interval": params.interval.value,
            "data_points": len(records),
            "start_date": records[0]["date"] if records else None,
            "end_date": records[-1]["date"] if records else None,
            "data": records
        }
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Historical Data: {params.symbol}\n\n"
            md += f"**Period:** {params.period.value} | **Interval:** {params.interval.value}\n"
            md += f"**Data Points:** {len(records)} | **Range:** {result['start_date'][:10]} to {result['end_date'][:10]}\n\n"
            md += "| Date | Open | High | Low | Close | Volume |\n"
            md += "|------|------|------|-----|-------|--------|\n"
            # Show last 20 records for readability
            for r in records[-20:]:
                md += f"| {r['date'][:10]} | {r['open']:.2f} | {r['high']:.2f} | {r['low']:.2f} | {r['close']:.2f} | {r['volume']:,} |\n"
            if len(records) > 20:
                md += f"\n*Showing last 20 of {len(records)} records*\n"
            return md
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_financials",
    annotations={
        "title": "Get Financial Statements",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_financials(params: TickerInput) -> str:
    """Get financial statements for a company.
    
    Retrieves income statement, balance sheet, and cash flow data.
    
    Args:
        params (TickerInput): Input containing:
            - symbol (str): Stock ticker symbol
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Financial statements data (income, balance sheet, cash flow)
    """
    try:
        ticker = yf.Ticker(params.symbol)
        
        # Get financial statements
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        
        def df_to_dict(df):
            if df is None or df.empty:
                return {}
            result = {}
            for col in df.columns:
                date_str = col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col)
                result[date_str] = {
                    str(idx): float(val) if not (val != val) else None  # Handle NaN
                    for idx, val in df[col].items()
                }
            return result
        
        result = {
            "symbol": params.symbol,
            "income_statement": df_to_dict(income_stmt),
            "balance_sheet": df_to_dict(balance_sheet),
            "cash_flow": df_to_dict(cash_flow)
        }
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Financial Statements: {params.symbol}\n\n"
            
            # Income Statement highlights
            if income_stmt is not None and not income_stmt.empty:
                latest = income_stmt.columns[0]
                md += "### Income Statement (Latest)\n\n"
                key_metrics = ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"]
                for metric in key_metrics:
                    if metric in income_stmt.index:
                        val = income_stmt.loc[metric, latest]
                        md += f"- **{metric}:** {_format_number(val)}\n"
                md += "\n"
            
            # Balance Sheet highlights
            if balance_sheet is not None and not balance_sheet.empty:
                latest = balance_sheet.columns[0]
                md += "### Balance Sheet (Latest)\n\n"
                key_metrics = ["Total Assets", "Total Liabilities Net Minority Interest", "Total Equity Gross Minority Interest"]
                for metric in key_metrics:
                    if metric in balance_sheet.index:
                        val = balance_sheet.loc[metric, latest]
                        md += f"- **{metric}:** {_format_number(val)}\n"
                md += "\n"
            
            return md
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_options",
    annotations={
        "title": "Get Options Chain",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_options(params: OptionsInput) -> str:
    """Get options chain data for a stock.
    
    Retrieves calls and puts with strike prices, premiums, IV, OI, and volume.
    
    Args:
        params (OptionsInput): Input containing:
            - symbol (str): Stock ticker symbol
            - expiration_date (str, optional): Specific expiration date
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Options chain with calls and puts data
    """
    try:
        ticker = yf.Ticker(params.symbol)
        
        # Get available expiration dates
        expirations = ticker.options
        if not expirations:
            return f"Error: No options data available for '{params.symbol}'"
        
        # Select expiration date
        exp_date = params.expiration_date if params.expiration_date in expirations else expirations[0]
        
        # Get options chain
        opt = ticker.option_chain(exp_date)
        
        def options_df_to_list(df):
            records = []
            for _, row in df.iterrows():
                records.append({
                    "contract": row.get("contractSymbol", ""),
                    "strike": float(row.get("strike", 0)),
                    "last_price": float(row.get("lastPrice", 0)),
                    "bid": float(row.get("bid", 0)),
                    "ask": float(row.get("ask", 0)),
                    "volume": int(row.get("volume", 0)) if not (row.get("volume") != row.get("volume")) else 0,
                    "open_interest": int(row.get("openInterest", 0)) if not (row.get("openInterest") != row.get("openInterest")) else 0,
                    "implied_volatility": float(row.get("impliedVolatility", 0)),
                    "in_the_money": bool(row.get("inTheMoney", False))
                })
            return records
        
        result = {
            "symbol": params.symbol,
            "expiration_date": exp_date,
            "available_expirations": list(expirations),
            "calls": options_df_to_list(opt.calls),
            "puts": options_df_to_list(opt.puts)
        }
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Options Chain: {params.symbol}\n\n"
            md += f"**Expiration:** {exp_date}\n"
            md += f"**Available Expirations:** {', '.join(expirations[:5])}{'...' if len(expirations) > 5 else ''}\n\n"
            
            md += "### Calls (Top 10 by Volume)\n\n"
            md += "| Strike | Last | Bid | Ask | Volume | OI | IV |\n"
            md += "|--------|------|-----|-----|--------|-------|------|\n"
            calls_sorted = sorted(result["calls"], key=lambda x: x["volume"], reverse=True)[:10]
            for c in calls_sorted:
                md += f"| {c['strike']:.2f} | {c['last_price']:.2f} | {c['bid']:.2f} | {c['ask']:.2f} | {c['volume']:,} | {c['open_interest']:,} | {c['implied_volatility']:.2%} |\n"
            
            md += "\n### Puts (Top 10 by Volume)\n\n"
            md += "| Strike | Last | Bid | Ask | Volume | OI | IV |\n"
            md += "|--------|------|-----|-----|--------|-------|------|\n"
            puts_sorted = sorted(result["puts"], key=lambda x: x["volume"], reverse=True)[:10]
            for p in puts_sorted:
                md += f"| {p['strike']:.2f} | {p['last_price']:.2f} | {p['bid']:.2f} | {p['ask']:.2f} | {p['volume']:,} | {p['open_interest']:,} | {p['implied_volatility']:.2%} |\n"
            
            return md
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_dividends",
    annotations={
        "title": "Get Dividend History",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_dividends(params: TickerInput) -> str:
    """Get dividend history and information for a stock.
    
    Retrieves historical dividend payments and current yield information.
    
    Args:
        params (TickerInput): Input containing:
            - symbol (str): Stock ticker symbol
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Dividend history and yield information
    """
    try:
        ticker = yf.Ticker(params.symbol)
        dividends = ticker.dividends
        info = ticker.info
        
        dividend_records = []
        if not dividends.empty:
            for date, amount in dividends.items():
                dividend_records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "amount": float(amount)
                })
        
        result = {
            "symbol": params.symbol,
            "dividend_yield": _safe_get(info, "dividendYield"),
            "dividend_rate": _safe_get(info, "dividendRate"),
            "ex_dividend_date": _safe_get(info, "exDividendDate"),
            "payout_ratio": _safe_get(info, "payoutRatio"),
            "five_year_avg_dividend_yield": _safe_get(info, "fiveYearAvgDividendYield"),
            "dividend_history": dividend_records[-20:] if dividend_records else []  # Last 20
        }
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Dividend Information: {params.symbol}\n\n"
            md += f"**Current Yield:** {_format_percentage(result['dividend_yield'])}\n"
            md += f"**Annual Rate:** ${result['dividend_rate'] or 'N/A'}\n"
            md += f"**Payout Ratio:** {_format_percentage(result['payout_ratio'])}\n"
            md += f"**5Y Avg Yield:** {result['five_year_avg_dividend_yield'] or 'N/A'}%\n\n"
            
            if dividend_records:
                md += "### Recent Dividend History\n\n"
                md += "| Date | Amount |\n"
                md += "|------|--------|\n"
                for d in dividend_records[-10:]:
                    md += f"| {d['date']} | ${d['amount']:.4f} |\n"
            else:
                md += "*No dividend history available*\n"
            
            return md
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_recommendations",
    annotations={
        "title": "Get Analyst Recommendations",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_recommendations(params: TickerInput) -> str:
    """Get analyst recommendations and price targets for a stock.
    
    Retrieves buy/sell/hold ratings and target price information.
    
    Args:
        params (TickerInput): Input containing:
            - symbol (str): Stock ticker symbol
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Analyst recommendations and price targets
    """
    try:
        ticker = yf.Ticker(params.symbol)
        info = ticker.info
        recommendations = ticker.recommendations
        price_data = _resolve_quote_price(info)
        
        rec_records = []
        if recommendations is not None and not recommendations.empty:
            for date, row in recommendations.tail(20).iterrows():
                rec_records.append({
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                    "firm": row.get("Firm", ""),
                    "to_grade": row.get("To Grade", ""),
                    "from_grade": row.get("From Grade", ""),
                    "action": row.get("Action", "")
                })
        
        result = {
            "symbol": params.symbol,
            "recommendation_key": _safe_get(info, "recommendationKey"),
            "recommendation_mean": _safe_get(info, "recommendationMean"),
            "number_of_analysts": _safe_get(info, "numberOfAnalystOpinions"),
            "target_high": _safe_get(info, "targetHighPrice"),
            "target_low": _safe_get(info, "targetLowPrice"),
            "target_mean": _safe_get(info, "targetMeanPrice"),
            "target_median": _safe_get(info, "targetMedianPrice"),
            "current_price": price_data["current_price"],
            "current_price_source": price_data["current_price_source"],
            "regular_market_price": price_data["regular_market_price"],
            "post_market_price": price_data["post_market_price"],
            "pre_market_price": price_data["pre_market_price"],
            "recent_recommendations": rec_records
        }
        
        if params.response_format == ResponseFormat.MARKDOWN:
            current = result['current_price'] or 0
            target_mean = result['target_mean'] or 0
            upside = ((target_mean - current) / current * 100) if current > 0 else 0
            
            md = f"## Analyst Recommendations: {params.symbol}\n\n"
            md += f"**Consensus:** {result['recommendation_key'] or 'N/A'} (Score: {result['recommendation_mean'] or 'N/A'}/5)\n"
            md += f"**Number of Analysts:** {result['number_of_analysts'] or 'N/A'}\n\n"
            
            md += "### Price Targets\n\n"
            md += f"| Metric | Price | vs Current |\n"
            md += f"|--------|-------|------------|\n"
            md += f"| Current | ${current:.2f} | - |\n"
            md += f"| Target Low | ${result['target_low'] or 0:.2f} | {((result['target_low'] or 0) - current) / current * 100:+.1f}% |\n"
            md += f"| Target Mean | ${target_mean:.2f} | {upside:+.1f}% |\n"
            md += f"| Target High | ${result['target_high'] or 0:.2f} | {((result['target_high'] or 0) - current) / current * 100:+.1f}% |\n"
            
            if rec_records:
                md += "\n### Recent Analyst Actions\n\n"
                md += "| Date | Firm | Rating | Action |\n"
                md += "|------|------|--------|--------|\n"
                for r in rec_records[-10:]:
                    md += f"| {r['date']} | {r['firm']} | {r['to_grade']} | {r['action']} |\n"
            
            return md
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_news",
    annotations={
        "title": "Get Stock News",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_news(params: TickerInput) -> str:
    """Get recent news articles related to a stock.
    
    Retrieves latest news headlines and summaries.
    
    Args:
        params (TickerInput): Input containing:
            - symbol (str): Stock ticker symbol
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Recent news articles with titles and links
    """
    try:
        ticker = yf.Ticker(params.symbol)
        news = ticker.news
        
        if not news:
            return f"No news found for '{params.symbol}'"
        
        news_records = []
        for item in news[:15]:
            # yfinance has changed the news payload structure over time.
            # Support both legacy top-level keys and the nested `content` schema.
            item = item if isinstance(item, dict) else {}
            content_raw = item.get("content")
            content = content_raw if isinstance(content_raw, dict) else {}

            provider_raw = content.get("provider")
            provider = provider_raw if isinstance(provider_raw, dict) else {}

            click_raw = content.get("clickThroughUrl")
            click = click_raw if isinstance(click_raw, dict) else {}

            canonical_raw = content.get("canonicalUrl")
            canonical = canonical_raw if isinstance(canonical_raw, dict) else {}

            title = (
                item.get("title")
                or content.get("title")
                or content.get("headline")
                or ""
            )
            publisher = (
                item.get("publisher")
                or item.get("provider")
                or provider.get("displayName")
                or provider.get("name")
                or ""
            )
            link = (
                item.get("link")
                or click.get("url")
                or canonical.get("url")
                or content.get("url")
                or ""
            )

            publish_raw = (
                item.get("providerPublishTime")
                or content.get("pubDate")
                or content.get("publishedAt")
            )
            published = None
            if isinstance(publish_raw, (int, float)):
                ts = float(publish_raw)
                # Some payloads come in milliseconds.
                if ts > 1e12:
                    ts /= 1000.0
                if ts > 0:
                    published = datetime.fromtimestamp(ts).isoformat()
            elif isinstance(publish_raw, str) and publish_raw.strip():
                published = publish_raw

            thumb_raw = item.get("thumbnail") or content.get("thumbnail") or {}
            thumb = thumb_raw if isinstance(thumb_raw, dict) else {}
            thumbnail = None
            resolutions = thumb.get("resolutions") if isinstance(thumb, dict) else []
            if isinstance(resolutions, list) and resolutions:
                first_res = resolutions[0] if isinstance(resolutions[0], dict) else {}
                thumbnail = first_res.get("url") if isinstance(first_res, dict) else None

            news_records.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": published,
                "type": item.get("type") or content.get("contentType") or "",
                "thumbnail": thumbnail
            })
        
        result = {
            "symbol": params.symbol,
            "news_count": len(news_records),
            "news": news_records
        }
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Recent News: {params.symbol}\n\n"
            for n in news_records:
                pub_date = n['published'][:10] if n['published'] else "Unknown"
                md += f"### [{n['title']}]({n['link']})\n"
                md += f"*{n['publisher']} - {pub_date}*\n\n"
            return md
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_holders",
    annotations={
        "title": "Get Major Holders",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_holders(params: TickerInput) -> str:
    """Get major institutional and mutual fund holders for a stock.
    
    Retrieves ownership breakdown and top institutional holders.
    
    Args:
        params (TickerInput): Input containing:
            - symbol (str): Stock ticker symbol
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Major holders and institutional ownership data
    """
    try:
        ticker = yf.Ticker(params.symbol)
        
        # Get different holder types
        major_holders = ticker.major_holders
        institutional = ticker.institutional_holders
        mutualfund = ticker.mutualfund_holders
        
        def df_to_list(df):
            if df is None or df.empty:
                return []
            return df.to_dict('records')
        
        result = {
            "symbol": params.symbol,
            "major_holders": df_to_list(major_holders) if major_holders is not None else [],
            "institutional_holders": df_to_list(institutional),
            "mutualfund_holders": df_to_list(mutualfund)
        }
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Major Holders: {params.symbol}\n\n"
            
            if major_holders is not None and not major_holders.empty:
                md += "### Ownership Breakdown\n\n"
                for _, row in major_holders.iterrows():
                    md += f"- {row.iloc[1]}: **{row.iloc[0]}**\n"
                md += "\n"
            
            if institutional is not None and not institutional.empty:
                md += "### Top Institutional Holders\n\n"
                md += "| Holder | Shares | Value | % Out |\n"
                md += "|--------|--------|-------|-------|\n"
                for _, row in institutional.head(10).iterrows():
                    holder = row.get("Holder", "")
                    shares = row.get("Shares", 0)
                    value = row.get("Value", 0)
                    pct = row.get("% Out", 0)
                    md += f"| {holder[:30]} | {shares:,.0f} | {_format_number(value)} | {pct:.2f}% |\n"
            
            return md
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_earnings",
    annotations={
        "title": "Get Earnings Data",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_earnings(params: TickerInput) -> str:
    """Get earnings history and upcoming earnings dates.
    
    Retrieves EPS history, revenue, and earnings calendar.
    
    Args:
        params (TickerInput): Input containing:
            - symbol (str): Stock ticker symbol
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Earnings history and calendar data
    """
    try:
        ticker = yf.Ticker(params.symbol)
        
        earnings = ticker.earnings_history
        calendar = ticker.calendar
        
        earnings_records = []
        if earnings is not None and not earnings.empty:
            for date, row in earnings.iterrows():
                earnings_records.append({
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                    "eps_actual": float(row.get("epsActual", 0)) if row.get("epsActual") == row.get("epsActual") else None,
                    "eps_estimate": float(row.get("epsEstimate", 0)) if row.get("epsEstimate") == row.get("epsEstimate") else None,
                    "surprise_pct": float(row.get("surprisePercent", 0)) if row.get("surprisePercent") == row.get("surprisePercent") else None
                })
        
        result = {
            "symbol": params.symbol,
            "earnings_history": earnings_records,
            "calendar": calendar.to_dict() if calendar is not None and not calendar.empty else {}
        }
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Earnings: {params.symbol}\n\n"
            
            if earnings_records:
                md += "### Earnings History\n\n"
                md += "| Date | EPS Actual | EPS Est | Surprise |\n"
                md += "|------|------------|---------|----------|\n"
                for e in earnings_records[-8:]:
                    actual = f"${e['eps_actual']:.2f}" if e['eps_actual'] else "N/A"
                    est = f"${e['eps_estimate']:.2f}" if e['eps_estimate'] else "N/A"
                    surprise = f"{e['surprise_pct']:.1f}%" if e['surprise_pct'] else "N/A"
                    md += f"| {e['date']} | {actual} | {est} | {surprise} |\n"
            
            return md
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_calendar",
    annotations={
        "title": "Get Earnings Calendar",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_calendar(params: TickerInput) -> str:
    """Get upcoming earnings calendar information for a stock.
    
    Retrieves the Yahoo Finance calendar and (if available) detailed earnings dates.
    
    Args:
        params (TickerInput): Input containing:
            - symbol (str): Stock ticker symbol
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Calendar data in requested format
    """
    try:
        ticker = yf.Ticker(params.symbol)
        calendar = _calendar_to_dict(ticker.calendar)

        earnings_dates_records: List[Dict[str, Any]] = []
        if hasattr(ticker, "get_earnings_dates"):
            try:
                earnings_dates_records = _df_to_records(ticker.get_earnings_dates(limit=12))
            except Exception:
                earnings_dates_records = []

        result = {
            "symbol": params.symbol,
            "calendar": calendar,
            "earnings_dates": earnings_dates_records
        }

        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Earnings Calendar: {params.symbol}\n\n"

            next_date = None
            if isinstance(calendar, dict):
                for key in calendar.keys():
                    if "earnings date" in str(key).lower():
                        next_date = calendar.get(key)
                        break

            if not next_date and earnings_dates_records:
                first = earnings_dates_records[0]
                for key in ("Earnings Date", "Date", "index"):
                    if key in first:
                        next_date = first[key]
                        break

            if next_date:
                if isinstance(next_date, list) and len(next_date) == 2:
                    md += f"**Next Earnings Window:** {next_date[0]} to {next_date[1]}\n\n"
                else:
                    md += f"**Next Earnings Date:** {next_date}\n\n"
            else:
                md += "*No upcoming earnings date found.*\n\n"

            if calendar:
                md += "### Calendar Fields\n\n"
                md += "| Field | Value |\n"
                md += "|-------|-------|\n"
                for key, value in calendar.items():
                    md += f"| {key} | {_format_value_for_table(value)} |\n"
                md += "\n"

            if earnings_dates_records:
                md += "### Earnings Dates (Recent/Upcoming)\n\n"
                sample = earnings_dates_records[:6]
                keys = list(sample[0].keys())
                date_key = next((k for k in ("Earnings Date", "Date", "index") if k in keys), None)
                data_keys = [k for k in keys if k != date_key][:3]
                md += "| Date | " + " | ".join(data_keys) + " |\n"
                md += "|------|" + "|".join(["------"] * len(data_keys)) + "|\n"
                for row in sample:
                    date_val = row.get(date_key, "N/A") if date_key else "N/A"
                    md += f"| {date_val} | " + " | ".join(_format_value_for_table(row.get(k)) for k in data_keys) + " |\n"
            return md

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return _handle_error(e, f"Symbol: {params.symbol}")

@mcp.tool(
    name="yf_get_sector",
    annotations={
        "title": "Get Sector Data",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_sector(params: SectorKeyInput) -> str:
    """Get sector data and related aggregates from Yahoo Finance.

    Retrieves sector overview, industries, and top companies/ETFs/funds.

    Args:
        params (SectorKeyInput): Input containing:
            - key (str): Sector key (e.g., 'technology')
            - response_format (ResponseFormat): Output format preference

    Returns:
        str: Sector data in requested format
    """
    try:
        sector = yf.Sector(params.key)

        result = {
            "key": getattr(sector, "key", params.key),
            "name": getattr(sector, "name", None),
            "symbol": getattr(sector, "symbol", None),
            "overview": _normalize_value(getattr(sector, "overview", None)),
            "industries": _normalize_value(getattr(sector, "industries", None)),
            "top_companies": _df_to_records(getattr(sector, "top_companies", None)),
            "research_reports": _df_to_records(getattr(sector, "research_reports", None)),
            "top_etfs": _df_to_records(getattr(sector, "top_etfs", None)),
            "top_mutual_funds": _df_to_records(getattr(sector, "top_mutual_funds", None)),
        }

        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Sector: {result.get('name') or result.get('key')}\n\n"
            md += f"**Key:** {result.get('key')}\n"
            if result.get("symbol"):
                md += f"**Symbol:** {result.get('symbol')}\n"

            if result.get("overview"):
                md += "\n### Overview\n\n"
                overview = result["overview"]
                if isinstance(overview, dict):
                    for k, v in list(overview.items())[:8]:
                        md += f"- **{k}:** {_format_value_for_table(v)}\n"
                else:
                    md += f"{overview}\n"

            industries_val = result.get("industries")
            if industries_val:
                md += "\n### Industries\n\n"
                if (
                    isinstance(industries_val, list)
                    and industries_val
                    and isinstance(industries_val[0], dict)
                ):
                    md += _records_to_markdown_table(industries_val) + "\n"
                else:
                    if isinstance(industries_val, dict):
                        industries_list = list(industries_val.keys())
                    elif isinstance(industries_val, list):
                        industries_list = industries_val
                    else:
                        industries_list = [industries_val]
                    md += ", ".join(str(i) for i in industries_list[:20])
                    if len(industries_list) > 20:
                        md += " ..."
                    md += "\n"

            sections = [
                ("Top Companies", "top_companies"),
                ("Top ETFs", "top_etfs"),
                ("Top Mutual Funds", "top_mutual_funds"),
                ("Research Reports", "research_reports"),
            ]
            for title, key in sections:
                records = result.get(key) or []
                if records:
                    md += f"\n### {title}\n\n"
                    md += _records_to_markdown_table(records) + "\n"

            return md

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return _handle_error(e, f"Sector key: {params.key}")

@mcp.tool(
    name="yf_get_industry",
    annotations={
        "title": "Get Industry Data",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_get_industry(params: IndustryKeyInput) -> str:
    """Get industry data and related aggregates from Yahoo Finance.

    Retrieves industry overview and top companies.

    Args:
        params (IndustryKeyInput): Input containing:
            - key (str): Industry key (e.g., 'software-infrastructure')
            - response_format (ResponseFormat): Output format preference

    Returns:
        str: Industry data in requested format
    """
    try:
        industry = yf.Industry(params.key)

        result = {
            "key": getattr(industry, "key", params.key),
            "name": getattr(industry, "name", None),
            "sector_key": getattr(industry, "sector_key", None),
            "sector_name": getattr(industry, "sector_name", None),
            "overview": _normalize_value(getattr(industry, "overview", None)),
            "top_companies": _df_to_records(getattr(industry, "top_companies", None)),
            "top_performing_companies": _df_to_records(
                getattr(industry, "top_performing_companies", None)
            ),
            "top_growth_companies": _df_to_records(getattr(industry, "top_growth_companies", None)),
        }

        if params.response_format == ResponseFormat.MARKDOWN:
            md = f"## Industry: {result.get('name') or result.get('key')}\n\n"
            md += f"**Key:** {result.get('key')}\n"
            if result.get("sector_name") or result.get("sector_key"):
                md += f"**Sector:** {result.get('sector_name') or 'N/A'}"
                if result.get("sector_key"):
                    md += f" ({result.get('sector_key')})"
                md += "\n"

            if result.get("overview"):
                md += "\n### Overview\n\n"
                overview = result["overview"]
                if isinstance(overview, dict):
                    for k, v in list(overview.items())[:8]:
                        md += f"- **{k}:** {_format_value_for_table(v)}\n"
                else:
                    md += f"{overview}\n"

            sections = [
                ("Top Companies", "top_companies"),
                ("Top Performing Companies", "top_performing_companies"),
                ("Top Growth Companies", "top_growth_companies"),
            ]
            for title, key in sections:
                records = result.get(key) or []
                if records:
                    md += f"\n### {title}\n\n"
                    md += _records_to_markdown_table(records) + "\n"

            return md

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return _handle_error(e, f"Industry key: {params.key}")

@mcp.tool(
    name="yf_compare_stocks",
    annotations={
        "title": "Compare Multiple Stocks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
@with_timeout()
def yf_compare_stocks(params: MultiTickerInput) -> str:
    """Compare key metrics across multiple stocks.
    
    Creates a side-by-side comparison of fundamental metrics.
    
    Args:
        params (MultiTickerInput): Input containing:
            - symbols (List[str]): List of ticker symbols to compare
            - response_format (ResponseFormat): Output format preference
    
    Returns:
        str: Comparison table with key metrics
    """
    try:
        comparison = []
        for symbol, data in _get_multiple_ticker_data(params.symbols).items():
            if "error" in data:
                comparison.append({"symbol": symbol, "error": data["error"]})
                continue
            comparison.append({
                "symbol": symbol,
                "name": data.get("name"),
                "price": data.get("current_price"),
                "market_cap": data.get("market_cap"),
                "pe_ratio": data.get("pe_ratio"),
                "forward_pe": data.get("forward_pe"),
                "peg_ratio": data.get("peg_ratio"),
                "price_to_book": data.get("price_to_book"),
                "dividend_yield": data.get("dividend_yield"),
                "profit_margin": data.get("profit_margin"),
                "revenue_growth": data.get("revenue_growth"),
                "beta": data.get("beta"),
                "52_week_change": data.get("52_week_change"),
            })
        
        if params.response_format == ResponseFormat.MARKDOWN:
            md = "## Stock Comparison\n\n"
            
            # Header
            md += "| Metric |"
            for c in comparison:
                md += f" {c['symbol']} |"
            md += "\n|--------|" + "|".join(["--------"] * len(comparison)) + "|\n"
            
            # Rows
            metrics = [
                ("Price", "price", lambda x: f"${x:.2f}" if x else "N/A"),
                ("Market Cap", "market_cap", lambda x: _format_number(x)),
                ("P/E Ratio", "pe_ratio", lambda x: f"{x:.2f}" if x else "N/A"),
                ("Forward P/E", "forward_pe", lambda x: f"{x:.2f}" if x else "N/A"),
                ("PEG Ratio", "peg_ratio", lambda x: f"{x:.2f}" if x else "N/A"),
                ("P/B Ratio", "price_to_book", lambda x: f"{x:.2f}" if x else "N/A"),
                ("Div Yield", "dividend_yield", lambda x: f"{x*100:.2f}%" if x else "N/A"),
                ("Profit Margin", "profit_margin", lambda x: f"{x*100:.1f}%" if x else "N/A"),
                ("Rev Growth", "revenue_growth", lambda x: f"{x*100:.1f}%" if x else "N/A"),
                ("Beta", "beta", lambda x: f"{x:.2f}" if x else "N/A"),
                ("52W Change", "52_week_change", lambda x: f"{x*100:.1f}%" if x else "N/A"),
            ]
            
            for metric_name, metric_key, formatter in metrics:
                md += f"| {metric_name} |"
                for c in comparison:
                    if "error" in c:
                        md += " Error |"
                    else:
                        md += f" {formatter(c.get(metric_key))} |"
                md += "\n"
            
            return md
        
        return json.dumps({"comparison": comparison}, indent=2, default=str)
        
    except Exception as e:
        return _handle_error(e)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
