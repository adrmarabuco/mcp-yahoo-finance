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
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any

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

def _ticker_to_dict(ticker: yf.Ticker) -> Dict[str, Any]:
    """Convert ticker info to a clean dictionary."""
    info = ticker.info
    return {
        "symbol": _safe_get(info, "symbol"),
        "name": _safe_get(info, "longName") or _safe_get(info, "shortName"),
        "currency": _safe_get(info, "currency"),
        "exchange": _safe_get(info, "exchange"),
        "market_cap": _safe_get(info, "marketCap"),
        "current_price": _safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice"),
        "previous_close": _safe_get(info, "previousClose"),
        "open": _safe_get(info, "open") or _safe_get(info, "regularMarketOpen"),
        "day_high": _safe_get(info, "dayHigh") or _safe_get(info, "regularMarketDayHigh"),
        "day_low": _safe_get(info, "dayLow") or _safe_get(info, "regularMarketDayLow"),
        "volume": _safe_get(info, "volume") or _safe_get(info, "regularMarketVolume"),
        "avg_volume": _safe_get(info, "averageVolume"),
        "52_week_high": _safe_get(info, "fiftyTwoWeekHigh"),
        "52_week_low": _safe_get(info, "fiftyTwoWeekLow"),
        "pe_ratio": _safe_get(info, "trailingPE"),
        "forward_pe": _safe_get(info, "forwardPE"),
        "eps": _safe_get(info, "trailingEps"),
        "dividend_yield": _safe_get(info, "dividendYield"),
        "dividend_rate": _safe_get(info, "dividendRate"),
        "beta": _safe_get(info, "beta"),
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
    prev_close = data.get("previous_close")
    
    change = ""
    if price != "N/A" and prev_close:
        price_change = price - prev_close
        pct_change = (price_change / prev_close) * 100
        direction = "🟢" if price_change >= 0 else "🔴"
        change = f" {direction} {price_change:+.2f} ({pct_change:+.2f}%)"
    
    return f"""## {data.get('name', 'Unknown')} ({data.get('symbol', 'N/A')})

**Price:** ${price}{change}

| Metric | Value |
|--------|-------|
| Open | ${data.get('open', 'N/A')} |
| Day High | ${data.get('day_high', 'N/A')} |
| Day Low | ${data.get('day_low', 'N/A')} |
| Volume | {data.get('volume', 'N/A'):,} |
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
async def yf_get_quote(params: TickerInput) -> str:
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
        ticker = yf.Ticker(params.symbol)
        data = _ticker_to_dict(ticker)
        
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
async def yf_get_multiple_quotes(params: MultiTickerInput) -> str:
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
        results = {}
        for symbol in params.symbols:
            try:
                ticker = yf.Ticker(symbol)
                results[symbol] = _ticker_to_dict(ticker)
            except Exception as e:
                results[symbol] = {"error": str(e)}
        
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
async def yf_get_historical_data(params: HistoricalDataInput) -> str:
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
async def yf_get_financials(params: TickerInput) -> str:
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
async def yf_get_options(params: OptionsInput) -> str:
    """Get options chain data for a stock.
    
    Retrieves calls and puts with strike prices, premiums, and Greeks.
    
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
async def yf_get_dividends(params: TickerInput) -> str:
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
async def yf_get_recommendations(params: TickerInput) -> str:
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
            "current_price": _safe_get(info, "currentPrice"),
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
async def yf_get_news(params: TickerInput) -> str:
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
            news_records.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat() if item.get("providerPublishTime") else None,
                "type": item.get("type", ""),
                "thumbnail": item.get("thumbnail", {}).get("resolutions", [{}])[0].get("url") if item.get("thumbnail") else None
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
async def yf_get_holders(params: TickerInput) -> str:
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
async def yf_get_earnings(params: TickerInput) -> str:
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
    name="yf_compare_stocks",
    annotations={
        "title": "Compare Multiple Stocks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def yf_compare_stocks(params: MultiTickerInput) -> str:
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
        
        for symbol in params.symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                comparison.append({
                    "symbol": symbol,
                    "name": _safe_get(info, "shortName"),
                    "price": _safe_get(info, "currentPrice") or _safe_get(info, "regularMarketPrice"),
                    "market_cap": _safe_get(info, "marketCap"),
                    "pe_ratio": _safe_get(info, "trailingPE"),
                    "forward_pe": _safe_get(info, "forwardPE"),
                    "peg_ratio": _safe_get(info, "pegRatio"),
                    "price_to_book": _safe_get(info, "priceToBook"),
                    "dividend_yield": _safe_get(info, "dividendYield"),
                    "profit_margin": _safe_get(info, "profitMargins"),
                    "revenue_growth": _safe_get(info, "revenueGrowth"),
                    "beta": _safe_get(info, "beta"),
                    "52_week_change": _safe_get(info, "52WeekChange")
                })
            except Exception as e:
                comparison.append({"symbol": symbol, "error": str(e)})
        
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
