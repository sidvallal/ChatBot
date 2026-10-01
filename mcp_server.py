"""MCP tool server (stdio) for the LangGraph chatbot."""

import os
from typing import Literal

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

mcp = FastMCP("Chatbot Tools")
_ddg = DuckDuckGoSearchRun()


@mcp.tool()
def calculator(
    op1: float,
    op2: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> float | str:
    """Perform basic arithmetic: add, subtract, multiply, or divide."""

    operation = operation.lower().strip()

    if operation == "add":
        return op1 + op2
    if operation == "subtract":
        return op1 - op2
    if operation == "multiply":
        return op1 * op2
    if operation == "divide":
        if op2 == 0:
            return "Error: cannot divide by zero"
        return op1 / op2

    return "Error: invalid operation"


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for current information and recent events."""
    try:
        return _ddg.invoke(query)
    except Exception as e:
        return f"Error: web search is unavailable right now ({e})"


@mcp.tool()
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock quote using Alpha Vantage."""

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is not set"}

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.strip().upper(),
                "apikey": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        return {"error": f"Stock request failed: {e}"}

    quote = data.get("Global Quote")
    if quote:
        return quote

    return {
        "error": data.get("Note")
        or data.get("Information")
        or "No data found"
    }


@mcp.tool()
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> dict:
    """Convert currency using the latest available exchange rate."""

    source = from_currency.strip().upper()
    target = to_currency.strip().upper()

    if source == target:
        return {
            "amount": amount,
            "from": source,
            "to": target,
            "rate": 1.0,
            "converted": round(amount, 2),
            "rate_date": None,
        }

    try:
        response = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": source, "symbols": target},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        rate = data["rates"][target]
    except (requests.RequestException, KeyError, ValueError) as e:
        return {"error": f"Currency conversion failed: {e}"}

    return {
        "amount": amount,
        "from": source,
        "to": target,
        "rate": rate,
        "converted": round(amount * rate, 2),
        "rate_date": data.get("date"),
    }


if __name__ == "__main__":
    # stdio is used by the MCP client.
    mcp.run(transport="stdio")