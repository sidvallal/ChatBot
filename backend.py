"""LangGraph chatbot backend: tools, graph, SQLite persistence, thread helpers."""

import os
import sqlite3
import threading
from typing import Annotated, Iterator, Literal, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

DB_PATH = "chatbot.db"

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant. Use the calculator tool for any arithmetic, "
        "the stock price tool for stock quotes, and web search for current events "
        "or facts you are unsure about. If a tool returns an error, explain it "
        "briefly to the user instead of guessing."
    )
)

# ---------------------- TOOLS ----------------------


@tool
def calculator(
    op1: float,
    op2: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
) -> float | str:
    """Perform a basic calculation on two numbers."""
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


_ddg = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for current information and recent events."""
    try:
        return _ddg.invoke(query)
    except Exception as e:
        return f"Error: web search is unavailable right now ({e})."


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock quote for a ticker symbol using Alpha Vantage."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is not set"}

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        return {"error": f"Stock request failed: {e}"}

    quote = data.get("Global Quote")
    if quote:
        return quote
    # Alpha Vantage returns HTTP 200 with a Note/Information key when rate limited
    return {"error": data.get("Note") or data.get("Information") or "No data found"}

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies using the latest exchange rate (e.g. USD to INR)."""
    try:
        response = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={
                "base": from_currency.upper(),
                "symbols": to_currency.upper(),
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        rate = data["rates"][to_currency.upper()]
    except (requests.RequestException, KeyError, ValueError) as e:
        return {"error": f"Currency conversion failed: {e}"}

    return {
        "amount": amount,
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "rate": rate,
        "converted": round(amount * rate, 2),
        "rate_date": data.get("date"),
    }


tools = [calculator, web_search, get_stock_price, convert_currency]

# ---------------------- GRAPH ----------------------

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    # The system prompt is prepended at call time so it is not saved in the history
    messages = [SYSTEM_PROMPT, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ---------------------- THREAD TITLES (separate table) ----------------------

_title_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_title_lock = threading.Lock()

with _title_lock:
    _title_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _title_conn.commit()


def make_title(text: str, max_len: int = 30) -> str:
    """Turn a message into a short sidebar title."""
    text = " ".join(text.split())
    return text[:max_len] + ("..." if len(text) > max_len else "")


def save_thread_title(thread_id: str, title: str) -> None:
    """Store a title for a thread (kept if one already exists)."""
    with _title_lock:
        _title_conn.execute(
            "INSERT OR IGNORE INTO thread_titles (thread_id, title) VALUES (?, ?)",
            (thread_id, title),
        )
        _title_conn.commit()


def retrieve_all_threads() -> list[dict]:
    """Return all saved threads, oldest first, as {'id', 'title'} dicts."""
    with _title_lock:
        rows = _title_conn.execute(
            "SELECT thread_id, title FROM thread_titles ORDER BY created_at, rowid"
        ).fetchall()
    return [{"id": thread_id, "title": title} for thread_id, title in rows]


# ---------------------- CONVERSATION HELPERS ----------------------


def load_conversation(thread_id: str) -> list[dict]:
    """Load a saved thread as UI-ready messages (no tool messages)."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            history.append({"role": "assistant", "content": msg.content})
    return history


def stream_reply(thread_id: str, user_input: str) -> Iterator[str]:
    """Yield the assistant's reply text chunk by chunk."""
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    for chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") != "chat_node":
            continue
        if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str):
            if chunk.content:
                yield chunk.content