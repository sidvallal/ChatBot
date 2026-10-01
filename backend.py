"""LangGraph backend using MCP tools, Groq, and SQLite persistence.

Design notes
------------
* Everything async (MCP session, LangGraph, SQLite) runs on ONE long-lived
  event loop living in a background thread. Streamlit (sync) talks to it via
  the small bridges at the bottom of this file, so no object is ever used on
  a different or closed loop.
* The checkpointer is ``AsyncSqliteSaver`` because the graph is driven with
  ``astream`` / ``aget_state``.
* The MCP stdio server is started once and kept alive by a dedicated task
  (instead of spawning a new subprocess per tool call).
* SQLite runs in WAL mode with a busy timeout to avoid "database is locked".
"""

import asyncio
import atexit
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Iterator, TypedDict

import aiosqlite
from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ---------------------- CONFIGURATION ----------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chatbot.db"
SERVER_PATH = BASE_DIR / "mcp_server.py"

GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant. Use the calculator tool for arithmetic, "
        "the stock price tool for stock quotes, currency conversion for "
        "exchange rates, and web search for current information or facts "
        "you are unsure about. If a tool returns an error, explain it "
        "briefly instead of guessing."
    )
)

# ---------------------- BACKGROUND EVENT LOOP ----------------------

_loop = asyncio.new_event_loop()


def _run_loop() -> None:
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_run_loop, name="backend-loop", daemon=True).start()


def run_async(coro) -> Any:
    """Run a coroutine on the backend loop and block for its result."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ---------------------- SHARED RESOURCES ----------------------

# These are only ever touched from the backend loop.
_init_lock = asyncio.Lock()
_title_lock = asyncio.Lock()

_chatbot = None
_checkpoint_conn: aiosqlite.Connection | None = None
_title_conn: aiosqlite.Connection | None = None
_mcp_task: asyncio.Task | None = None
_mcp_stop: asyncio.Event | None = None

mcp_client = MultiServerMCPClient(
    {
        "chatbot_tools": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVER_PATH)],
        }
    }
)


async def _connect() -> aiosqlite.Connection:
    """Open a SQLite connection configured for concurrent use."""
    conn = await aiosqlite.connect(str(DB_PATH), timeout=30)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=30000")
    await conn.commit()
    return conn


# ---------------------- MCP SESSION ----------------------

async def _start_mcp() -> list:
    """Start the MCP server once and keep its session open.

    A dedicated task owns the session so that it is entered and exited in
    the same task (required by the underlying anyio task groups).
    """
    global _mcp_task, _mcp_stop

    ready: asyncio.Future = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()

    async def owner() -> None:
        try:
            async with mcp_client.session("chatbot_tools") as session:
                tools = await load_mcp_tools(session)
                if not ready.done():
                    ready.set_result(tools)
                await stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)

    _mcp_stop = stop
    _mcp_task = asyncio.create_task(owner(), name="mcp-session")

    return await ready


# ---------------------- LANGGRAPH STATE ----------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------- GRAPH INITIALIZATION ----------------------

async def initialize_chatbot():
    """Load MCP tools and build the LangGraph chatbot once."""
    global _chatbot, _checkpoint_conn

    if _chatbot is not None:
        return _chatbot

    async with _init_lock:
        if _chatbot is not None:
            return _chatbot

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )

        if not SERVER_PATH.exists():
            raise FileNotFoundError(
                f"MCP server file not found: {SERVER_PATH}"
            )

        tools = await _start_mcp()

        if not tools:
            await _stop_mcp()
            raise RuntimeError(
                "No MCP tools were discovered. Check mcp_server.py."
            )

        try:
            conn = await _connect()
            checkpointer = AsyncSqliteSaver(conn)
            await checkpointer.setup()
        except Exception:
            await _stop_mcp()
            raise

        model = ChatGroq(model=GROQ_MODEL, temperature=0)
        llm_with_tools = model.bind_tools(tools)

        async def chat_node(state: ChatState):
            """Invoke the LLM with conversation history."""
            messages = [SYSTEM_PROMPT, *state["messages"]]
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        graph_builder = StateGraph(ChatState)
        graph_builder.add_node("chat_node", chat_node)
        graph_builder.add_node("tools", ToolNode(tools))

        graph_builder.add_edge(START, "chat_node")
        graph_builder.add_conditional_edges("chat_node", tools_condition)
        graph_builder.add_edge("tools", "chat_node")

        _checkpoint_conn = conn
        _chatbot = graph_builder.compile(checkpointer=checkpointer)

        print(f"Chatbot initialized with {len(tools)} MCP tools.")
        return _chatbot


# ---------------------- THREAD TITLES ----------------------

async def _get_title_conn() -> aiosqlite.Connection:
    global _title_conn

    async with _title_lock:
        if _title_conn is None:
            conn = await _connect()
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_titles (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.commit()
            _title_conn = conn
        return _title_conn


async def _save_thread_title(thread_id: str, title: str) -> None:
    conn = await _get_title_conn()
    await conn.execute(
        "INSERT OR IGNORE INTO thread_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title),
    )
    await conn.commit()


async def _retrieve_all_threads() -> list[dict]:
    conn = await _get_title_conn()
    async with conn.execute(
        """
        SELECT thread_id, title
        FROM thread_titles
        ORDER BY created_at, rowid
        """
    ) as cursor:
        rows = await cursor.fetchall()

    return [{"id": thread_id, "title": title} for thread_id, title in rows]


def make_title(text: str, max_len: int = 30) -> str:
    """Create a short conversation title."""
    text = " ".join(text.split())
    return text[:max_len] + ("..." if len(text) > max_len else "")


def save_thread_title(thread_id: str, title: str) -> None:
    """Save a title for a conversation."""
    run_async(_save_thread_title(thread_id, title))


def retrieve_all_threads() -> list[dict]:
    """Return all saved conversations."""
    return run_async(_retrieve_all_threads())


# ---------------------- CONVERSATION HELPERS ----------------------

def _text(content: Any) -> str:
    """Flatten message content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


async def _load_conversation(thread_id: str) -> list[dict]:
    """Load saved messages for the Streamlit UI.

    All assistant text produced between two user messages (including text
    emitted before a tool call) is merged into a single assistant message,
    exactly as it appears while streaming live.
    """
    graph = await initialize_chatbot()

    state = await graph.aget_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    history: list[dict] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            history.append(
                {"role": "assistant", "content": "\n\n".join(buffer)}
            )
            buffer.clear()

    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            flush()
            history.append({"role": "user", "content": _text(msg.content)})

        elif isinstance(msg, AIMessage):
            text = _text(msg.content).strip()
            if text:
                buffer.append(text)

    flush()
    return history


async def _stream_reply(
    thread_id: str,
    user_input: str,
) -> AsyncIterator[str]:
    """Stream the assistant response and execute MCP tools."""
    graph = await initialize_chatbot()

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    last_id = None
    emitted = False

    async for chunk, metadata in graph.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") != "chat_node":
            continue

        if not isinstance(chunk, AIMessageChunk):
            continue

        text = _text(chunk.content)
        if not text:
            continue

        # A new model call (e.g. after a tool ran) starts a new paragraph.
        if emitted and last_id is not None and chunk.id != last_id:
            yield "\n\n"

        last_id = chunk.id
        emitted = True
        yield text


# ---------------------- SYNC BRIDGES FOR STREAMLIT ----------------------

_DONE = object()


def load_conversation(thread_id: str) -> list[dict]:
    """Sync wrapper: load saved messages for a thread."""
    return run_async(_load_conversation(thread_id))


def stream_reply(thread_id: str, user_input: str) -> Iterator[str]:
    """Sync generator: stream the assistant reply token by token."""
    q: queue.Queue = queue.Queue()

    async def pump() -> None:
        try:
            async for token in _stream_reply(thread_id, user_input):
                q.put(token)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    future = asyncio.run_coroutine_threadsafe(pump(), _loop)

    try:
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # If Streamlit interrupts the script mid-stream, stop the work.
        if not future.done():
            future.cancel()


# ---------------------- CLEANUP ----------------------

async def _stop_mcp() -> None:
    global _mcp_task, _mcp_stop

    if _mcp_stop is not None:
        _mcp_stop.set()
    if _mcp_task is not None:
        try:
            await asyncio.wait_for(_mcp_task, timeout=5)
        except Exception:
            _mcp_task.cancel()
    _mcp_task = None
    _mcp_stop = None


async def _shutdown() -> None:
    await _stop_mcp()
    for conn in (_checkpoint_conn, _title_conn):
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass


def close_database() -> None:
    """Stop the MCP server and close SQLite connections."""
    try:
        asyncio.run_coroutine_threadsafe(_shutdown(), _loop).result(timeout=10)
    except Exception:
        pass


atexit.register(close_database)