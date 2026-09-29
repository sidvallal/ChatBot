from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------------- SIDEBAR HELPERS ----------------------

def make_title(text, max_len=30):
    """Turn a message into a short sidebar title."""
    text = " ".join(text.split())  # remove newlines / extra spaces
    return text[:max_len] + ("..." if len(text) > max_len else "")


def get_thread_title(thread_id):
    """Use the first user message of a saved thread as its title."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    for msg in messages:
        if isinstance(msg, HumanMessage):
            return make_title(msg.content)

    return "New Chat"


def retrive_all_threads():
    """Return all saved threads (oldest first) with titles from the conversation."""
    thread_ids = {}  # dict keeps insertion order, unlike set()

    for checkpoint in checkpointer.list(None):
        thread_ids[checkpoint.config["configurable"]["thread_id"]] = None

    # checkpointer.list gives newest first -> reverse so oldest is first
    # (frontend uses reversed(), so the newest ends up on top)
    ordered_ids = list(thread_ids.keys())[::-1]

    return [{"id": tid, "title": get_thread_title(tid)} for tid in ordered_ids]