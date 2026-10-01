"""Streamlit frontend for the LangGraph chatbot."""

import uuid

import streamlit as st

from backend import (
    load_conversation,
    make_title,
    retrieve_all_threads,
    save_thread_title,
    stream_reply,
)

NEW_CHAT_TITLE = "New Chat"

# ---------------------- UTILITY FUNCTIONS ----------------------


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    """Add a thread to the sidebar list if it isn't there already."""
    if any(t["id"] == thread_id for t in st.session_state["chat_threads"]):
        return
    st.session_state["chat_threads"].append({"id": thread_id, "title": NEW_CHAT_TITLE})


def reset_chat() -> None:
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["messages_history"] = []
    add_thread(thread_id)


def open_thread(thread_id: str) -> None:
    st.session_state["thread_id"] = thread_id
    st.session_state["messages_history"] = load_conversation(thread_id)


# ---------------------- SESSION SETUP ----------------------

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

add_thread(st.session_state["thread_id"])

# ---------------------- SIDEBAR ----------------------

st.sidebar.title("LangGraph ChatBot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

# Latest chats on top
for thread in reversed(st.session_state["chat_threads"]):
    is_active = thread["id"] == st.session_state["thread_id"]

    if st.sidebar.button(
        thread["title"],
        key=thread["id"],
        type="primary" if is_active else "secondary",
        use_container_width=True,
    ):
        open_thread(thread["id"])
        st.rerun()

# ---------------------- CHAT WINDOW ----------------------

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

# ---------------------- HANDLE USER MESSAGE ----------------------

if user_input:
    thread_id = st.session_state["thread_id"]

    # First message of a chat sets (and saves) its sidebar title
    for thread in st.session_state["chat_threads"]:
        if thread["id"] == thread_id and thread["title"] == NEW_CHAT_TITLE:
            thread["title"] = make_title(user_input)
            save_thread_title(thread_id, thread["title"])
            break

    st.session_state["messages_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(stream_reply(thread_id, user_input))
        except Exception as e:
            ai_message = f"Sorry, something went wrong: {e}"
            st.error(ai_message)

    st.session_state["messages_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    st.rerun()  # refresh the sidebar title