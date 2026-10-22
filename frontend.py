"""Streamlit frontend for the LangGraph MCP chatbot with document RAG."""

import uuid

import streamlit as st

from backend import (
    index_uploaded_file,
    load_conversation,
    make_title,
    retrieve_all_threads,
    save_thread_title,
    stream_reply,
)

# ---------------------- CONFIGURATION ----------------------

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEW_CHAT_TITLE = "New Chat"


# ---------------------- UTILITY FUNCTIONS ----------------------

def generate_thread_id() -> str:
    """Generate a unique ID for a conversation."""
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    """Add a thread to the session list if it does not exist."""
    if not any(
        thread["id"] == thread_id
        for thread in st.session_state["chat_threads"]
    ):
        st.session_state["chat_threads"].append(
            {"id": thread_id, "title": NEW_CHAT_TITLE}
        )


def get_current_thread_title() -> str:
    """Get the active conversation title."""
    thread_id = st.session_state["thread_id"]

    for thread in st.session_state["chat_threads"]:
        if thread["id"] == thread_id:
            return thread["title"]

    return NEW_CHAT_TITLE


def reset_chat() -> None:
    """Create a new conversation unless the current one is still empty."""
    st.session_state["last_error"] = None

    if (
        not st.session_state["messages_history"]
        and get_current_thread_title() == NEW_CHAT_TITLE
    ):
        return

    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["messages_history"] = []

    add_thread(thread_id)


def open_thread(thread_id: str) -> None:
    """Load an existing conversation."""
    st.session_state["thread_id"] = thread_id
    st.session_state["last_error"] = None

    try:
        st.session_state["messages_history"] = load_conversation(thread_id)
    except Exception as exc:
        st.session_state["messages_history"] = []
        st.session_state["last_error"] = (
            f"Unable to load conversation: {exc}"
        )


# ---------------------- SESSION SETUP ----------------------

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

add_thread(st.session_state["thread_id"])


# ---------------------- SIDEBAR ----------------------

with st.sidebar:
    st.title("🤖 LangGraph Chatbot")
    st.caption("Powered by LangGraph, MCP, Groq, and Gemini")

    st.divider()

    if st.button(
        "➕ New Chat",
        use_container_width=True,
        type="primary",
    ):
        reset_chat()
        st.rerun()

    st.divider()

    # Document upload and indexing.
    st.subheader("📄 Upload a Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, TXT, or DOCX file",
        type=["pdf", "txt", "docx"],
        key="document_uploader",
    )

    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")

        if st.button(
            "📥 Index Document",
            use_container_width=True,
            key="index_document",
        ):
            try:
                with st.spinner(
                    "Extracting text and creating Gemini embeddings..."
                ):
                    chunk_count = index_uploaded_file(
                        st.session_state["thread_id"],
                        uploaded_file,
                    )

                st.success(
                    f"Document indexed successfully! "
                    f"{chunk_count} chunks created."
                )

            except Exception as exc:
                st.error(f"Unable to index document: {exc}")

    st.caption(
        "Documents are indexed for the currently selected conversation."
    )

    st.divider()

    # Conversation list.
    st.subheader("💬 My Conversations")

    if not st.session_state["chat_threads"]:
        st.caption("No conversations yet.")

    for thread in reversed(st.session_state["chat_threads"]):
        is_active = thread["id"] == st.session_state["thread_id"]

        if st.button(
            thread["title"],
            key=f"thread_{thread['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                open_thread(thread["id"])
                st.rerun()


# ---------------------- MAIN CHAT WINDOW ----------------------

st.title("LangGraph Chatbot")
st.caption(
    "Ask questions, perform calculations, search the web, "
    "and ask questions about your uploaded documents."
)

# Display conversation history.
for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display errors from previous operations.
if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])


# ---------------------- HANDLE USER INPUT ----------------------

user_input = st.chat_input("Type your message...")

if user_input:
    thread_id = st.session_state["thread_id"]
    st.session_state["last_error"] = None

    # Give the conversation a title when its first message is sent.
    for thread in st.session_state["chat_threads"]:
        if (
            thread["id"] == thread_id
            and thread["title"] == NEW_CHAT_TITLE
        ):
            try:
                title = make_title(user_input)
                thread["title"] = title
                save_thread_title(thread_id, title)
            except Exception:
                # Chat generation should still proceed if title generation fails.
                pass
            break

    # Add and display the user's message.
    st.session_state["messages_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream the assistant response.
    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(
                stream_reply(thread_id, user_input)
            )

            st.session_state["messages_history"].append(
                {"role": "assistant", "content": ai_message}
            )

        except Exception as exc:
            st.session_state["last_error"] = (
                f"Something went wrong: {exc}"
            )

    # Refresh titles and conversation history.
    st.rerun()
