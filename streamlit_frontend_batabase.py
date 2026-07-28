import streamlit as st
from langgarph_database_backend import chatbot, retrive_all_threads
from langchain_core.messages import HumanMessage
import uuid

# ---------------------- UTILITY FUNCTIONS ----------------------

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    # Prevent duplicate threads
    for thread in st.session_state["chat_threads"]:
        if thread["id"] == thread_id:
            return

    st.session_state["chat_threads"].append(
        {
            "id": thread_id,
            "title": "New Chat"
        }
    )


def reset_chat():
    new_thread = generate_thread_id()

    st.session_state["thread_id"] = new_thread
    st.session_state["messages_history"] = []

    add_thread(new_thread)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    return state.values.get("messages", [])


# ---------------------- SESSION SETUP ----------------------

if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrive_all_threads()

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

    if st.sidebar.button(thread["title"], key=thread["id"]):

        st.session_state["thread_id"] = thread["id"]

        messages = load_conversation(thread["id"])

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append(
                {
                    "role": role,
                    "content": msg.content
                }
            )

        st.session_state["messages_history"] = temp_messages

        st.rerun()

# ---------------------- CHAT WINDOW ----------------------

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

# ---------------------- USER MESSAGE ----------------------

if user_input:

    # Update sidebar title using first message
    for thread in st.session_state["chat_threads"]:
        if (
            thread["id"] == st.session_state["thread_id"]
            and thread["title"] == "New Chat"
        ):
            thread["title"] = user_input[:30]
            break

    st.session_state["messages_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            (
                chunk.content
                for chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                )
                if chunk.content
            )
        )

    st.session_state["messages_history"].append(
        {
            "role": "assistant",
            "content": ai_message,
        }
    )

    st.rerun()