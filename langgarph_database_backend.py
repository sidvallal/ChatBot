from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

class ChatState(TypedDict):
    
    messages : Annotated[list[BaseMessage], add_messages]
    

def chat_node(state: ChatState):
    
    # take query from state
    messages = state['messages']
    
    # send to llm
    response = model.invoke(messages)
    
    # store response in state
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)


graph.add_node('chat_node',chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

# test
# CONFIG = {
#         "configurable": {
#             "thread_id": "thrad-2"
#         }
#     }

# res = chatbot.invoke(
#         {
#             "messages": [HumanMessage(content='What is my name')]},
#              config=CONFIG,
#         )

# print(res)

def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return [{"id": tid, "title": "New Chat"} for tid in all_threads]


