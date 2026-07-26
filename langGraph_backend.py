from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage,HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

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


checkpointer = InMemorySaver()
graph = StateGraph(ChatState)


graph.add_node('chat_node',chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)