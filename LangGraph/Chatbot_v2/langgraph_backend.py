from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

model = ChatOpenAI()


class ChatState(TypedDict):
    response: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['response']
    response = model.invoke(messages).content
    return {"response": [response]}

conn = sqlite3.connect('chatbot_v2.db', check_same_thread=False)

checkpointer = SqliteSaver(conn = conn)


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
                        
    return list(all_threads)





