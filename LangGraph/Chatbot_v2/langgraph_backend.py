from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langgraph.graph.message import add_messages

load_dotenv()

model = ChatOpenAI()


class ChatState(TypedDict):
    response: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['response']
    response = model.invoke(messages).content
    return {"response": [response]}


checkpointer = InMemorySaver()


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)





