from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano')

chat_history =[
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI:", response.content)

print("chat history:", chat_history)

