from re import S
from click import prompt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Extract the key points from the following text: {text}",
    input_variables=["text"]
)

loader = TextLoader("RAG_Components/Document_Loaders/sample.txt")

document = loader.load()

chain = prompt | model | parser

print(chain.invoke({"text": document[0].page_content}))