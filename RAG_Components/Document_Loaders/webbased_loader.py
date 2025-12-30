from pyexpat import model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from tomlkit import document

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")  

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Extract the main ideas from the following webpage content: {content}",
    input_variables=["content"]
)

loader = WebBaseLoader("https://introtodeeplearning.com/")

document = loader.load()

chain = prompt | model | parser

print(chain.invoke({"content": document[0].page_content}))