from json import load
from click import prompt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

loader = TextLoader("RAG_Components/Document_Loaders/sample.txt")

document = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)

texts = text_splitter.split_documents(document)

print(texts[0].page_content)

