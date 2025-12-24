import re
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables=['topic'])


# prompt1 = template1.invoke({'topic':'the impact of climate change on global agriculture'})

# result = model.invoke(prompt1)

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# prompt2 = template2.invoke({'text':result.content})

# result1 = model.invoke(prompt2)

# print(result1.content)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'the impact of climate change on global agriculture'})

print(result)



