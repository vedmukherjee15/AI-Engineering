from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

load_dotenv()

parser = StrOutputParser()

model = ChatOpenAI(model="gpt-5-nano")

prompt1 = PromptTemplate(
    template="Generate a short story about {topic}.",
    input_variables=["topic"]
    )


prompt2 = PromptTemplate(
    template="Summarize the following text {text}.",
    input_variables=["text"]
)


chain1 = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = chain1 | branch_chain

print(final_chain.invoke({"topic": "a futuristic city"}))
