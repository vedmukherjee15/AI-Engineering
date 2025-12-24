from heapq import merge
from pyexpat import model
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(model_name="gpt-5-nano")

model2 = model = ChatAnthropic(model='claude-3-haiku-20240307')

template1 = PromptTemplate(
    template = "Generate short and simple notes for the following topic. \n {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Create a quiz with 5 questions based on the following notes. \n {topic}",
    input_variables = ["topic"]
)

template3 = PromptTemplate(
    template = "Combine the following notes and quiz into a single output. \n Notes: {notes} \n Quiz: {quiz}",
    input_variables = ["notes", "quiz"]
)

parser = StrOutputParser()

Parallel_Chain = RunnableParallel({
   'notes' : template1 | model1 | parser ,
   'quiz' : template2 | model2 | parser 
})

merge_chain = template3 | model1| parser


chain = Parallel_Chain | merge_chain

result = chain.invoke({"topic":"The impact of climate change on global agriculture"})


print(result)

chain.get_graph().print_ascii()