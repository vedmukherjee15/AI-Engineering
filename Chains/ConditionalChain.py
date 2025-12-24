from click import prompt
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1 = ChatOpenAI(model_name="gpt-5-nano")

model2 = ChatAnthropic(model='claude-3-haiku-20240307')

class Feedback(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

parser = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into postive or negative {feedback} \n {format_instruction} ",
    input_variables = ["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classification_chain = prompt | model1 | parser2


prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
   (lambda x:x.sentiment == "positive", prompt2 | model2 | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model2 | parser),
    (RunnableLambda(lambda x: "The feedback is neutral, no response needed."))
     
)

chain = classification_chain | branch_chain

result = chain.invoke({"feedback":"The product quality is excellent and I am very satisfied with my purchase."})

print(result)