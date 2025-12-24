from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv  
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model_name="gpt-5-nano")

class ReportSummary(BaseModel):
    summary: str = Field(description="A concise summary of the report")
    sentiment : Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the report")

parser = PydanticOutputParser(pydantic_object=ReportSummary)


template1 = PromptTemplate(template="Give me a detailed report on the {topic}. \n {format_instructions}",
                           input_variables=["topic"],
                           partial_variables={"format_instructions": parser.get_format_instructions()})


template2 = PromptTemplate(template="Provide a brief summary and sentiment analysis of the following report: {report}\n {format_instructions}",
                            input_variables=["report"],
                            partial_variables={"format_instructions": parser.get_format_instructions()})

chain = template1 | model | parser |template2 | model | parser


result= chain.invoke({"topic":"Geopolitics in the Indian subcontinent"})


print(result)

chain.get_graph().print_ascii()