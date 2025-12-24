from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18,description="Age of the person")
    place: str = Field(description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Give me name, age and place of a fictious {nationality} person, \n {format_instructions} ",
    input_variables = ["nationality"],
    partial_variables= {'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"nationality":"Indian"})

print(result)