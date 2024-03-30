import os
from apikey import apikey
from sklearn.utils import resample
import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.colored_header import colored_header 
from streamlit_extras.row import row
import pandas as pd
import re
import altair as alt
from pathlib import Path
from PIL import Image
import helper 

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from lida import Manager, TextGenerationConfig , llm  
from lida.utils import plot_raster
from dotenv import load_dotenv, find_dotenv


from lida import Manager, TextGenerationConfig , llm as lidallm
from lida.utils import plot_raster
load_dotenv()

df = pd.read_csv("onlinefoods.csv")

class AgentResponseSchema(BaseModel):
    answer: str = Field(description="The direct answer derived from the DataFrame analysis.")
    isChart: bool = Field(description="A boolean indicating whether a chart is necessary to represent the answer visually.")
    chartPrompt: str = Field(description="If a chart is necessary, this field provides details on what the chart should represent, including the best possible title.")


parser = PydanticOutputParser(pydantic_object=AgentResponseSchema)

# Update the prompt to match the new query and desired format.
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(
            "answer the users question as best as possible.\n{format_instructions}\n{question}"
        )
    ],
    input_variables=["question"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=1000
)

user_query = "Give me a summary of the data in the DataFrame."

_input = prompt.format_prompt(question=user_query)

output = chat_model(_input.to_messages())
parsed = parser.parse(output.content)
print(output.content)
print(parsed)





# lida = Manager(text_gen = llm("openai")) 
# textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
# summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
# charts = lida.visualize(summary=summary,
#                         goal="create a line chart of age vs monthly income",
#                         textgen_config=textgen_config,
#                         library="matplotlib",
#                         return_error=True)  
# print(charts[0])