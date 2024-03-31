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






# chat_model = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     max_tokens=1000
# )

# user_query = "Give me a summary of the data in the DataFrame."

# _input = prompt.format_prompt(question=user_query)

# output = chat_model(_input.to_messages())


# parsed = parser.parse(output.content)
# print(output.content)
# print(parsed)





# TEST 2: api response handler
# import re
# import json

# def escape_for_json_markdown(original_text):
#     # Escape backslashes first to avoid double escaping later characters
#     escaped_text = original_text.replace('\\', '\\\\')
    
#     # Escape double quotes
#     escaped_text = escaped_text.replace('"', '\\"')
    
#     # Replace newlines with \n to preserve them in the JSON string
#     escaped_text = escaped_text.replace('\n', '\\n')
    
#     # Escape backticks by doubling them (specific to your Markdown processor's requirements)
#     escaped_text = escaped_text.replace('`', '``')
    
#     return escaped_text

# response = '''{ "answer": "222", "chartPrompt": "" } { "answer": "222", "chartPrompt": "" }'''
# response = '''{ "answer": "222", "chartPrompt": "" } There are 222 males in the dataframe.'''
# # response = '''5```json
# # {
# #     "answer": "5",
# #     "chartPrompt": ""
# # }
# # ```'''
# response= '''{"answer": "To create an Axios request in JavaScript, you can use the Axios library which simplifies the process of making HTTP requests. First, you need to include Axios in your project either via a CDN or by installing it using npm. Then, you can use Axios to make GET, POST, PUT, DELETE, etc. requests to a server. Here is a basic example of making a GET request using Axios:\n\n```javascript\naxios.get('https://api.example.com/data')\n  .then(function (response) {\n    // handle success\n    console.log(response.data);\n  })\n  .catch(function (error) {\n    // handle error\n    console.log(error);\n  });\n```","chartPrompt": ""}'''

# # Regex pattern to match a JSON-like substring
# # pattern = r'\{[^\{]*?\}'
# pattern = r'''\{\s*"answer":\s*".*?",\s*"chartPrompt":\s*".*?"\s*\}'''
# # pattern = r'\{.*\}'

# # Search for the pattern in the response
# match = re.search(pattern, response, re.DOTALL)

# # Extract the matched JSON-like substring
# json_substring = match.group(0) if match else None
# print(json_substring)
# # json_substring = escape_for_json_markdown(json_substring)

# # Convert the JSON-like substring to a dictionary if it was found
# if json_substring:
#     parsed = parser.parse(json_substring.replace("```", "PLACEHOLDER_FOR_TRIPLE_BACKTICKS"))
#     parsed.answer = parsed.answer.replace("PLACEHOLDER_FOR_TRIPLE_BACKTICKS", "```")
#     print(parsed)
# else:
#     print("No JSON-like substring found.")









# lida = Manager(text_gen = llm("openai")) 
# textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
# summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
# charts = lida.visualize(summary=summary,
#                         goal="create a line chart of age vs monthly income",
#                         textgen_config=textgen_config,
#                         library="matplotlib",
#                         return_error=True)  
# print(charts[0])