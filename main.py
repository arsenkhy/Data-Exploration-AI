from dis import Instruction
import os
from apikey import apikey
from cohere import ChatMessage
from openai import chat
from sklearn.utils import resample
import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.colored_header import colored_header 
from streamlit_extras.row import row
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import re
import altair as alt
from pathlib import Path
from PIL import Image
from sympy import use
import helper 
import logging
import json
import time
import numpy as np

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import ChatMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

from lida import Manager, TextGenerationConfig , llm as lidallm
from lida.utils import plot_raster
from dotenv import load_dotenv, find_dotenv
load_dotenv()

st.set_page_config(
    page_title="Dataset Exploration",
    page_icon="🔍",
    initial_sidebar_state="collapsed",
    layout="wide")

st.title('Dataset Exploration')

def create_agent_safely(df):
    try:
        if df is not None:
            llm = ChatOpenAI(temperature=0, model="gpt-4")
            agent = create_pandas_dataframe_agent(llm,
                                                df,
                                                verbose=True,
                                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                                handle_parsing_errors=True)
            return agent
        else:
            raise ValueError("DataFrame is not defined.")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        return None
    



class AgentResponseSchema(BaseModel):
    answer: str = Field(description="The direct answer derived from your the DataFrame analysis. Must be in markdown format")
    chartPrompt: str = Field(description="If asked to create chart or visualize data, this field provides a prompt for another agent to create what chart it should create in detail. This prompt must include the title and type of chart. If no chart is necessary, leave this field empty.")

parser = PydanticOutputParser(pydantic_object=AgentResponseSchema)

def create_chat_prompt(chat_history, user_question):
    chat_messages: List[ChatMessage] = [ChatMessage(role=chat_item["role"], content=chat_item["content"]) for chat_item in chat_history]

    prompt = ChatPromptTemplate(
        messages= [
            HumanMessagePromptTemplate.from_template(
                """answer this user question as best as possible.
                Always answer based on the full dataframe, not only head rows
                You must return answer using these format instructions\n{format_instructions} Question:\n{question}"""
            )
        ],
        input_variables=["question"],
        partial_variables={
            "format_instructions": """The output must be formatted as a JSON
                                        instance that conforms to the JSON schema below.
                                        As an example, the schema {"answer": "The dataset has x number of rows",
                                        "chartPrompt": "Descriptive chart creation prompt for another agent"} \nthe json 
                                        is a well-formatted instance of the schema. The object {"answer": {"foo": ["bar", "baz"]}} is not well-formatted.""",
        },
    )
    print(prompt.format_prompt(question=user_question).to_messages())

    input_messages = prompt.format_prompt(question=user_question).to_messages()

    return {"input": input_messages}

def parse_response(response_text) -> AgentResponseSchema:
    # Sometimes the JSON-like substring is not valid JSON, so need to extract it manually
    pattern = r'''\{\s*"answer":\s*".*?",\s*"chartPrompt":\s*".*?"\s*\}'''
    match = re.search(pattern, response_text)
    json_substring = match.group(0) if match else None

    if json_substring:
        try: 
            # Workaround for triple backticks in the response
            agentResponse = parser.parse(json_substring.replace("```", "PLACEHOLDER_FOR_TRIPLE_BACKTICKS"))
            agentResponse.answer = agentResponse.answer.replace("PLACEHOLDER_FOR_TRIPLE_BACKTICKS", "```")
            return agentResponse
        except Exception as e:
            logging.error("Parsing response error: %s", str(e))
            error_message = "Sorry, cannot process the given prompt"
            return AgentResponseSchema(answer=error_message, isChart=False, chartPrompt="")
    else:  
        return AgentResponseSchema(answer=response_text, isChart=False, chartPrompt="")
    

def generate_chart(prompt, df) -> Image:
    lida = Manager(text_gen = lidallm("openai"))
    textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
    summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
    charts = lida.visualize(summary=summary,
                            goal=prompt,
                            textgen_config=textgen_config,
                            library="matplotlib",
                            return_error=True)  
    image_base64 = charts[0].raster
    chart = helper.base64_to_image(image_base64)
    return chart


col1, col2 = st.columns([3,4])

with col1:
    # Load data
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv'])

    df = None  
    agent = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display the DataFrame
        st.subheader("Current dataframe:", divider='blue')
        filtered_df = dataframe_explorer(df, case=False)
        st.data_editor(filtered_df, height=545)  
        agent = create_agent_safely(filtered_df)

with col2:
    suggestions_toggle = st.toggle("Enable suggestions", False)
    # To draw the border line
    container = st.container(border=True)
    with container.container():

        chat_grid = grid(1, [1, 1], [7, 1],  vertical_align="center")
        with chat_grid.container(height = 700, border=False):

            # Messages
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display previous messages
            for message in st.session_state.messages:
                if message["role"] == "assistant":
                    with st.chat_message(message["role"], avatar="✨"):
                        st.markdown(message["content"])
                        if message["chart"] != None:
                            st.image(message["chart"], width=500)
                else:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"]) 

            # Suggestions
            if agent and suggestions_toggle:
                with chat_grid.container():
                    with stylable_container(
                    key="clear_chdaat",
                    css_styles="""
                        button {
                            background-color: black;
                            color: white;
                            border-radius: 20px;
                        }
                        """,
                    ):
                        st.button("Clear chasftlksjhdlkfghsklfjghlksjhdglkjchasfftlksjhdlkfghsklfjghlksjhdglkj", key="clear_chdaat", use_container_width=True)
                with chat_grid.container():    
                    with stylable_container(
                    key="clear_chadt",
                    css_styles="""
                        button {
                            background-color: black;
                            color: white;
                            border-radius: 20px;
                        }
                        """,
                    ):
                        st.button("Clear chatschasdstlksjhdlkfghsklfjghlksjhdglkj", key="clear_chadt", use_container_width=True)
            else: 
                chat_grid.empty()
                chat_grid.empty()


            # Chat input for new questions 
            prompt = chat_grid.chat_input("Message assistant...")
            chat    = chat_grid.button("Clear chat", key="clear_chat")


            # Display welcome message
            if prompt == None:
                with st.container(height = 300, border=False):
                    st.empty()

                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="display: inline-block; width: 70px;">{helper.render_svg('assets/robot.svg')}</div>
                    <p>How can I assist with your data?</p>
                </div>
                """, unsafe_allow_html=True)

            # Display chat messages
            else:
                if agent is None:
                    st.info("Please upload a dataset to start the chat.")
                    st.stop()
                # Append the user's message to the chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate a response from the agent
                with st.spinner("Generating response..."):
                    try:
                        apiOutput = agent.invoke(create_chat_prompt(st.session_state.messages, prompt)).get("output")
                        agentResponse = parse_response(apiOutput)
                        response = agentResponse.answer
                    except Exception as e:
                        st.error(f"An error from agent: {e}")

                # Generate a chart if necessary        
                chart = None
                if agentResponse.chartPrompt:
                    with st.spinner("Creating chart..."):
                        for attempt in range(3):  # Allows up to 3 attempts
                            try:
                                chart = generate_chart(agentResponse.chartPrompt, filtered_df)
                                break
                            except Exception as e:
                                logging.error("Chart generation: %s", str(e))
                                if attempt == 2:
                                    st.error("Sorry, could not generate the chart.")

                # Display the agent's response
                with st.chat_message("assistant", avatar="✨"):
                    st.markdown(response)
                    if chart != None:
                        st.image(chart, width=500)

                # Append the agent's response to the chat history
                st.session_state.messages.append({"role": "assistant", "content": response, "chart": chart})








# Sidebar
with st.sidebar:
    st.divider()

    # Github link
    col1, col2 = st.columns([1, 7], gap='small') 

    with col1:
        st.image('assets/github.svg', caption="", width=30)

    with col2:
        st.markdown(
            """
            <style>
                .custom-link {
                    color: white !important;
                    font-family: 'Courier New';
                    text-decoration: none;
                    font-size: 18px;
                }
                .custom-link:hover {
                    color: #add8e6 !important;
                }
            </style>
            <a href="https://github.com/arsenkhy" class="custom-link" >arsenkhy</a>
            """,
            unsafe_allow_html=True
        )