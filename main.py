from dis import Instruction
import os
from apikey import apikey
from cohere import ChatMessage
from sklearn.utils import resample
import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.colored_header import colored_header 
from streamlit_extras.row import row
from streamlit_extras.dataframe_explorer import dataframe_explorer
import pandas as pd
import re
import altair as alt
from pathlib import Path
from PIL import Image
import helper 
import json

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
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

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

st.title('Dataset Exploration')

def create_agent_safely(llm, df):
    try:
        if df is not None:
            agent = create_pandas_dataframe_agent(llm,
                                                df,
                                                verbose=True,
                                                agent_type=AgentType.OPENAI_FUNCTIONS)
            return agent
        else:
            raise ValueError("DataFrame is not defined.")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        return None
    
def create_chat_prompt(chat_history, user_question):
    instructions = """
        For every single query, you must provide a direct answer in a dictionary format:
        {"answer": "", "isChart": "", "chartPrompt": ""}
        This dictionary should include keys for the answer text ('answer'), a boolean indicating if a chart is
        needed ('isChart'), and a chart prompt with the best possible title for the chart if a
        chart is needed ('chartPrompt').

        Examples:

        Query: How many rows are in the dataset?
        Your Response: {"answer": "There are 12 rows", "isChart": "False", "chartPrompt": ""}

        Context: The dataset contains information about the GDP growth in the United States.
        Query: Show the trend of GDP growth in the United States over the last decade.
        Your Response: {"answer": "Please refer to the chart below.", "isChart": "True", "chartPrompt": "Plot a line chart showing the GDP growth of the United States. Title: 'Decade Trend of GDP Growth in the United States'"}

        Context: The dataset does not contain information about the GDP growth in the United States.
        Query: Show the trend of GDP growth in the United States over the last decade.
        Your Response: {"answer": "Sorry, this dataset does not have enough information", "isChart": "False", "chartPrompt": ""}
        """

    chat_messages: List[ChatMessage] = [ChatMessage(role=chat_item["role"], message=chat_item["content"]) for chat_item in chat_history]
    print(chat_messages)
    
    formatted_chat = instructions + "\n\nChat history:\n"

    # Iterate over chat history items and format each
    for chat_item in chat_history:
        role = chat_item["role"]
        content = chat_item["content"]
        formatted_chat += f"{role.title()}: {content}\n"

    # Add the new user question
    formatted_chat += f"User question: {user_question}"

    return formatted_chat



def escape_special_chars_in_json(json_string):
    # Dictionary mapping special characters to their escaped versions
    chars_to_escape = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\b": "\\b",
        "\f": "\\f"
    }
    
    # Iterate through the dictionary and replace each character in the string
    for char, escaped_char in chars_to_escape.items():
        json_string = json_string.replace(char, escaped_char)
    
    return json_string

def parse_response(response_text):
    # Convert the response text (JSON string) to a Python dictionary
    response_dict = json.loads(escape_special_chars_in_json(response_text))

    # Extract values directly from the dictionary
    answer = response_dict.get("answer", "No answer provided.")
    is_chart = response_dict.get("isChart", False)
    chart_prompt = response_dict.get("chartPrompt", "")

    return {
        "answer": answer,
        "isChart": is_chart,
        "chartPrompt": chart_prompt
    }


col1, col2 = st.columns([3,4])

with col1:
    # Load data
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv'])

    df = None
    agent = None
    if uploaded_file is not None:
        # Read the uploaded CSV file with Pandas
        df = pd.read_csv(uploaded_file)
        
        # Display the DataFrame
        st.subheader("Current dataframe:", divider='blue')
        st.data_editor(df, height=545)  
        # st.dataframe(dataframe_explorer(df), use_container_width=True)
        
        agent = create_agent_safely(llm, df)

with col2:
    # To draw the border line
    container = st.container(border=True)
    with container.container():

        chat_grid = grid(1, 1,  vertical_align="bottom")
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

            # Chat input for new questions 
            prompt = chat_grid.chat_input("Message assistant...")

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
                        # apiOutput = agent.invoke(create_chat_prompt(st.session_state.messages, prompt)).get("output")
                        print(create_chat_prompt(st.session_state.messages, prompt))
                        # apiOutput = """{"answer": "Here are three possible analyses you can perform on this dataset:\n\n1.", "isChart": "False", "chartPrompt": ""}"""
                        # try: 
                        #     responseDictionary = parse_response(apiOutput)
                        #     # response = responseDictionary["answer"]
                        # except Exception as e:
                        #     print(f"Parsing error: {e}")

                        response = "apiOutput"


                        # response = "I am here to help you with your data exploration."

                        # lida = Manager(text_gen = lidallm("openai"))
                        # textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                        # summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
                        # charts = lida.visualize(summary=summary,
                        #                         goal=prompt,
                        #                         textgen_config=textgen_config,
                        #                         library="matplotlib",
                        #                         return_error=True)  
                        # image_base64 = charts[0].raster
                        # img = helper.base64_to_image(image_base64)
                        
                        img = None

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

                # Display the agent's response
                with st.chat_message("assistant", avatar="✨"):
                    st.markdown(response)
                    if img != None:
                        st.image(img, width=500)

                # Append the agent's response to the chat history
                st.session_state.messages.append({"role": "assistant", "content": response, "chart": img})








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