import streamlit as st
import re
import logging
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import ChatMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
load_dotenv()


def create_agent_safely(df):
    try:
        if df is not None:
            llm = ChatOpenAI(temperature=0, model="gpt-4")
            agent = create_pandas_dataframe_agent(llm,
                                                df,
                                                verbose=True,
                                                agent_type=AgentType.OPENAI_FUNCTIONS,
                                                )
            return agent
        else:
            raise ValueError("DataFrame is not defined.")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
        return None

def get_agent_response(messages, prompt, agent):
    with st.spinner("Generating response..."):
        try:
            api_output = agent.invoke(create_chat_prompt(messages, prompt)).get("output")
            parsed_response = parse_response(api_output)
        except Exception as e:
            st.error(f"An error from agent: {e}")
    return parsed_response

class AgentResponseSchema(BaseModel):
    answer: str = Field(description="A string. The direct answer derived from your the DataFrame analysis. Must be in markdown format")
    chartPrompt: str = Field(description="A string. If asked to create chart or visualize data, this field provides a prompt for another agent to create what chart it should create in detail. This prompt must include the title and type of chart. If no chart is necessary, leave this field empty.")

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
                                        "chartPrompt": "Descriptive chart creation prompt for another agent with title"} \nthe json 
                                        is a well-formatted instance of the schema.""",
        },
    )

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