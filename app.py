import pandas as pd
import streamlit as st
import logging
from streamlit_extras.grid import grid
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.stylable_container import stylable_container
from dotenv import load_dotenv
from sympy import use
load_dotenv()

# Custom imports
import helper 
import agent as openai_agent
import chart as lida_chart    


def display_chart(chartPrompt, df):
    chart = None
    if chartPrompt:
        with st.spinner("Creating chart..."):
            for attempt in range(3):  # Allows up to 3 attempts
                try:
                    chart = lida_chart.generate_chart(chartPrompt, df)
                    break
                except Exception as e:
                    logging.error("Chart generation: %s", str(e))
                    if attempt == 2:
                        st.error("Sorry, could not generate the chart.")
    return chart

def display_message_and_response(message, agent):
    # Append the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)
    
    agentResponse = openai_agent.get_agent_response(st.session_state.messages, message, agent)
    responseAnswer = agentResponse.answer
    chart = display_chart(agentResponse.chartPrompt, filtered_df)

    # Display the agent's response
    with st.chat_message("assistant", avatar="✨"):
        st.markdown(responseAnswer)
        if chart != None:
            st.image(chart, width=500)

    # Append the agent's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": responseAnswer, "chart": chart})

def reset_session():
    st.session_state.messages = []
    st.session_state.suggestions = []

def clear_chat():
    st.session_state.messages = []
    st.rerun()

# App UI
st.set_page_config(
    page_title="Dataset Exploration",
    page_icon="🔍",
    initial_sidebar_state="collapsed",
    layout="wide")

st.title('Dataset Exploration')

col1, col2 = st.columns([3,4])

with col1:
    df = None  
    agent = None
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv'])

    if "dataset" not in st.session_state:
        st.session_state.dataset = None

    # User uploaded file
    if uploaded_file is not None:
        if st.session_state.dataset != None:
            reset_session()
        st.session_state.dataset = None
        df = pd.read_csv(uploaded_file)

    # Dropdown selection
    else:   
        option1 = "Online Food Order Dataset"
        option2 = "Heart Attack Prediction"
        option3 = "Student Study Performance"
        dataset_option = st.selectbox("or try sample datasets",
                                        options=[option1, option2, option3],
                                        index=None)
        if dataset_option == option1:
            if st.session_state.dataset != option1:
                reset_session()
                st.session_state.dataset = option1
            df = pd.read_csv("onlinefoods.csv")

        elif dataset_option == option2:
            if st.session_state.dataset != option2:
                reset_session()
                st.session_state.dataset = option2
            df = pd.read_csv("heart_attack_prediction_dataset.csv")

        elif dataset_option == option3:
            if st.session_state.dataset != option3:
                reset_session()
                st.session_state.dataset = option3
            df = pd.read_csv("study_performance.csv")
    
    # Create and display the DataFrame
    if df is not None:
        filtered_df = dataframe_explorer(df, case=False)
        agent = openai_agent.create_agent_safely(filtered_df)
        st.subheader("Current dataframe:", divider='blue')
        st.dataframe(filtered_df, height=400)
    else:
        st.session_state.messages = []
        st.session_state.suggestions = []

with col2:
    suggestions_toggle = st.toggle("Enable suggestions", False)
    # To draw the border line
    container = st.container(border=True)
    with container.container():

        chat_grid = grid(1, [1, 1], [7, 1],  vertical_align="center")
        with chat_grid.container(height = 550, border=False):

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

            if "suggestions" not in st.session_state:
                st.session_state.suggestions = []
        
            st.session_state.suggestionprompt = None

            # Suggestions
            if agent and suggestions_toggle:
                if st.session_state.suggestions == []:
                    st.session_state.suggestions = lida_chart.generate_suggestions(filtered_df)

                goals = st.session_state.suggestions
                with chat_grid.container():
                    with stylable_container(
                        key="suggestion1",
                        css_styles="""
                            button {
                                background-color: #202020;
                                border-radius: 5px;
                                height: 50px;
                            }
                            """,
                        ):
                        if st.button(goals[0].question, key="suggestion1", use_container_width=True):
                            st.session_state.suggestionprompt = goals[0].question

                with chat_grid.container():    
                    with stylable_container(
                        key="suggestion2",
                        css_styles="""
                            button {
                                background-color: #202020;
                                border-radius: 5px;
                                height: 50px;
                            }
                            """,
                        ):
                        if st.button(goals[1].question, key="suggestion2", use_container_width=True):
                            st.session_state.suggestionprompt = goals[1].question
            else: 
                chat_grid.empty()
                chat_grid.empty()

            # Chat input for new questions 
            prompt = chat_grid.chat_input("Message assistant...")
            clear_chat_button   = chat_grid.button("Clear chat", key="clear_chat", use_container_width=True)
            if clear_chat_button:
                clear_chat()

            if st.session_state.suggestionprompt != None:
                if agent is None:
                    st.info("Please upload a dataset to start the chat.")
                    st.stop()
                display_message_and_response(st.session_state.suggestionprompt, agent)
                st.session_state.suggestions = lida_chart.generate_suggestions(filtered_df)
                st.rerun()
                
            # Welcome message
            if prompt == None:
                if st.session_state.messages == []:
                    with st.container(height = 200, border=False):
                        st.empty()

                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="display: inline-block; width: 70px;">{helper.render_svg('assets/robot.svg')}</div>
                        <p>How can I assist with your data?</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.empty()

            else:
                if agent is None:
                    st.info("Please upload a dataset to start the chat.")
                    st.stop()
                display_message_and_response(prompt, agent)








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