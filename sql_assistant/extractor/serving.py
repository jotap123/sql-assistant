import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from sql_assistant.extractor.chat import SQLAgent
from sql_assistant.config import path_db


st.set_page_config(page_title="SQL Extractor Assistant", page_icon="🤖")
st.title("All things SQL 🤖")

def prepare_agent(user_query):
    agent = SQLAgent(db_path=path_db)
    return agent.run(user_query)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="""Hello! I'm here to help you query and extract your files in an easy manner."""
        ),
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


user_query = st.chat_input("Enter your query:")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        resp = st.write_stream(prepare_agent(user_query))