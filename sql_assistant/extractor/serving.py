import streamlit as st

from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

from sql_assistant.extractor.chat import SQLAgent
from sql_assistant.config import path_db, FILEPATH

def app():
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

        if user_query:
            with st.chat_message("AI"):
                resp = st.write(prepare_agent(user_query))

            if Path(FILEPATH).exists():
                with open(FILEPATH, "rb") as file:
                    btn = st.download_button(
                        label="Download file",
                        data=file,
                        file_name="results.csv",
                        mime="text/csv"
                    )

        else:
            st.write("Please enter a query")


if __name__=='__main__':
    app()