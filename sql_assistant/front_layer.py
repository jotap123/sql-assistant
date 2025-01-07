import streamlit as st

from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

from sql_assistant.extractor.chat import ExtractorAgent
from sql_assistant.config import FILEPATH

class AgentUI:
    def __init__(self, llm_agent):
        self.agent = llm_agent


    def run_agent(self, user_query):
        return self.agent.run(user_query)


    def app(self):
        st.set_page_config(page_title="SQL Extractor Assistant", page_icon="🤖")
        st.title("All things SQL 🤖")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(
                    content="""Hello! I'm here to help you query or extract your files in an easy manner."""
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
                    with st.spinner("Thinking..."):
                        # Get the response from the agent
                        ai_response = self.run_agent(user_query)

                        # Display the response in the chat
                        st.write(ai_response)

                        # Wrap the response in an AIMessage object and save it
                        ai_message = AIMessage(content=ai_response)
                        st.session_state.chat_history.append(ai_message)

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
        
        with st.sidebar:
            st.header("About")
            st.write(
                """
                This AI assistant uses:
                - Chinook database
                    - https://www.kaggle.com/datasets/nancyalaswad90/chinook-sample-database
                - LLM to extract data
                - LLM to analyze data (text, charts)
                """
            )


if __name__=='__main__':
    llm_agent = ExtractorAgent()
    ui = AgentUI(llm_agent)
    ui.app()