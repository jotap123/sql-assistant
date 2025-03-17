import os

from pathlib import Path
from langchain_core.messages import AIMessage

from sql_assistant.database import DatabaseConnection
from sql_assistant.chains import Chains
from sql_assistant.query import SQLQuery, QueryStatus
from sql_assistant.config import FILEPATH, chat, path_db
from sql_assistant.state import AgentState
from sql_assistant.utils import load_llm_chat


class SQLBaseAgent:
    def __init__(
        self,
        db_path: Path = path_db,
        max_retries: int = 2
    ):
        self.max_retries = max_retries
        self.llm_chat = load_llm_chat(chat)
        self.db = DatabaseConnection(db_path)
        self.chains = Chains()


    def _generate(self, state: AgentState) -> AgentState:
        request = state['messages'][-1].content
        state['user_input'] = request

        query_text = self.chains.generate.invoke({
            "schema": self.db.get_schema(),
            "request": request
        }).strip("```").strip("sql\n")

        state['query'] = SQLQuery(
            text=query_text,
            status=QueryStatus.NEEDS_REVIEW
        )
        state['messages'].append(AIMessage(content=f"Generated SQL Query: {query_text}"))
        return state


    def _review(self, state: AgentState) -> AgentState:
        feedback = self.chains.review.invoke({
            "query": state["query"].text,
            "schema": self.db.get_schema()
        })

        state['query'].feedback = feedback
        state['messages'].append(AIMessage(content=f"Review Feedback: {feedback}"))

        if "INCORRECT" in feedback.upper():
            state['query'].status = QueryStatus.NEEDS_CORRECTION
        if "INVALID" in feedback.upper():
            state['query'].status = QueryStatus.FAILED
        else:
            state['query'].status = QueryStatus.READY

        return state


    def _correct(self, state: AgentState) -> AgentState:
        if state['query'].retry_count >= self.max_retries:
            state['query'].status = QueryStatus.FAILED
            return state

        corrected_query = self.chains.correct.invoke({
            "query": state['query'].text,
            "feedback": state['query'].feedback,
            "schema": self.db.get_schema()
        })

        state['query'].text = corrected_query
        state['query'].status = QueryStatus.READY
        state['messages'].append(AIMessage(content=f"Corrected SQL Query: {corrected_query}"))
        return state


    def _extract(self, state: AgentState) -> AgentState:
        result = self.db.extract_query(state['query'].text)
        state['result'] = result

        try:
            os.remove(FILEPATH)
        except:
            pass

        if not state['result'].empty:
            print("SUCCESS")
            state['query'].status = QueryStatus.COMPLETE
            os.makedirs(os.path.dirname(FILEPATH), exist_ok=True)
            result.to_csv(FILEPATH, index=False)
            state['messages'].append(AIMessage(content=f"Execution successful"))
        else:
            print("FAIL")
            state['query'].retry_count += 1
            message = f"Error executing query: Check Langsmith"
            state['messages'].append(AIMessage(content=message))
            if state['query'].retry_count >= self.max_retries:
                state['query'].status = QueryStatus.FAILED
            else:
                state['query'].status = QueryStatus.NEEDS_REVIEW

        return state


    def _execute(self, state: AgentState) -> AgentState:
        try:
            result = self.db.execute_query(state['query'].text)
            state['result'] = result

            print("SUCCESS")
            state['query'].status = QueryStatus.COMPLETE
        except:
            print("FAIL")
            state['query'].retry_count += 1
            message = "Error executing query: Check Langsmith"
            state['messages'].append(AIMessage(content=message))

            if state['query'].retry_count >= self.max_retries:
                state['query'].status = QueryStatus.FAILED
            else:
                state['query'].status = QueryStatus.NEEDS_REVIEW

        return state