import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing_extensions import TypedDict
from typing import List, Optional, Literal, Tuple, Annotated
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from sql_assistant.extractor.database import DatabaseConnection
from sql_assistant.extractor.chain import SQLChains
from sql_assistant.extractor.SQL import SQLQuery, QueryResult, QueryStatus
from sql_assistant.config import DOWNLOAD_ENDPOINT, chat, coder
from sql_assistant.utils import load_llm_chat
from sql_assistant.extractor.download_file import Config


@dataclass
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: SQLQuery
    result: Optional[QueryResult] = None


class SQLAgent:
    def __init__(
        self,
        db_path: Path,
        max_retries: int = 1
    ):
        self.llm_chat = load_llm_chat(chat)
        self.llm_coder = load_llm_chat(coder)
        self.db = DatabaseConnection(db_path)
        self.chains = SQLChains(self.llm_coder)
        self.max_retries = max_retries
        self.graph = self._build_graph()
        
        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


    def create_output_chain(self, llm: BaseChatModel) -> ChatPromptTemplate:
        output_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
             You shall assist the user in any topics related to extracting data from your database,
             if the response is a succes include the download link for the file."""),
            ("user", """Query results are ready with the following details:
            - Row count: {{row_count}}
            - Columns: {{columns}}

            You can download your results at: {DOWNLOAD_ENDPOINT}

            Please format a response that includes:
            1. A note that the link will be available for download
            2. The download link for the results

            Please format a response informing the user about the results and how to download them.""")
        ])
        return output_prompt | llm | StrOutputParser()


    def _route(self, state: AgentState) -> Literal["generate", "review", "correct", "execute", "end"]:
        match state.query.status:
            case QueryStatus.PENDING:
                return "generate"
            case QueryStatus.NEEDS_REVIEW:
                return "review"
            case QueryStatus.NEEDS_CORRECTION:
                return "correct"
            case QueryStatus.READY:
                return "execute"
            case QueryStatus.FAILED:
                return "end"
            case _:
                return "end"


    def _generate(self, state: AgentState) -> AgentState:
        query_text = self.chains.generate.invoke({
            "schema": self.db.get_schema(),
            "request": state.messages[-1].content
        }).strip("```").strip("sql\n")

        state.query = SQLQuery(
            text=query_text,
            status=QueryStatus.NEEDS_REVIEW
        )
        state.messages.append(AIMessage(content=f"Generated SQL Query: {query_text}"))
        return state


    def _review(self, state: AgentState) -> AgentState:
        feedback = self.chains.review.invoke({
            "query": state.query.text,
            "schema": self.db.get_schema()
        })

        state.query.feedback = feedback
        state.messages.append(AIMessage(content=f"Review Feedback: {feedback}"))

        if "INCORRECT" in feedback.upper():
            state.query.status = QueryStatus.NEEDS_CORRECTION
        else:
            state.query.status = QueryStatus.READY

        return state


    def _correct(self, state: AgentState) -> AgentState:
        if state.query.retry_count >= self.max_retries:
            state.query.status = QueryStatus.FAILED
            return state
            
        corrected_query = self.chains.correct.invoke({
            "query": state.query.text,
            "feedback": state.query.feedback,
            "schema": self.db.get_schema()
        })
        
        state.query.text = corrected_query
        state.query.status = QueryStatus.READY
        state.messages.append(AIMessage(content=f"Corrected SQL Query: {corrected_query}"))
        return state


    def _execute(self, state: AgentState) -> AgentState:
        result = self.db.execute_query(state.query.text)
        state.result = result

        if not state.result.empty:
            print("SUCCESS")
            state.query.status = QueryStatus.COMPLETE
            result.to_csv(Config.get_results_path(), index=False)
            
            # Format output message with download info
            output_message = self.format_output.invoke({
                "row_count": len(result),
                "columns": ", ".join(result.columns)
            })
            
            state.messages.append(AIMessage(content=output_message))
        else:
            print("FAIL")
            state.query.retry_count += 1
            message = f"Error executing query: Check Langsmith"
            state.messages.append(AIMessage(content=message))
            if state.query.retry_count >= self.max_retries:
                state.query.status = QueryStatus.FAILED
            else:
                state.query.status = QueryStatus.NEEDS_REVIEW
        
        return state
    

    def _format_output(self, state: AgentState) -> AgentState:
        """Format the final output message with download link."""
        if state.query.status == QueryStatus.COMPLETE and hasattr(state, 'result'):
            output_message = self.create_output_chain().invoke({
                "row_count": len(state.result),
                "columns": ", ".join(state.result.columns),
                "download_link": DOWNLOAD_ENDPOINT
            })
            state.messages.append(AIMessage(content=output_message))
        return state


    def _should_continue(self, state: AgentState) -> bool:
        return state.query.status != QueryStatus.COMPLETE and state.query.status != QueryStatus.FAILED


    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("generate", self._generate)
        workflow.add_node("review", self._review)
        workflow.add_node("correct", self._correct)
        workflow.add_node("execute", self._execute)
        workflow.add_node("format_output", self._format_output)

        workflow.add_edge("generate", "review")
        workflow.add_conditional_edges(
            "review",
            lambda x: "correct" if x.query.status == QueryStatus.NEEDS_CORRECTION else "execute",
            {"correct": "correct", "execute": "execute"}
        )
        workflow.add_edge("correct", "execute")

        workflow.add_conditional_edges(
            "execute",
            lambda x: "review" if x.query.status == QueryStatus.NEEDS_REVIEW else "format_output",
            {"review": "review", "format_output": "format_output"}
        )
        workflow.add_edge("format_output", END)
        workflow.set_entry_point("generate")

        return workflow.compile()


    def run(self, user_request: str) -> Tuple[pd.DataFrame, List[BaseMessage]]:
        """
        Execute a SQL query based on the user request and return messages.
        Results will be available via the download endpoint.
        """
        initial_state = AgentState(
            messages=[HumanMessage(content=user_request)],
            query=SQLQuery(text="", status=QueryStatus.PENDING)
        )
        
        final_state = self.graph.invoke(initial_state)
        return final_state.messages


if __name__ == "__main__":
    from sql_assistant.config import path_db

    agent = SQLAgent(db_path=path_db)
    df, result = agent.run("How many items each customer has bought?")
    print(result)