import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END

from sql_assistant.extractor.database import DatabaseConnection
from sql_assistant.extractor.chain import SQLChains
from sql_assistant.extractor.SQL import SQLQuery, QueryResult, QueryStatus


@dataclass
class AgentState:
    messages: List[BaseMessage]
    query: SQLQuery
    result: Optional[QueryResult] = None


class SQLAgent:
    def __init__(
        self,
        db_path: Path,
        llm: BaseChatModel,
        max_retries: int = 1
    ):
        self.db = DatabaseConnection(db_path)
        self.chains = SQLChains(llm)
        self.max_retries = max_retries
        self.graph = self._build_graph()
    

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
        })
        
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
        state.query.retry_count += 1
        state.messages.append(AIMessage(content=f"Corrected SQL Query: {corrected_query}"))
        return state


    def _execute(self, state: AgentState) -> AgentState:
        result = self.db.execute_query(state.query.text)
        state.result = result

        print(state.query)
        print(result.success)
        
        if result.success and result.data is not None:
            print("SUCCESS")
            state.query.status = QueryStatus.COMPLETE
            state.df = result.data  # Store the DataFrame in state

            message = (
                f"Query executed successfully. Found {len(result.data)} rows.\n\n"
            )
        else:
            print("FAIL")
            message = f"Error executing query: {result.error}"
            if state.query.retry_count >= self.max_retries:
                state.query.status = QueryStatus.FAILED
            else:
                state.query.status = QueryStatus.NEEDS_REVIEW
            
        state.messages.append(AIMessage(content=message))
        return state


    def _should_continue(self, state: AgentState) -> bool:
        return state.query.status != QueryStatus.COMPLETE and state.query.status != QueryStatus.FAILED


    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("generate", self._generate)
        workflow.add_node("review", self._review)
        workflow.add_node("correct", self._correct)
        workflow.add_node("execute", self._execute)

        workflow.add_edge("generate", "review")
        workflow.add_conditional_edges(
            "review",
            lambda x: "correct" if x.query.status == QueryStatus.NEEDS_CORRECTION else "execute",
            {"correct": "correct", "execute": "execute"}
        )
        workflow.add_edge("correct", "execute")

        workflow.add_conditional_edges(
            "execute",
            lambda x: "review" if x.query.status == QueryStatus.NEEDS_REVIEW else "END",
            {"review": "review", "END": END}
        )
        workflow.set_entry_point("generate")

        return workflow.compile()


    def run(self, user_request: str) -> Tuple[pd.DataFrame, List[BaseMessage]]:
        initial_state = AgentState(
            messages=[HumanMessage(content=user_request)],
            query=SQLQuery(text="", status=QueryStatus.PENDING)
        )
        final_state = self.graph.invoke(initial_state)

        if hasattr(final_state, 'df'):
            final_state.df.to_csv('test.csv', index=False)
            return final_state.df, final_state.messages

        else:
            return pd.DataFrame(), final_state.messages


if __name__ == "__main__":
    from sql_assistant.config import path_db, chat
    from sql_assistant.utils import load_llm_chat

    llm = load_llm_chat(chat)
    agent = SQLAgent(
        db_path=path_db,
        llm=llm,
    )
    df, result = agent.run("How many items each customer has bought?")
    print(result)