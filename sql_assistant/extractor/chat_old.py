import sqlite3
import pandas as pd

from typing import Dict, Any, Union

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from sql_assistant.agent_log import AgentLog
from sql_assistant.config import chat, coder, path_db
from sql_assistant.prompts import extract_query_gen_template, review_prompt, correction_prompt
from sql_assistant.utils import load_llm_chat, State


class DataExtractionAgent(AgentLog):
    color = AgentLog.GREEN
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.chat_llm = load_llm_chat(chat)
        self.coder_llm = load_llm_chat(coder)
        self.ai = self._build_graph()
        self.config = {"configurable": {"thread_id": "1"}}
        self.max_retries = 2  # Add maximum retry attempts
        self.log("Initializing Agent...")


    def _get_database_schema(self) -> str:
        self.log("Connecting to database...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            schema = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                schema.append(f"Table: {table_name}")
                schema.extend([f"- {col[1]} ({col[2]})" for col in columns])
                schema.append("")  # Add empty line between tables

            return "\n".join(schema)


    def _generate_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log("Generating query...")
        messages = state['messages']
        schema = self._get_database_schema()

        if 'retry_count' not in state:
            state['retry_count'] = 0

        llm_chain = extract_query_gen_template | self.coder_llm
        generated_query = llm_chain.invoke({"schema": schema, "request": messages[-1].content})

        return {
            **state,
            "messages": messages + [AIMessage(content=f"Generated SQL Query: {generated_query}")]
        }
    

    def _review_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reviews the generated SQL query for correctness.

        Args:
            state (Dict[str, Any]): Current state of the conversation.

        Returns:
            Dict[str, Any]: Updated state with review feedback.
        """
        self.log("Query under revision")
        messages = state['messages']
        sql_query = messages[-1].content.split("Generated SQL Query: ")[-1].strip()

        review_chain = review_prompt | self.coder_llm | StrOutputParser()
        review_feedback = review_chain.invoke({
            "sql_query": sql_query,
            "schema": self._get_database_schema()
        })

        return {
            **state,
            "messages": messages + [AIMessage(content=f"Review Feedback: {review_feedback.strip()}")],
            "needs_correction": "incorrect" in review_feedback.lower()
        }


    def _should_retry(self, state: Dict[str, Any]) -> Union[str, bool]:
        """Determines if the query should be retried based on the review feedback."""
        if not state.get("needs_correction"):
            return "execute_sql"
        
        if state['retry_count'] >= self.max_retries:
            return "execute_sql"  # Give up and try to execute anyway
            
        return "correct_sql"


    def _correct_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Corrects the SQL query if there are issues.

        Args:
            state (Dict[str, Any]): Current state of the conversation.

        Returns:
            Dict[str, Any]: Updated state with the corrected SQL query.
        """
        self.log("Correcting query...")
        messages = state['messages']
        feedback = messages[-1].content.split("Review Feedback: ")[-1].strip()
        original_query = [msg for msg in messages if "Generated SQL Query:" in msg.content][-1].content
    
        correction_chain = correction_prompt | self.coder_llm | StrOutputParser()
    
        corrected_query = correction_chain.invoke({
            "original_query": original_query,
            "feedback": feedback,
            "schema": self._get_database_schema()
        })
        
        return {
            **state,
            "retry_count": state.get('retry_count', 0) + 1,
            "messages": messages + [AIMessage(content=f"Generated SQL Query: {corrected_query}")]
        }


    def _execute_query(self, query: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)


    def _execute_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the generated SQL query and appends the result to the state as a downloadable CSV link.

        Args:
            state (Dict[str, Any]): Current state of the conversation.

        Returns:
            Dict[str, Any]: Updated state with the query execution result.
        """
        messages = state['messages']
        sql_query = [msg for msg in messages if "Generated SQL Query:" in msg.content][-1].content
        sql_query = sql_query.split("Generated SQL Query: ")[-1].strip()

        try:
            result = self._execute_query(sql_query)
            summary = (f"Query executed successfully. Found {len(result)} rows.\n\n")
            self.log(f"Query executed successfully. Found {len(result)} rows.")
            result.to_csv('test.csv', index=False)
            
            return {
                **state,
                "messages": messages + [ToolMessage(content=summary)],
                "results": result  # Store the full DataFrame in state
            }

        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            self.error(f"Error executing query: {str(e)}")
            if state.get('retry_count', 0) < self.max_retries:
                return {
                    **state,
                    "needs_correction": True,
                    "messages": messages + [ToolMessage(content=error_message)]
                }
            else:
                return {
                    **state,
                    "messages": messages + [
                        ToolMessage(content=f"{error_message}\nMax retry attempts ({self.max_retries}) reached.")
                    ]
                }

    
    def extract_data(self, user_request: str) -> str:
        messages = self.ai.invoke(
            {
                "messages": [HumanMessage(content=user_request)],
                "retry_count": 0,
                "needs_correction": False
            },
            self.config
        )
        return messages


    def _build_graph(self) -> StateGraph:
        memory = MemorySaver()

        self.error(f"Compiling Graph reasoning...")
        graph = StateGraph(State)
        graph.add_node("generate_sql", self._generate_sql)
        graph.add_node("review_sql", self._review_sql)
        graph.add_node("correct_sql", self._correct_sql)
        graph.add_node("execute_sql", self._execute_sql)

        # Add edges with conditional routing
        graph.add_edge("generate_sql", "review_sql")
        graph.add_conditional_edges(
            "review_sql",
            self._should_retry,
            {"correct_sql": "correct_sql", "execute_sql": "execute_sql"}
        )
        graph.add_edge("correct_sql", "review_sql")
        
        graph.set_entry_point("generate_sql")

        return graph.compile(checkpointer=memory)


# Example usage
if __name__ == "__main__":
    agent = DataExtractionAgent(path_db)

    # Extract data based on user request
    user_request = "How many units each employee has sold?"
    result = agent.extract_data(user_request)
    print("\nExtracted Data:")
    print(result)
