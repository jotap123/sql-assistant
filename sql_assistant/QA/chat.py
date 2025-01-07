from typing import Dict, Any, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import END, StateGraph

from sql_assistant.config import chat, coder, path_db
from sql_assistant.query import QueryStatus
from sql_assistant.utils import (
    AgentState,
    load_llm_chat,
)


class SQLAgent:
    """Main SQL Agent class that handles database interactions and natural language processing"""

    def __init__(self, db_uri: str):
        # Initialize database and LLM
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = load_llm_chat(chat)
        
        # Set up SQL tool
        self.tool_executor = ToolExecutor([
            Tool(
                name="sql_db",
                description="Execute SQL queries on the database",
                func=self._execute_sql
            )
        ])

        # Initialize prompts
        self.sql_prompt = self._create_sql_prompt()
        self.response_prompt = self._create_response_prompt()
        
        # Build and compile the workflow
        self.chain = self._build_workflow()


    def _execute_sql(self, query: str) -> str:
        """Execute SQL query and return results"""
        try:
            result = self.db.run(query)
            return str(result)
        except Exception as e:
            return f"Error executing query: {str(e)}"


    def _create_sql_prompt(self) -> ChatPromptTemplate:
        """Create the SQL generation prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Your task is to convert natural language questions 
            about the database into SQL queries. The database contains tables with their schemas.
            Provide only the SQL query without any explanation."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{query}")
        ])


    def _create_response_prompt(self) -> ChatPromptTemplate:
        """Create the response generation prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that explains database query results in 
            natural language. Given the original question and the query results, provide a clear 
            and concise explanation."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", """Original question: {query}
            Query result: {sql_result}
            Please explain this result in natural language.""")
        ])


    def _generate_sql(self, state: AgentState) -> Dict[str, Any]:
        """Generate SQL query from natural language"""
        messages = self.sql_prompt.format_messages(
            messages=state["messages"],
            query=state["query"]
        )
        response = self.llm.invoke(messages)
        return {"sql_result": self.tool_executor.invoke({"input": response.content})["output"]}


    def _generate_response(self, state: AgentState) -> Dict[str, Any]:
        """Generate natural language response from SQL results"""
        messages = self.response_prompt.format_messages(
            messages=state["messages"],
            query=state["query"],
            sql_result=state["sql_result"]
        )
        response = self.llm.invoke(messages)
        return {"messages": [*state["messages"], AIMessage(content=response.content)]}


    def _build_workflow(self) -> StateGraph:
        """Build and compile the workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.add_edge("generate_sql", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()


    def query(self, query: str, messages: Optional[list] = None) -> str:
        """Process a natural language query and return response"""
        if messages is None:
            messages = []
        
        state = {
            "messages": messages,
            "query": query,
            "sql_result": None
        }
        
        result = self.chain.invoke(state)
        return result["messages"][-1].content


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize the agent
    agent = SQLAgent(db_uri="sqlite:///your_database.db")

    # Example queries
    queries = [
        "How many users are in the database?",
        "What's the average order value?",
    ]
    
    for query in queries:
        response = agent.query(query)
        print(f"\nQuestion: {query}")
        print(f"Response: {response}")
