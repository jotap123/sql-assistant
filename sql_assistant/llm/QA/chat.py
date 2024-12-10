import os

from typing import Literal
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.graph import END, StateGraph, START

from sql_assistant.llm.QA.utils import (
    State,
    SubmitFinalAnswer,
    create_tool_node_with_fallback,
    handle_tool_error,
    load_llm_chat,
)
from sql_assistant.llm.QA.constants import chat, coder, query_check_system, query_gen_system


class SQLAssistant:
    path_db = os.getcwd() + '/chinook.db'
    db = SQLDatabase.from_uri(f"sqlite:///{path_db}")
    chat_llm = load_llm_chat(chat)
    coder_llm = load_llm_chat(coder)


    @tool
    def db_query_tool(self, query: str) -> str:
        """
        Execute a SQL query against the database and get back the result.
        If the query is not correct, an error message will be returned.
        If an error is returned, rewrite the query, check the query, and try again.
        """
        result = self.db.run_no_throw(query)
        if not result:
            return "Error: Query failed. Please rewrite your query and try again."
        return result
    

    def build_tools(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.coder_llm)
        tools = toolkit.get_tools()

        self.list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        self.get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

        query_check_prompt = ChatPromptTemplate.from_messages(
            [("system", query_check_system), ("placeholder", "{messages}")]
        )
        self.query_check = query_check_prompt | self.coder_llm.bind_tools(
            [self.db_query_tool]
        )


    # Add a node for the first tool call
    def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_db_list_tables",
                            "args": {},
                            "id": "tool_abcd123",
                        }
                    ],
                )
            ]
        }


    def model_check_query(self, state: State) -> dict[str, list[AIMessage]]:
        """
        Use this tool to double-check if your query is correct before executing it.
        """
        return {"messages": [self.query_check.invoke({"messages": [state["messages"][-1]]})]}


    def query_gen_node(self, state: State):
        message = self.query_gen.invoke(state)

        # Sometimes, the LLM will hallucinate and call the wrong tool.
        # We need to catch this and return an error message.
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"""Error: The wrong tool was called: {tc['name']}.
                            Please fix your mistakes. Remember to only call SubmitFinalAnswer
                            to submit the final answer. Generated queries should be outputted
                            WITHOUT a tool call.""",
                            tool_call_id=tc["id"],
                        )
                    )
        else:
            tool_messages = []
        return {"messages": [message] + tool_messages}


    # Define a conditional edge to decide whether to continue or end the workflow
    def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
        messages = state["messages"]
        last_message = messages[-1]
        # If there is a tool call, then we finish
        if getattr(last_message, "tool_calls", None):
            return END
        if last_message.content.startswith("Error:"):
            return "query_gen"
        else:
            return "correct_query"

    
    def generate_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("first_tool_call", self.first_tool_call)

        # Add nodes for the first two tools
        workflow.add_node(
            "list_tables_tool", create_tool_node_with_fallback([self.list_tables_tool])
        )
        workflow.add_node(
            "get_schema_tool", create_tool_node_with_fallback([self.get_schema_tool])
        )

        # Add a node for a model to choose the relevant tables based on the question and
        # available tables
        model_get_schema = self.coder_llm.bind_tools(
            [self.get_schema_tool]
        )
        workflow.add_node(
            "model_get_schema",
            lambda state: {
                "messages": [model_get_schema.invoke(state["messages"])],
            },
        )

        query_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", query_gen_system), ("placeholder", "{messages}")]
        )
        query_gen = query_gen_prompt | self.chat_llm.bind_tools(
            [SubmitFinalAnswer]
        )

        workflow.add_edge(START, "first_tool_call")
        workflow.add_edge("first_tool_call", "list_tables_tool")
        workflow.add_edge("list_tables_tool", "model_get_schema")
        workflow.add_edge("model_get_schema", "get_schema_tool")
        workflow.add_edge("get_schema_tool", "query_gen")
        workflow.add_conditional_edges(
            "query_gen",
            self.should_continue,
        )
        workflow.add_edge("correct_query", "execute_query")
        workflow.add_edge("execute_query", "query_gen")

        # Compile the workflow into a runnable
        self.app = workflow.compile()
