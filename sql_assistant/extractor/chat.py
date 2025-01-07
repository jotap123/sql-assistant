from typing import List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END

from sql_assistant.chains import Chains
from sql_assistant.query import SQLQuery, QueryStatus
from sql_assistant.config import FILEPATH, chat, coder
from sql_assistant.state import AgentState
from sql_assistant.base import SQLBaseAgent
from sql_assistant.utils import load_llm_chat


class ExtractorAgent(SQLBaseAgent):
    def __init__(self):
        super().__init__()
        self.llm_chat = load_llm_chat(chat)
        self.llm_coder = load_llm_chat(coder)
        self.chains = Chains(self.llm_coder)
        self.graph = self._build_graph()

        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


    def _format_output(self, state: AgentState) -> AgentState:
        """Format the final output message with download link."""
        if state['query'].status == QueryStatus.COMPLETE and state['result'] is not None:
            output_message = self.chains.file_output_chain.invoke({
                "row_count": len(state['result']),
                "columns": ", ".join(state['result'].columns),
                "endpoint": FILEPATH
            })
            state['messages'].append(AIMessage(content=output_message))

        elif state['query'].status == QueryStatus.FAILED:
            state['messages'].append(AIMessage(content="Query execution failed. Please check if your query makes sense or try to reformulate it."))

        return state


    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("generate", self._generate)
        workflow.add_node("review", self._review)
        workflow.add_node("correct", self._correct)
        workflow.add_node("execute", self._extract)
        workflow.add_node("format_output", self._format_output)

        workflow.add_edge("generate", "review")
        workflow.add_conditional_edges(
            "review",
            lambda x: x['query'].status,
            {
                QueryStatus.NEEDS_CORRECTION: "correct",
                QueryStatus.READY: "execute",
                QueryStatus.FAILED: END
            }
        )
        workflow.add_edge("correct", "execute")

        workflow.add_conditional_edges(
            "execute",
            lambda x: x['query'].status,
            {QueryStatus.NEEDS_REVIEW: "review", QueryStatus.READY: "format_output"}
        )
        workflow.add_edge("format_output", END)
        workflow.set_entry_point("generate")

        return workflow.compile()


    def run(self, user_request: str) -> List[BaseMessage]:
        """
        Execute a SQL query based on the user request and return messages.
        Results will be available via the download endpoint.
        """
        initial_state = AgentState(
            messages=[HumanMessage(content=user_request)],
            query=SQLQuery(text="", status=QueryStatus.PENDING)
        )

        final_state = self.graph.invoke(initial_state)
        return final_state['messages'][-1].content


if __name__ == "__main__":
    agent = ExtractorAgent()
    result = agent.run("How many items each customer has bought?")
    print(result)