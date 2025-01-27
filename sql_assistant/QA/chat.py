from typing import Dict, Any
from langgraph.graph import END, StateGraph

from sql_assistant.query import QueryStatus
from sql_assistant.chains import Chains
from sql_assistant.state import AgentState
from sql_assistant.utils import load_llm_chat
from sql_assistant.config import chat
from sql_assistant.base import SQLBaseAgent


class SQLAgent(SQLBaseAgent):
    def __init__(self):
        super().__init__()
        self.llm = load_llm_chat(chat)
        self.chains = Chains(self.llm)
        self.graph = self._build_graph()

        # self.graph.get_graph().draw_mermaid_png(output_file_path="QAgraph.png")


    def _generate_response(self, state: AgentState) -> Dict[str, Any]:
        """Generate natural language response from SQL results"""
        output_message = self.chains.sql_output_chain.invoke({
            "messages": state["messages"],
            "input": state["user_input"],
            "sql_result": state["result"]
        })

        return state


    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("generate", self._generate)
        workflow.add_node("review", self._review)
        workflow.add_node("correct", self._correct)
        workflow.add_node("execute", self._execute)
        workflow.add_node("generate_response", self._generate_response)

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


    def run(self, query: str) -> str:
        """Process a natural language query and return response"""
        result_state = self.graph.invoke({"messages": query}, self.state_config)
        response = result_state['messages'][-1].content

        return response


# Example usage
if __name__ == "__main__":
    agent = SQLAgent()
    response = agent.run("How many items each customer has bought?")
    print(f"Response: {response}")
