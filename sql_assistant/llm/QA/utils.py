from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")


def load_llm_chat(model):
        llm = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            temperature=0.1,
            max_new_tokens=1024,
            return_full_text=False,
        )
        chat = ChatHuggingFace(llm=llm)

        return chat


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }