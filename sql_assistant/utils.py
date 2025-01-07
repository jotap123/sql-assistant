from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Annotated, Optional, Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from sql_assistant.query import SQLQuery, QueryResult


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: SQLQuery
    result: Optional[QueryResult] = None


class AnalysisType(Enum):
    DESCRIPTIVE = "descriptive"
    TEMPORAL = "temporal"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    AGGREGATION = "aggregation"


@dataclass
class AnalysisResult:
    analysis_type: AnalysisType
    visualization_type: str
    description: str
    viz_path: Optional[Path] = None


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