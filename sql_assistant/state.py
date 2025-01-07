from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Annotated, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

from sql_assistant.query import SQLQuery, QueryResult


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: SQLQuery
    result: Optional[QueryResult] = None
    user_input: Optional[str] = None


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
