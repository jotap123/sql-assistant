from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import List, Annotated, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

from sql_assistant.query import SQLQuery, QueryResult


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: SQLQuery
    result: Optional[QueryResult] = None
    user_input: Optional[str] = None


class AnalysisType(Enum):
    TEMPORAL = "temporal"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    AGGREGATION = "aggregation"


class AnalysisContext:
    analysis_type: str
    visualization_type: str
    user_question: str
    target_columns: List[str]
    time_period: Optional[str] = None
    comparison_groups: Optional[List[str]] = None
