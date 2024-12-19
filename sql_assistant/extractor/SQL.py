import pandas as pd

from enum import Enum
from typing import Optional
from dataclasses import dataclass


class QueryStatus(Enum):
    PENDING = "pending"
    NEEDS_REVIEW = "needs_review"
    NEEDS_CORRECTION = "needs_correction"
    READY = "ready"
    FAILED = "failed"
    COMPLETE = "complete"


@dataclass
class SQLQuery:
    text: str
    status: QueryStatus
    feedback: Optional[str] = None
    retry_count: int = 0
    
@dataclass
class QueryResult:
    success: bool
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    row_count: Optional[int] = None