from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

from sql_assistant.extractor.chat import SQLAgent
from sql_assistant.config import DOWNLOAD_ENDPOINT


class Config:
    RESULTS_DIR = Path("query-results")
    
    @classmethod
    def ensure_results_dir(cls):
        cls.RESULTS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_results_path(cls):
        return cls.RESULTS_DIR / "query_results.csv"


class QueryRequest(BaseModel):
    query: str


app = FastAPI()

@app.post("/query")
async def execute_query(request: QueryRequest, agent: SQLAgent = Depends()):
    messages = agent.run(request.query)
    return {"messages": [m.dict() for m in messages]}

@app.get(DOWNLOAD_ENDPOINT)
async def download_query_results():
    file_path = Config.get_results_path()
    if not file_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="No query results available. Please execute a query first."
        )
    return FileResponse(
        path=file_path, 
        filename="query_results.csv", 
        media_type="text/csv"
    )

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    Config.ensure_results_dir()