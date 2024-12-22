from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class AnalysisChains:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._init_chains()


    def _init_chains(self):
        """Create a chain for determining appropriate analysis type."""
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analysis expert. Based on the query results and their structure,
            determine the most appropriate type of analysis and visualization.
            Return in the format:
            ANALYSIS_TYPE: [descriptive|temporal|correlation|distribution|aggregation]
            VISUALIZATION: [line|bar|scatter|histogram|heatmap|box]
            DESCRIPTION: Brief description of why this analysis is appropriate"""),
            ("user", """Query results structure:
            Columns: {columns}
            Data sample: {sample}
            
            Determine the most appropriate analysis approach.""")
        ])
        self.analysis_reflection = analysis_prompt | self.llm | StrOutputParser()