from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class SQLChains:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.output_parser = StrOutputParser()
        self._init_chains()
    
    def _init_chains(self):
        # Generation Chain
        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL expert. Generate a SQL query based on the user's request."),
            ("user", """Database Schema:
            {schema}
            
            User Request: {request}
            
            Generate a SQL query to fulfill this request.""")
        ])
        self.generate = generation_prompt | self.llm | self.output_parser
        
        # Review Chain
        review_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL expert. Review the query for correctness."),
            ("user", """Review this SQL query:
            {query}
            
            Schema:
            {schema}
            
            Start with CORRECT or INCORRECT followed by detailed feedback.""")
        ])
        self.review = review_prompt | self.llm | self.output_parser
        
        # Correction Chain
        correction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL expert. Correct the query based on the feedback."),
            ("user", """Query: {query}
            Feedback: {feedback}
            Schema: {schema}
            
            Provide the corrected query only.""")
        ])
        self.correct = correction_prompt | self.llm | self.output_parser