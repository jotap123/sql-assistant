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
            ("system", "You are a SQL expert. You know everything about SQL and its operations. Don't give explanations, return only the SQL query."),
            ("user", """Database Schema:
            {schema}
            
            User Request: {request}
            
            Generate only a SQL query to fulfill this request.""")
        ])
        self.generate = generation_prompt | self.llm | self.output_parser
        
        # Review Chain
        review_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL expert. You will review the query for correctness according to the user request."),
            ("user", """Review this SQL query:
            {query}
            
            Schema:
            {schema}
            
            Start with CORRECT or INCORRECT followed by a brief feedback.""")
        ])
        self.review = review_prompt | self.llm | self.output_parser
        
        # Correction Chain
        correction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL expert. The following query seems to be wrong. Make any corrections based on the feedback given. Return only the query to the user."),
            ("user", """Query: {query}
            Feedback: {feedback}
            Schema: {schema}

            Provide only the corrected query.""")
        ])
        self.correct = correction_prompt | self.llm | self.output_parser
        