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
            ("system", """You are a SQL expert. You know everything about SQL and its operations.
             Don't give explanations, return only the SQL query.
             DO NOT generate a query if the request is invalid, empty or you don't understand it.
             If that is the case you should return the text 'invalid request'"""),
            ("user", """Database Schema:
            {schema}

            User Request: {request}

            If the request is valid generate a SQL query to fulfill this request.""")
        ])
        self.generate = generation_prompt | self.llm | self.output_parser

        # Review Chain
        review_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert.
             You will review the query for correctness according to the user request.
             DO NOT try to fix the request if a query isn't provided.
             If a query isn't provided mark it as INVALID.
             """),
            ("user", """Review this SQL query:
            {query}

            Schema:
            {schema}

            Start with CORRECT, INCORRECT or INVALID followed by a brief feedback.""")
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
        