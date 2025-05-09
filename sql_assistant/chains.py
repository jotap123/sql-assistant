from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from sql_assistant.config import chat
from sql_assistant.utils import load_llm_chat


class Chains:
    def __init__(self):
        self.llm = load_llm_chat(chat)
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
        self.generate = generation_prompt | self.llm | StrOutputParser()

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
        self.review = review_prompt | self.llm | StrOutputParser()
        
        # Correction Chain
        correction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a SQL expert. The following query seems to be wrong. Make any corrections based on the feedback given. Return only the query to the user."),
            ("user", """Query: {query}
            Feedback: {feedback}
            Schema: {schema}

            Provide only the corrected query.""")
        ])
        self.correct = correction_prompt | self.llm | StrOutputParser()

        file_output_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
             You shall assist the user in any topics related to extracting data from your database.
             You shall only engage in topics related to the database or to SQL.
             If the response is a success include the download link for the file."""),
            ("user", """Query results are ready with the following details:
            - Row count: {row_count}
            - Columns: {columns}

            You can download your results at: {endpoint}

            Please format a response informing the user the row count, columns and about
             a download button made available for downloading the data.""")
        ])

        self.file_output_chain = file_output_prompt | self.llm | StrOutputParser()

        # Analysis reflection chain
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the user question, determine the most appropriate type of analysis.
            User Question: {question}

            Provide your response in the following format:
            ANALYSIS_TYPE: [TEMPORAL|CORRELATION|DISTRIBUTION|COMPARISON|COMPOSITION]
            VISUALIZATION: [recommended visualization type]
            TARGET_COLUMNS: [columns to analyze]
            RATIONALE: [brief explanation of your choice]""")
        ])
        self.analysis_reflection = analysis_prompt | self.llm | StrOutputParser()

        # Natural language output chain
        sql_output_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that explains database query results in 
            natural language. Given the original question and the query results, provide a clear 
            and concise explanation.
            Unless the user specifies a specific number of examples they wish to obtain,
            always limit your query to at most 5 results.
            You can order the results by a relevant column to return the most interesting
            examples in the database.
            Never query for all the columns from a specific table,
            only ask for the relevant columns given the question."""),
            MessagesPlaceholder(variable_name="messages"),
            ("system", """Original question: {input}
            Query result: {sql_result}
            Please explain this result in natural language.""")
        ])
        self.sql_output_chain = sql_output_prompt | self.llm | StrOutputParser()
        