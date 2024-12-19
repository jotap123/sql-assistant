import openai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import List, Dict, Any

from sql_assistant.utils import load_llm_chat

class DataAnalystAgent:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conversation_history = []

    def _execute_query(self, query: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def _get_database_schema(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema = ""
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                schema += f"Table: {table_name}\n"
                schema += "\n".join([f"- {col[1]} {col[2]}" for col in columns])
                schema += "\n\n"
        return schema

    def analyze_data(self, query: str) -> str:
        # Add the user's query to the conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Fetch data summary for all tables
        schema = self._get_database_schema()

        # Construct the prompt
        prompt = f"""
        You are a data analyst AI assistant. Analyze the following database schema and answer the user's query.

        Database Schema:
        {schema}

        User query: {query}

        Provide a detailed analysis based on the database schema and the user's query.
        """

        # Get the response from the LLM
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.conversation_history + [{"role": "user", "content": prompt}]
        )

        analysis = response.choices[0].message.content

        # Add the AI's response to the conversation history
        self.conversation_history.append({"role": "assistant", "content": analysis})

        return analysis

    def visualize_data(self, visualization_type: str, x_column: str, y_column: str, table_name: str = None) -> None:
        if table_name is None:
            # If table_name is not provided, try to infer it from the column names
            schema = self._get_database_schema()
            tables = [line.split(":")[1].strip() for line in schema.split("\n") if line.startswith("Table:")]
            for table in tables:
                if x_column in schema and y_column in schema:
                    table_name = table
                    break
            if table_name is None:
                raise ValueError("Could not infer table name. Please provide it explicitly.")

        data = self._execute_query(f"SELECT {x_column}, {y_column} FROM {table_name}")

        plt.figure(figsize=(10, 6))

        if visualization_type == "scatter":
            sns.scatterplot(data=data, x=x_column, y=y_column)
        elif visualization_type == "line":
            sns.lineplot(data=data, x=x_column, y=y_column)
        elif visualization_type == "bar":
            sns.barplot(data=data, x=x_column, y=y_column)
        else:
            raise ValueError("Unsupported visualization type")

        plt.title(f"{visualization_type.capitalize()} plot of {y_column} vs {x_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def get_recommendations(self) -> List[str]:
        schema = self._get_database_schema()

        prompt = f"""
        You are a data analyst AI assistant. Based on the following database schema,
        provide a list of 5 recommendations for analysis or insights that might be valuable.

        Database Schema:
        {schema}

        Provide 5 recommendations, each on a new line starting with a dash (-).
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        recommendations = response.choices[0].message.content.split("\n")
        return [rec.strip("- ") for rec in recommendations if rec.strip().startswith("-")]

    def extract_data(self, user_request: str) -> pd.DataFrame:
        schema = self._get_database_schema()

        prompt = f"""
        You are a data analyst AI assistant. Generate an SQL query to extract data based on the user's request.
        
        Database Schema:
        {schema}

        User request: {user_request}

        Generate only the SQL query without any explanations. The query should start with 'SELECT' and end with a semicolon.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        generated_query = response.choices[0].message.content.strip()

        # Execute the generated query and return the results
        try:
            result = self._execute_query(generated_query)
            print(f"Executed query: {generated_query}")
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Replace with your actual OpenAI API key
    api_key = "your_openai_api_key_here"
    
    # Path to your SQLite database
    db_path = "your_database.db"

    # Create an instance of the DataAnalystAgent
    agent = DataAnalystAgent(api_key, db_path)

    # Analyze the data
    analysis = agent.analyze_data("What are the main trends in the sales data?")
    print("Analysis:", analysis)

    # Visualize the data
    agent.visualize_data("scatter", "quantity", "price", "sales")

    # Get recommendations
    recommendations = agent.get_recommendations()
    print("Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Extract data based on user request
    user_request = "Show me the total sales for each product"
    extracted_data = agent.extract_data(user_request)
    print("\nExtracted Data:")
    print(extracted_data)