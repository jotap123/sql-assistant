import pandas as pd
import plotly.express as px

from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from sql_assistant.chains import Chains
from sql_assistant.query import SQLQuery, QueryStatus
from sql_assistant.state import AgentState, AnalysisType
from sql_assistant.base import SQLBaseAgent


class DataAnalyst(SQLBaseAgent):
    def __init__(self,):
        super().__init__()
        self.chains = Chains()
        self.graph = self._build_graph()

        # self.graph.get_graph().draw_mermaid_png(output_file_path="DA_graph.png")


    def _create_visualization(
        self,
        df: pd.DataFrame,
        analysis_type: AnalysisType,
        viz_type: str
    ) -> Dict[str, Any]:
        """Create visualization using Plotly."""
        if analysis_type == AnalysisType.TEMPORAL:
            fig = px.line(df, x=df.columns[0], y=df.columns[1:])
            
        elif analysis_type == AnalysisType.CORRELATION:
            if viz_type == 'heatmap':
                corr = df.corr()
                fig = px.imshow(corr, 
                              labels=dict(color="Correlation"),
                              x=corr.columns,
                              y=corr.columns)
            else:
                fig = px.scatter_matrix(df)
                
        elif analysis_type == AnalysisType.DISTRIBUTION:
            if viz_type == 'histogram':
                fig = px.histogram(df, x=df.columns[0])
            else:
                fig = px.box(df, y=df.columns[0])
                
        elif analysis_type == AnalysisType.AGGREGATION:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
            
        else:  # DESCRIPTIVE
            if df.select_dtypes(include=['number']).shape[1] > 0:
                numeric_col = df.select_dtypes(include=['number']).columns[0]
                fig = px.box(df, y=numeric_col)
            else:
                fig = px.bar(df.iloc[:, 0].value_counts())

        # Update layout for better presentation
        fig.update_layout(
            template='plotly_white',
            title={
                'text': f"{analysis_type.value.title()} Analysis",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            margin=dict(t=100, l=50, r=50, b=50)
        )

        return fig


    def _analyze_data(self, df: pd.DataFrame, state: AgentState) -> Dict[str, Any]:
        """Determine and perform appropriate analysis on the data."""
        analysis_plan = self.chains.analysis_reflection.invoke(
            {"input": state['user_input']}
        )

        # Parse recommendation
        plan_parts = {}
        for line in analysis_plan.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                plan_parts[key.strip()] = value.strip()
                
        analysis_type = AnalysisType(plan_parts.get('ANALYSIS_TYPE', 'descriptive').lower())
        viz_type = plan_parts.get('VISUALIZATION', 'bar').lower()
        description = plan_parts.get('DESCRIPTION', '')
        
        # Create visualization data
        fig = self._create_visualization(df, analysis_type, viz_type)
        
        return fig


    def _analyze(self, state: AgentState) -> AgentState:
        """Perform analysis on the query results."""
        if state['result'] is not None and not state['result'].empty:
            analysis_result = self._analyze_data(state['result'])
            state['analysis'] = analysis_result
            state['messages'].append(AIMessage(content=f"Analysis complete: {analysis_result.description}"))
        else:
            state['messages'].append(AIMessage(content="No data available for analysis."))
        return state


    def _format_analysis(self, state: AgentState) -> AgentState:
        """Format the analysis results with embedded visualization."""
        if 'analysis' in state and state['analysis'] is not None:
            analysis = state['analysis']
            
            # Convert the Plotly figure to HTML
            html_content = f"""
            <html>
                <head>
                    <title>Analysis Results</title>
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.24.3/plotly.min.js"></script>
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background-color: #f5f5f5;
                        }}
                        .container {{
                            max-width: 1200px;
                            margin: 0 auto;
                            background-color: white;
                            padding: 20px;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }}
                        .header {{
                            margin-bottom: 20px;
                        }}
                        .plot {{
                            width: 100%;
                            height: 600px;
                            margin: 20px 0;
                        }}
                        .metadata {{
                            margin-top: 20px;
                            padding: 15px;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>Analysis Results</h1>
                            <p><strong>Analysis Type:</strong> {analysis.analysis_type.value}</p>
                            <p><strong>Description:</strong> {analysis.description}</p>
                        </div>
                        <div id="plot" class="plot"></div>
                        <div class="metadata">
                            <h3>Data Summary</h3>
                            <p>Rows: {state['result'].shape[0]}</p>
                            <p>Columns: {state['result'].shape[1]}</p>
                            <p>Analyzed columns: {', '.join(state['result'].columns)}</p>
                        </div>
                    </div>
                    <script>
                        {analysis.fig.to_json()}
                    </script>
                </body>
            </html>
            """
            
            state['messages'].append(AIMessage(content=html_content))
        else:
            state['messages'].append(AIMessage(content="Analysis could not be completed."))
        return state


    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("generate", self._generate)
        workflow.add_node("review", self._review)
        workflow.add_node("correct", self._correct)
        workflow.add_node("execute", self._execute)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("format_analysis", self._format_analysis)

        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "review")
        workflow.add_conditional_edges(
            "review",
            lambda x: x['query'].status,
            {
                QueryStatus.NEEDS_CORRECTION: "correct",
                QueryStatus.READY: "execute",
                QueryStatus.FAILED: END
            }
        )
        workflow.add_edge("correct", "execute")

        workflow.add_conditional_edges(
            "execute",
            lambda x: x['query'].status,
            {QueryStatus.NEEDS_REVIEW: "review", QueryStatus.ANALYZE: "analyze"}
        )
        workflow.add_edge("analyze", "format_analysis")
        workflow.add_edge("format_analysis", END)

        return workflow.compile()


    def run(self, user_request: str) -> List[BaseMessage]:
        """
        Execute a SQL query based on the user request and return messages.
        Results will be available via the download endpoint.
        """
        initial_state = AgentState(
            messages=[HumanMessage(content=user_request)],
            query=SQLQuery(text="", status=QueryStatus.PENDING)
        )

        final_state = self.graph.invoke(initial_state)
        return final_state['messages'][-1].content


if __name__ == "__main__":
    agent = DataAnalyst()
    result = agent.run("How many items each customer has bought?")
    print(result)