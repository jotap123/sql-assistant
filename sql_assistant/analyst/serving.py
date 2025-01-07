from sql_assistant.front_layer import AgentUI
from sql_assistant.analyst.chat import DataAnalyst


if __name__=='__main__':
    agent = DataAnalyst()
    ui = AgentUI(agent)
    ui.app()