from sql_assistant.front_layer import AgentUI
from sql_assistant.QA.chat import SQLAgent


if __name__=='__main__':
    agent = SQLAgent()
    ui = AgentUI(agent)
    ui.app()