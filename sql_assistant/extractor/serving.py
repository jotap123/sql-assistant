from sql_assistant.front_layer import AgentUI
from sql_assistant.extractor.chat import ExtractorAgent


if __name__=='__main__':
    agent = ExtractorAgent()
    ui = AgentUI(agent)
    ui.app()