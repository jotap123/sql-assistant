from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

#chat = "Qwen/Qwen2.5-3B-Instruct"
#coder = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
chat = "microsoft/Phi-3.5-mini-instruct"
#coder = "microsoft/Phi-3-mini-128k-instruct"
coder = "meta-llama/Llama-3.2-3B-Instruct"


def get_root_dir():
    cur_dir = Path(__file__).resolve().parent

    while cur_dir.name != 'sql-assistant':
        cur_dir = cur_dir.parent
    return str(cur_dir)

path_db = get_root_dir() + '/data/db/chinook.db'
FILEPATH = get_root_dir() + "/data/query-results/query_results.csv"
