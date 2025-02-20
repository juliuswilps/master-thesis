import joblib
from dotenv import load_dotenv
load_dotenv()
from functions import adjust_graph

llm = "gpt-4o-mini"
ebm = joblib.load("ebm-heloc.pkl")

response = adjust_graph(llm, ebm, 0)
print(response)