import joblib
from dotenv import load_dotenv
load_dotenv()
from functions import describe_graph

llm = "gpt-4o-mini"
ebm = joblib.load("ebm-heloc.pkl")

response = describe_graph(llm, ebm, 0)
print(type(response))
print(response)