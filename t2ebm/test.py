import joblib
from dotenv import load_dotenv
load_dotenv()
from functions import describe_graph

llm = "gpt-4o-mini"
ebm = joblib.load("ebm_heloc.pkl")

desc = describe_graph(llm, ebm, 0)

print(desc)

