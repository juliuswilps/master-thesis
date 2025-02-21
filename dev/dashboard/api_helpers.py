from openai import OpenAI
import os
from dotenv import load_dotenv

def setup():
    if not os.getenv("OPENAI_API_KEY"):
        load_dotenv()
    client = OpenAI()
    return client

def get_system_msg(graph):
    system_msg = f"""
    You are an expert data scientist and financial analyst tasked with analyzing and adjusting shape functions produced by a generalized additive model (GAM), which was trained to predict if a loan will be paid off (1) or default (0). Your job is to ensure the shape function aligns with financial domain knowledge.
    The shape function is provided as a dictionary where the keys are the x-values (input feature values), and the corresponding values are the y-values (predicted model outputs).
    You answer all questions to the best of your ability, relying on the graphs provided by the user, any other information you are given, and your knowledge about the real world.

    Your goal is to:
    1. Analyze the given shape function for any patterns or inconsistencies that contradict established domain knowledge.
    2. If you find any such patterns, adjust the y-values accordingly.
    3. The adjusted y-values should preserve the same length as the original dictionary, which has exactly {len(graph.keys())} elements, and their order must match the x-values exactly.

    The output should be an array of adjusted y-values that has exactly {len(graph.keys())} elements, ensuring that each adjusted y-value corresponds to the respective x-value and maintains the correct alignment with the original data.
    """

    return system_msg
