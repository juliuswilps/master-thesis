from openai import OpenAI
import os
from dotenv import load_dotenv
import ast

import data

def setup():
    if not os.getenv("OPENAI_API_KEY"):
        load_dotenv()
    client = OpenAI()
    return client

def get_system_msg(feature_data: dict, domain: str, prediction_target: str):
    return  f"""
You are a data scientist specializing in {domain}. 
Your goal is to integrate domain knowledge into interpretable machine learning models, ensuring that shape functions in Generalized Additive Models (GAMs) align with expert expectations.
The GAM is trained to {prediction_target}.
Given a shape function graph, you will identify patterns that contradict well-established domain knowledge and generate an adjusted shape function based on your findings.

You must:
1. Provide only Python code in your responses. Any explanation or additional text should be omitted.
2. Ensure that any arrays or lists you return have **exactly** {len(feature_data["x_vals"])} elements. Do not return extra or fewer elements.
3. For any graph adjustments, ensure that the **x-values** remain identical to the original ones, and **only adjust the y-values** based on domain knowledge. The x-values should not change, but the y-values should be modified to reflect the identified contradictions.
"""

def get_analyze_graph_prompt(feature_data: dict, domain: str, target: str):
    return f"""Analyze the following shape function for contradictions with domain knowledge.

Domain: {domain}
Prediction Target: {target}

Feature: {feature_data["feature_name"]}
Data Type: {feature_data["feature_type"]}
Description: {feature_data.get("feature_description", "No description provided.")}

X-values: {feature_data["x_vals"]}
Y-values: {feature_data["y_vals"][feature_data["current_iteration"]]}

Identify patterns in the shape function that contradict well-established domain knowledge in this field. 
Only highlight contradictions—do not suggest corrections yet.
Clearly describe these contradictions so they can be addressed in the next step.
"""

def get_adjust_graph_prompt(feature_data: dict):
    return f"""Based on your previous analysis and the domain knowledge contradictions identified, generate an adjusted shape function.
    
The adjusted graph must:
1. Maintain the same length as the original graph, meaning **exactly** {len(feature_data["x_vals"])} x-values and y-values).
2. Ensure the x-values are **identical** to the original.
3. Adjust only the y-values to reflect domain knowledge, addressing the contradictions identified earlier.

**Return only the adjusted y-values** in the same order, ensuring alignment with the original x-values. Do not include any further text or explanations, just the adjusted values in a list format, like so:

```python
# New y_values for shape function of feature {feature_data["feature_name"]} containing **exactly** {len(feature_data["x_vals"])} values
adjusted_y_vals = [<adjusted_y_value_1>, <adjusted_y_value_2>, ..., <adjusted_y_value_{len(feature_data["x_vals"]) - 1}>]
```end
"""

def get_explanation_system_msg(domain: str):
    return f"""
You are an expert in {domain}.
Your task is to explain why a graph was adjusted in clear and precise language.
Your explanation should be **one sentence long** and use terminology relevant to {domain}.
Avoid stating that an adjustment was made—focus solely on the reasoning behind it.
"""


def get_explanation_prompt(feature_data: dict, domain: str):
    return f"""
The shape function for {feature_data["feature_name"]} was adjusted.
Briefly explain **why** this adjustment was made in terms relevant to {domain}.
Keep your answer to **one sentence**.
"""

def parse_adjusted_y_vals(response: str, feature_data: dict):
    # Strip everything after the 'end' marker, including the marker itself
    python_code_block = response.split("```")[1].split("```")[0].strip()

    # Extract the adjusted y-values after the "adjusted_y_vals = " part and evaluate it
    adjusted_y_vals = ast.literal_eval(python_code_block.split("adjusted_y_vals = ")[1])

    expected_length = len(feature_data["x_vals"])
    actual_length = len(adjusted_y_vals)

    # Adjust the length to match the expected size
    if actual_length < expected_length:
        print(f"[Warning] Adjusted y-values list is too short. Expected {expected_length}, got {actual_length}. Adding {expected_length - actual_length} values.")
        last_value = adjusted_y_vals[-1] if adjusted_y_vals else 0  # Default to 0 if empty
        adjusted_y_vals.extend([last_value] * (expected_length - actual_length))
    elif actual_length > expected_length:
        print(f"[Warning] Adjusted y-values list is too long. Expected {expected_length}, got {actual_length}. Removing {actual_length - expected_length} values.")
        adjusted_y_vals = adjusted_y_vals[:expected_length]

    return adjusted_y_vals


