from t2ebm.graphs import EBMGraph
import copy
import ast
from typing import Optional
from caafe.run_llm_code import check_ast
import numpy as np

def get_prompt(
    ebm_graph: EBMGraph, data_description_unparsed=None, **kwargs
):
    """
    Generates a prompts.py to identify and correct only the most significant domain knowledge anomaly in an EBM graph shape function.

    Parameters:
    ebm_graph (EBMGraph): The EBM graph object containing feature name, type, x-values, scores, and standard deviations.
    data_description_unparsed (str, optional): A description of the dataset for additional context.
    """

    return f"""
    The Explainable Boosting Machine (EBM) shape function graph for a specific feature is loaded as the `EBMGraph` object.

    Feature analyzed: "{ebm_graph.feature_name}" (Type: {ebm_graph.feature_type}).

    x-values and corresponding shape scores represent the relationship between this feature and the target:
    - X-values (ranges or categories): {ebm_graph.x_vals}
    - Shape function scores: {ebm_graph.scores}
    - Standard deviations (uncertainty at each x-value): {ebm_graph.stds}

    Dataset description for context:
    "{data_description_unparsed}"

    Task:
    1. Identify the **single most significant anomaly** in the shape function scores for this feature. The anomaly should be based on domain expectations (e.g., smooth score progression for numeric features, logical ordering for categorical features).
    2. Explain why this anomaly may affect interpretability or predictive accuracy if left unaddressed.
    3. Provide a code snippet to correct this specific anomaly in the `EBMGraph.scores` list, modifying only the values needed for this correction.

    Examples of common anomalies to prioritize:
    - For numeric features: A sudden score jump without domain basis, suggesting a lack of smoothness in the progression.
    - For categorical features: Score inconsistencies between categories that defy logical or domain expectations.
    - Large deviations in shape scores that correlate with high standard deviations, indicating unreliable scores needing adjustment.

    Code format for correction:
    ```python
    # (Brief description of the identified anomaly in "{ebm_graph.feature_name}")
    # Correction rationale: (Explanation for the change based on domain knowledge.)
    # Corrective code:
    for i, (x_val, score) in enumerate(zip({ebm_graph.x_vals}, {ebm_graph.scores})):
        if some_condition:  (Define condition for the identified anomaly)
            ebm_graph.scores[i] = new_score_value  # Define the correction
    ```end

    Each codeblock should only address **one significant anomaly** and suggest the simplest adjustment necessary to address it.
    Codeblock:
    """

def generate_code(messages, client):
    """if model == "skip":
        return ""
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stop=["```end"],
        temperature=0.5,
        max_tokens=500,
    )
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("<end>", "")
    return code

def run_llm_code(
    code: str, ebm_graph: EBMGraph
) -> EBMGraph:
    """
    Executes the given code on the given EBMGraph object and returns the modified object.

    Parameters:
    code (str): The code to execute.
    ebm_graph (EBMGraph): The EBMGraph object to execute the code on.

    Returns:
    EBMGraph: The resulting EBMGraph after executing the code.
    """
    try:
        # Create a deep copy of ebm_graph to avoid modifying the original object
        ebm_graph_copy = copy.deepcopy(ebm_graph)

        # Define the local scope in which to execute the code
        access_scope = {
            "ebm_graph": ebm_graph_copy,
            "np": np,
        }

        # Parse and validate the code to ensure it's safe to execute
        parsed = ast.parse(code)
        #check_ast(parsed)  # Assuming check_ast validates the code for safety

        # Execute the code within the restricted scope
        exec(compile(parsed, filename="<ast>", mode="exec"), access_scope)

        # After execution, return the modified ebm_graph_copy
        return ebm_graph_copy

    except Exception as e:
        print("Code could not be executed:", e)
        raise e
