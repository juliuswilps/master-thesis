from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from typing import Union


import joblib
def fix_model(
        ebm: Union[ExplainableBoostingRegressor, ExplainableBoostingClassifier],
    ):
    if not isinstance(ebm, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
        print("fail")

    # setup_llm(model, role + context, content)

    # simplify graph?

    # string = graphs.graph_to_text(graph)

    # graph_as_text = edit_text(string)

    # prompts.py = get_prompt(graph_as_text)
    # Wie sehen caafe/ t2ebm prompts aus?

    # response = api_call(prompts.py)

    # execute_response()
