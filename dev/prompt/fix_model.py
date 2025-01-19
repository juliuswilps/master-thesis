from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from typing import Union
from load_api_key import load_openai_api_key
load_openai_api_key()
from t2ebm import graphs

import joblib
def fix_model(
        ebm: Union[ExplainableBoostingRegressor, ExplainableBoostingClassifier],
    ):
    if not isinstance(ebm, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
        print("fail")

    # setup_llm(model, role + context, content)

    #print(ebm.feature_names_in_)
    for i in range(len(ebm.feature_names_in_)):
        graph = graphs.extract_graph(ebm, i)
        #graphs.plot_graph(graph)

        string = graphs.graph_to_text(graph)

        # graph_as_text = edit_text(string)

        # prompt = get_prompt(graph_as_text)
        # Wie sehen caafe/ t2ebm prompts aus?

        # response = api_call(prompt)

        # execute_response()

        # highlight_changes()

        # user_decision()
    return 0


# Testing
#ebm = joblib.load("../trained_ebm.pkl")
#fix_model(ebm=ebm)