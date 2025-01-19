from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor
)
from typing import Union
from t2ebm.graphs import EBMGraph


def insert_graph(
        graph: EBMGraph,
        ebm: Union[ExplainableBoostingRegressor, ExplainableBoostingClassifier]
):
    feature_index = ebm.feature_names_in_.index(graph.feature_name)
    ebm.term_scores_[feature_index][1:-1] = graph.scores
    return ebm
