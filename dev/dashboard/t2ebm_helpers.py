import typing
import numpy as np
from dataclasses import dataclass
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from typing import Union
from interpret.glassbox._ebm._utils import convert_to_intervals

@dataclass
class EBMGraph:
    """A datastructure for the graphs of Explainable Boosting Machines.
    """
    feature_name: str
    feature_type: str
    x_vals: typing.List[
        typing.Tuple[float, float]
    ]  # todo add union with categorical features
    scores: typing.List[float]
    stds: typing.List[float]


def extract_graph(
        ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
        feature_index: int,
        normalization="none",
        use_feature_bounds=True,
):
    """Extract the graph of a feature from an Explainable Boosting Machine.

    This is a low-level function. It does not return the final format in which the graph is presented to the LLM.

    The purpose of this function is to extract the graph from the interals of the EBM and return it in a format that is easy to work with.

    Args:
        ebm (_type_): The EBM.
        feature_index (int): The index of the feature in the EBM.
        normalization (str, optional): How to normalize the graph (shift on the y-axis). Possible values are: 'mean', 'min', 'none'. Defaults to "none".
        use_feature_bounds (bool, optional): If True, the first and last x-axis bins are the min and max values of the feature stored in the EBM. If false, the first and last value are -inf and inf, respectively. Defaults to True.

    Raises:
        Exception: If an error occurs.

    Returns:
        EBMGraph: The graph.
    """
    # read the variables from the ebm
    feature_name = ebm.feature_names_in_[feature_index]
    feature_type = ebm.feature_types_in_[feature_index]
    scores = ebm.term_scores_[feature_index][1:-1]  # Drop missing and unknown bins
    stds = ebm.standard_deviations_[feature_index][1:-1]

    # normalize the graph
    normalization_constant = None
    if normalization == "mean":
        normalization_constant = np.mean(scores)
    elif normalization == "min":
        normalization_constant = np.min(scores)
    elif normalization == "none":
        normalization_constant = 0
    else:
        raise Exception(f"Unknown normalization {normalization}")
    scores = scores - normalization_constant

    # read the x-axis bins from the ebm
    if feature_type == "continuous":
        x_vals = convert_to_intervals(ebm.bins_[feature_index][0])
        # feature bounds apply to continuous features only
        if use_feature_bounds:
            x_vals[0] = (ebm.feature_bounds_[feature_index][0], x_vals[0][1])
            x_vals[-1] = (x_vals[-1][0], ebm.feature_bounds_[feature_index][1])
    elif feature_type == "nominal":
        x_vals = ebm.bins_[feature_index][
            0
        ]  # todo: check this transformation with Paul
        x_vals = {v - 1: k for k, v in x_vals.items()}
        x_vals = [x_vals[idx] for idx in range(len(x_vals.keys()))]
    else:
        raise Exception(
            f"Feature {feature_index} is of unknown feature_type {feature_type}."
        )
    assert len(x_vals) == len(scores), "The number of bins and scores does not match."

    return EBMGraph(feature_name, feature_type, x_vals, scores, stds)