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

def graph_to_text(
    graph: EBMGraph,
    include_description=True,
    feature_format=None,
    x_axis_precision=None,
    y_axis_precision="auto",
    confidence_bounds=True,
    confidence_level=0.95,
    max_tokens=3000,
):
    """Convert an EBMGraph to text. This is the text that we then pass to the LLM.

    The function takes care of a variety of different formatting issues that can arise in the process of converting a graph to text.

    Args:
        graph (EBMGraph): The graph.
        include_description (bool, optional): Whether to include a short descriptive preamble that describes the graph to the LLM. Defaults to True.
        feature_format (_type_, optional): The format of the feature (continuous, cateorical, boolean). Defaults to None (auto-detect).
        x_axis_precision (_type_, optional): The precision of the values on the x-axis. Defaults to None.
        y_axis_precision (str, optional): The precision of the values on the x-axis. Defaults to "auto".
        confidence_bounds (bool, optional): Whether to inlcude confidence bounds. Defaults to True.
        confidence_level (float, optional): The desired confidence level of the bounds. Defaults to 0.95.
        max_tokens (int, optional): The maximum number of tokens that the textual description of the graph can have. The function simplifies the graph so to fit into this token limit. Defaults to 3000.

    Raises:
        Exception: If an error occurs.

    Returns:
        str: The textual representation of the graph.
    """
    # a simple auto-detect for boolean feautres
    try:
        if (
            len(graph.x_vals) == 2
            and graph.x_vals[0].upper() == "FALSE"
            and graph.x_vals[1].upper() == "TRUE"
        ):
            feature_format = "boolean"
    except:
        pass

    # determine the feature format
    if feature_format is None:
        feature_format = graph.feature_type
    if feature_format == "nominal":
        feature_format = "categorical"

    # the description of the graph depends on the feature format
    if feature_format == "continuous":
        description_text = (
            "This graph represents a continuous-valued feature. The keys are intervals"
            " that represent ranges where the function predicts the same value.\n\n"
        )
    elif feature_format == "categorical":
        description_text = (
            "This graph represents categorical feature. Each key represents a possible"
            " value that the feature can take.\n\n"
        )
    elif feature_format == "boolean":
        description_text = (
            "This graph represents a boolean feature. The keys are 'True' and 'False',"
            " the two possible values of the feature.\n\n"
        )
    else:
        raise Exception(f"Unknown feature format {feature_format}")

    # simplify the graph until it fits into the max_tokens limit
    total_tokens = None
    min_variation_per_cent = 0.0
    while True:
        # simplify the graph
        simplified_graph = graph
        if feature_format == "continuous":
            simplified_graph = simplify_graph(
                graph, min_variation_per_cent=min_variation_per_cent
            )

        # confidence bounds via normal approximation
        scores = simplified_graph.scores
        if confidence_bounds:
            factor = scipy.stats.norm.interval(confidence_level, loc=0, scale=1)[1]
            lower_bounds = [
                scores[idx] - factor * simplified_graph.stds[idx]
                for idx in range(len(scores))
            ]
            upper_bounds = [
                scores[idx] + factor * simplified_graph.stds[idx]
                for idx in range(len(scores))
            ]

        # adjust the precision on the x-axis. this can be dangerous if we don't know the distribution of the data (it can impact the accuracy).
        # however, this is a necessary formatting feature because certain (old?) versions of EBMs introduce trailing digits for the xbins (e.g. age as a floating point number with 10 significant digits)
        x_vals = simplified_graph.x_vals
        if x_axis_precision is not None:
            x_vals = [
                (
                    np.round(x[0], x_axis_precision).astype(str),
                    np.round(x[1], x_axis_precision).astype(str),
                )
                for x in x_vals
            ]

        # adjust the precision on the y-axis
        if y_axis_precision == "auto":  # at least 3 significant digits
            total_variation = np.max(scores) - np.min(scores)
            y_fraction = total_variation / 100.0
            y_axis_precision = 1
            while y_fraction < 1:
                y_fraction *= 10
                y_axis_precision += 1
        scores = np.round(scores, y_axis_precision).astype(str)
        if confidence_bounds:
            lower_bounds = np.round(lower_bounds, y_axis_precision).astype(str)
            upper_bounds = np.round(upper_bounds, y_axis_precision).astype(str)

        # formatting for boolean features
        if feature_format == "boolean":
            assert len(x_vals) == 2, (
                "Requested a boolean format, but the feature has more than x-axis"
                f" values: {x_vals}"
            )
            # we assume that the left key (the lower numer, typically -1 or 0), is 'False'
            x_vals = ["False", "True"]

        # create the textual representation of the graph
        prompt = ""
        if include_description:
            prompt += description_text
        prompt += f"Feature Name: {graph.feature_name}\n"
        prompt += f"Feature Type: {feature_format}\n"
        prompt += f"Means: {xy_to_json_(x_vals, scores)}\n"
        if confidence_bounds:
            prompt += (
                f"Lower Bounds ({confidence_level*100:.0f}%-Confidence Interval):"
                f" {xy_to_json_(x_vals, lower_bounds)}\n"
            )
            prompt += (
                f"Upper Bounds ({confidence_level*100:.0f}%-Confidence Interval):"
                f" {xy_to_json_(x_vals, upper_bounds)}\n"
            )

        # count the number of tokens
        total_tokens = num_tokens_from_string_(prompt, "gpt-4")
        if feature_format == "continuous":
            if total_tokens > max_tokens:
                if min_variation_per_cent > 0.1:
                    raise Exception(
                        f"The graph for feature {graph.feature_name} of type"
                        f" {graph.feature_type} requires {total_tokens} tokens even at"
                        " a simplification level of 10%. This graph is too complex to"
                        " be passed to the LLM within the loken limit of"
                        f" {max_tokens} tokens."
                    )
                min_variation_per_cent += 0.001
            else:
                if min_variation_per_cent > 0:
                    print(
                        f"INFO: The graph of feature {graph.feature_name} was"
                        f" simplified by {min_variation_per_cent * 100:.1f}%."
                    )
                return prompt
        else:
            if total_tokens > max_tokens:
                raise Exception(
                    f"The graph for feature {graph.feature_name} of type"
                    f" {graph.feature_type} requires {total_tokens} tokens and exceeds"
                    f" the token limit of {max_tokens}."
                )
            else:
                return prompt