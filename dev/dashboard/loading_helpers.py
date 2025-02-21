from typing import Union
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret.glassbox._ebm._utils import convert_to_intervals

def get_x_vals (
        ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
        feature_index: int
    ):
    if ebm.feature_types_in_[feature_index] == "continuous":
        intervals = convert_to_intervals(ebm.bins_[feature_index][0]) # Convert bin edges to interval tuples

        # Set feature bounds if necessary
        if intervals[0][0] == float('-inf'):
            intervals[0] = (ebm.feature_bounds_[feature_index][0], intervals[0][1])
        if intervals[-1][1] == float('inf'):
            intervals[-1] = (intervals[-1][0], ebm.feature_bounds_[feature_index][-1])

        # Calculate midpoints
        x_vals = [(interval[0] + interval[1]) / 2 for interval in intervals]

    elif ebm.feature_types_in_ == "nominal":
        x_vals = ebm.bins_[feature_index][0] # TODO: Check categorical features

    return x_vals






    """    
    if feature_type == "continuous":
        x_vals = convert_to_intervals(ebm.bins_[feature_index][0])
        # feature bounds apply to continuous features only
        if use_feature_bounds:
            x_vals[0] = (ebm.feature_bounds_[feature_index][0], x_vals[0][1])
            x_vals[-1] = (x_vals[-1][0], ebm.feature_bounds_[feature_index][1])
    elif feature_type == "nominal":
        x_vals = ebm.bins_[feature_index][
            0
        ]
    """