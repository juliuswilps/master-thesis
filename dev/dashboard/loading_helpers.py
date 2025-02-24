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

def get_shape_function(ebm, feature_index, min_variation_pct=0.01):
    """
    Simplify the bins of a given feature in the EBM and load the simplified data into ebm_data.

    Args:
        ebm (ExplainableBoostingClassifier/Regressor): The trained EBM model.
        feature_index (int): The index of the feature to simplify.
        min_variation_pct (float): The threshold percentage for bin simplification.

    Returns:
        simplified_x_vals (list): Simplified midpoints for the bins.
        simplified_y_vals (list): Simplified y-values for the shape function.
    """
    if ebm.feature_types_in_[feature_index] == "continuous":
        # Get the original bin edges and scores for the feature
        bin_edges = ebm.bins_[feature_index][0]
        scores = ebm.term_scores_[feature_index][1:-1]  # drop missing/ unknown values

        # Simplify the bins by merging adjacent bins with small score differences
        new_bins = []
        new_scores = []
        prev_bin_start = bin_edges[0]
        prev_bin_end = bin_edges[0]
        prev_score = scores[0]

        for i in range(1, len(bin_edges)):
            current_bin_end = bin_edges[i]
            current_score = scores[i]

            # Check if the score difference is below the threshold
            if abs(prev_score - current_score) <= (max(scores) - min(scores)) * min_variation_pct:
                # Merge the bins by extending the previous bin
                prev_bin_end = current_bin_end
                prev_score = (prev_score + current_score) / 2  # Average score
            else:
                # Add the previous bin to the new bins list
                new_bins.append((prev_bin_start, prev_bin_end))
                new_scores.append(prev_score)
                prev_bin_start = current_bin_end
                prev_score = current_score

        # Add the last bin
        new_bins.append((prev_bin_start, bin_edges[-1]))
        new_scores.append(prev_score)

        # Ensure the first and last bins are within the feature bounds
        #if ebm.feature_bounds_[feature_index]:
            # Adjust the first bin to stay within bounds
        new_bins[0] = (max(new_bins[0][0], ebm.feature_bounds_[feature_index][0]), new_bins[0][1])
            # Adjust the last bin to stay within bounds
        new_bins[-1] = (new_bins[-1][0], min(new_bins[-1][1], ebm.feature_bounds_[feature_index][1]))

        # Recalculate midpoints for x_vals
        x_vals = [(bin[0] + bin[1]) / 2 for bin in new_bins]

        # Update the EBM's bins and term_scores_
        ebm.bins_[feature_index][0] = new_bins
        ebm.term_scores_[feature_index] = new_scores

    else:
        x_vals = ebm.bins_[feature_index][0]
        new_scores = ebm.term_scores_[feature_index][1:-1]

    # Return the simplified x_vals and scores
    return x_vals, new_scores, ebm

