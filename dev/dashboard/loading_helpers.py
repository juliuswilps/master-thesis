from typing import Union
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret.glassbox._ebm._utils import convert_to_intervals
from scipy.interpolate import interp1d

def get_x_vals (
        ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
        feature_index: int,
    ):
    if ebm.feature_types_in_[feature_index] == "continuous":
        bins = convert_to_intervals(ebm.bins_[feature_index][0]) # Convert bin edges to interval tuples

        # Set feature bounds if necessary
        if bins[0][0] == float('-inf'):
            bins[0] = (ebm.feature_bounds_[feature_index][0], bins[0][1])
        if bins[-1][1] == float('inf'):
            bins[-1] = (bins[-1][0], ebm.feature_bounds_[feature_index][-1])

        # Calculate midpoints
        x_vals = [(bin[0] + bin[1]) / 2 for bin in bins]

        #y_vals = ebm.term_scores_[feature_index][1:-1] # drop missing/unknown bins

        #x_vals, y_vals = simplify_graph(x_vals, y_vals)

    else: # categorical features
        x_vals = ebm.bins_[feature_index][0] # TODO: Check categorical features
        #y_vals = ebm.term_scores_[feature_index][1:-1]

    return x_vals

def simplify_graph(
        x_vals: list,
        y_vals: list,
        min_variation_pct: float = 0.01
):
    """
    Simplifies a graph by adaptively merging adjacent bins with similar values.

    Args:
        x_vals (list): The x-values (midpoints of bins).
        y_vals (list): The y-values (scores from the EBM).
        min_variation_pct (float): Minimum relative variation between adjacent bins for them to be considered distinct.

    Returns:
        tuple: New simplified x_vals and y_vals (both lists).
    """
    # Calculate the total variation in the y-values
    total_variation = max(y_vals) - min(y_vals)

    # Initialize new lists for the simplified x and y values
    new_x_vals = []
    new_y_vals = []

    # Start with the first bin
    new_x_vals.append(x_vals[0])
    new_y_vals.append(y_vals[0])

    for i in range(1, len(x_vals)):
        prev_y = new_y_vals[-1]
        curr_y = y_vals[i]

        # Calculate the relative difference
        variation = abs(curr_y - prev_y) / total_variation

        if variation <= min_variation_pct:
            # Merge bins by averaging the x-values and keeping the previous y-value (or vice versa)
            new_x_vals[-1] = (new_x_vals[-1] + x_vals[i]) / 2  # Use the midpoint of the merged bins
        else:
            # Add the current bin as a new entry
            new_x_vals.append(x_vals[i])
            new_y_vals.append(curr_y)

    """if len(x_vals) != len(new_x_vals) or len(y_vals) != len(new_y_vals):
        print(f"Graph was simplified. \nOriginal x: {len(x_vals)} | New x: {len(new_x_vals)} \nOriginal y: {len(x_vals)} | New y: {len(new_x_vals)}")
    else:
        print("Graph was NOT simplified")"""

    return new_x_vals, new_y_vals

def interpolate_scores(x_simple, y_simple, x_complex):
    interp_func = interp1d(x_simple, y_simple, kind='linear', fill_value="extrapolate")
    y_complex = interp_func(x_complex)
    return y_complex

def get_shape_function_bins(ebm, feature_index, min_variation_pct=0.01):
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

