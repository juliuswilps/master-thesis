def simplify_graph_adaptive(
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

    return new_x_vals, new_y_val