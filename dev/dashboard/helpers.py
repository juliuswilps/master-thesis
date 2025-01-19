import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
from t2ebm_helpers import extract_graph


def load_model(file_path: str):
    ebm = joblib.load(file_path)
    if isinstance(ebm, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
        return ebm
    raise TypeError("The loaded object is not an Explainable Boosting Machine.")


def load_ebm_data(file_path: str):
    ebm = joblib.load(file_path)

    if isinstance(ebm, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
        ebm_data = {}

        for i, graph in enumerate(
                [extract_graph(ebm, idx) for idx in range(len(ebm.feature_names_in_))]
        ):
            if graph.feature_type == "continuous":
                x_values = [(interval[0] + interval[1]) / 2 for interval in graph.x_vals]
            elif graph.feature_type == "nominal":
                x_values = graph.x_vals
            else:
                raise ValueError(f"Unknown feature type: {graph.feature_type}")

            ebm_data[graph.feature_name] = {
                "x_vals": x_values,
                "y_vals": graph.scores,
                "adjusted_y_vals": [],
                "adjusted_visible": False,
                "stds": graph.stds,
                "explanation": f"Graph for {graph.feature_name}",
                "feature_type": graph.feature_type,
                "feature_name": graph.feature_name,
            }
        return ebm, ebm_data


def create_shape_function_plot(feature_data):
    # Set x_key and x_label for both feature types
    x_key = "X"
    x_label = "Category" if feature_data["feature_type"] == "nominal" else "X-axis"

    # Create data for plotting
    plot_data = pd.DataFrame({
        "X": feature_data["x_vals"],
        "Original Shape Function": feature_data["y_vals"],
    })

    # Add the adjusted shape function if visible
    if feature_data["adjusted_visible"]:
        plot_data["Adjusted Shape Function"] = feature_data["adjusted_y_vals"]

    # Select plot type based on feature type
    plot_func = px.bar if feature_data["feature_type"] == "nominal" else px.line

    # Create the plot
    fig = plot_func(
        plot_data,
        x=x_key,
        y=plot_data.columns[1:],  # Handles both Original and Adjusted shapes
        labels={"value": "Function Value", x_key: x_label},
        title=f"Shape Function for {feature_data['feature_name']}",
    )

    # Highlight baseline line
    fig.add_hline(y=0, line_dash="dash", line_color="red")

    return fig


# Helper function to generate adjusted graph
def generate_adjusted_graph(feature_name: str, feature_type: str, x_values, state):
    """
    Generate adjusted shape values for the selected feature.

    Args:
        feature_name (str): Name of the selected feature.
        feature_type (str): Type of the feature (nominal or continuous).
        x_values (list): X-values of the feature.
    """
    # Static demo values for now
    if feature_type == "continuous":
        # Use a sine wave to make the adjusted graph visually distinct
        adjusted_values = np.sin(2 * np.pi * np.array(x_values)) + 0.5
    elif feature_type == "nominal":
        adjusted_values = [1.5, -0.5, 0.5]  # Example for nominal data
    else:
        # Fallback for unknown features
        adjusted_values = [0] * len(x_values)  # Default fallback

    # Save adjusted values to session state
    state["ebm_data"][feature_name]["adjusted_y_vals"] = adjusted_values
    state["ebm_data"][feature_name]["adjusted_visible"] = True

def calculate_model_accuracy(ebm, test_data_path):
    """
    Calculates the accuracy of an Explainable Boosting Machine on a test dataset.

    Parameters:
        ebm: The Explainable Boosting Machine (either classifier or regressor).
        test_data_path: Path to the CSV file containing the test dataset.

    Returns:
        Accuracy as a float value (accuracy score for classification or R² score for regression).
    """
    # Load test dataset
    test_data = pd.read_csv(test_data_path)

    # Split features and target
    X_test = test_data.iloc[:, :-1]  # All columns except the last one
    y_test = test_data.iloc[:, -1]  # Last column as the target

    # Make predictions
    y_pred = ebm.predict(X_test)

    # Calculate accuracy
    if isinstance(ebm, ExplainableBoostingClassifier):
        accuracy = accuracy_score(y_test, y_pred)
    elif isinstance(ebm, ExplainableBoostingRegressor):
        accuracy = r2_score(y_test, y_pred)
    else:
        raise TypeError("The provided model is not a supported EBM.")

    return accuracy


def update_term_scores(ebm, feature_data):
    """
    Update the term_scores_ of the EBM model with adjusted values from ebm_data.

    Args:
        ebm: The Explainable Boosting Machine model.
        feature_data: The dictionary containing adjusted y_vals the selected feature.

    Returns:
        Updated EBM model with adjusted term_scores_.
    """
    adjusted_ebm = ebm.copy()
    idx = ebm.feature_names_in_.index(feature_data["feature_name"])
    adjusted_ebm.term_scores_[idx][1:-1] = feature_data["adjusted_y_vals"]  # preserve missing and unknown bins

    return adjusted_ebm

def keep_changes(feature, state):
    """
    Keeps the adjusted shape function and updates the original function with the adjusted one.

    Args:
        feature (str): The selected feature name.
        state (dict): The session state.
    """
    state["ebm_data"][feature]["y_vals"] = state["ebm_data"][feature]["adjusted_y_vals"]
    state["ebm_data"][feature]["adjusted_y_vals"] = []
    state["ebm_data"][feature]["adjusted_visible"] = False


def discard_changes(feature, state):
    """
    Discards the adjusted changes and reverts to the original shape function.

    Args:
        feature (str): The selected feature name.
        state (dict): The session state.
    """
    state["ebm_data"][feature]["adjusted_y_vals"] = []
    state["ebm_data"][feature]["adjusted_visible"] = False


def save_adjusted_model(ebm, ebm_data: dict, save_path: str):
    """
    Save the adjusted model by updating its term_scores_ with values from ebm_data.

    Args:
        file_path (str): Path to the original .pkl file containing the EBM.
        ebm_data (dict): The updated EBM data containing y_vals for each feature.
        save_path (str): Path to save the adjusted EBM.
    """

    if isinstance(ebm, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
        for feature_name, feature_data in ebm_data.items():
            idx = ebm.feature_names_in_.index(feature_name)
            ebm.term_scores_[idx][1:-1] = feature_data["y_vals"] # preserve missing and unknown bins

        # Save the adjusted model
        joblib.dump(ebm, save_path)
        return

    raise TypeError("The loaded object is not an Explainable Boosting Machine.")


