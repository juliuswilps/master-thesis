import joblib
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
from typing import Union
from loading_helpers import get_x_vals, simplify_graph, interpolate_scores
from adjust_graph import adjust_graph, adjust_graph_reasoning
import time


def load_ebm_data(ebm_path: str, description_path: str = ""):
    ebm = joblib.load(ebm_path)

    if isinstance(ebm, (ExplainableBoostingClassifier, ExplainableBoostingRegressor)):
        ebm_data = {}
        descriptions = {}

        if description_path:
            with open(description_path, "r") as f:
                descriptions = json.load(f)

        for idx, feature_name in enumerate(ebm.feature_names_in_):
            feature_type = ebm.feature_types_in_[idx]
            scores = ebm.term_scores_[idx][1:-1]
            x_vals =  get_x_vals(ebm, idx)

            ebm_data[feature_name] = {
                "x_vals": x_vals,
                "y_vals": [list(scores)],
                "adjusted_y_vals": [],
                "explanation": f"Graph for {feature_name}",
                "feature_type": feature_type,
                "feature_name": feature_name,
                "feature_description": descriptions.get(feature_name),
                "current_iteration": 0,
            }

        return ebm, ebm_data


    raise TypeError("The loaded object is not an Explainable Boosting Machine.")


def create_shape_function_plot(feature_data, state):
    # Set x_key and x_label for both feature types
    x_key = "X"
    x_label = "Category" if feature_data["feature_type"] == "nominal" else feature_data["feature_name"]

    # Create data for plotting
    plot_data = pd.DataFrame({
        "X": feature_data["x_vals"],
        "Influence Curve": feature_data["y_vals"][feature_data["current_iteration"]],
    })

    # Add the adjusted shape function if visible
    #if feature_data["adjusted_visible"]:
    if state["adjusted_visible"]:
        plot_data["Adjusted Influence Curve"] = feature_data["adjusted_y_vals"]

    # Select plot type based on feature type
    plot_func = px.bar if feature_data["feature_type"] == "nominal" else px.line

    color_map = {
        "Influence Curve": "blue",
        "Adjusted Influence Curve": "orange",
    }

    # Create the plot
    fig = plot_func(
        plot_data,
        x=x_key,
        y=plot_data.columns[1:],
        color_discrete_map=color_map,
        labels={"value": "Influence on Prediction", x_key: x_label},
        title=f"Influence Curve for {feature_data['feature_name']}",
    )

    fig.update_layout(
        xaxis=dict(autorange=True),  # Lock x-axis to autoscale
        yaxis=dict(autorange=True),  # Lock y-axis to autoscale
    )

    # Highlight baseline line
    fig.add_hline(y=0, line_dash="dash", line_color="red")

    return fig


def generate_adjusted_graph(ebm_data, selected_feature, state, simplify = False, reasoning = False):
    start_time = time.perf_counter()

    feature_data = ebm_data[selected_feature]

    # Use reasoning or convential LLM for API call
    api_call = adjust_graph_reasoning if reasoning else adjust_graph

    # Use simplified graph shorter runtime and less token usage
    if simplify:
        print("SIMPLE")
        # Save original graph
        x_original = feature_data["x_vals"]
        y_original = feature_data["y_vals"][feature_data["current_iteration"]]

        # Simplify graph by merging neighboring bins with marginally different y-values
        x_simple, y_simple = simplify_graph(x_original, y_original)
        feature_data["x_vals"] = x_simple
        feature_data["y_vals"][feature_data["current_iteration"]] = y_simple

        # Generate adjusted graph
        adjusted_y_simple, explanation = api_call(feature_data)

        # Interpolate adjusted graph to match original shape
        adjusted_y = interpolate_scores(x_simple, adjusted_y_simple, x_original)
        feature_data["x_vals"] = x_original
        feature_data["y_vals"][feature_data["current_iteration"]] = y_original

    # Use raw graph
    else:
        print("FULL")
        # Generate adjusted graph
        adjusted_y, explanation = api_call(feature_data)

    feature_data["adjusted_y_vals"] = adjusted_y
    feature_data["explanation"] = explanation
    state["adjusted_visible"] = True

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(execution_time)



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


def update_term_scores(ebm, ebm_data, selected_feature, adjusted=False):
    """
    Update the term_scores_ of the EBM model with adjusted values from ebm_data.

    Args:
        ebm: The Explainable Boosting Machine model.
        feature_data: The dictionary containing adjusted y_vals the selected feature.

    Returns:
        Updated EBM model with adjusted term_scores_.
    """
    updated_ebm = ebm.copy()

    for feature in updated_ebm.feature_names_in_:
        idx = ebm.feature_names_in_.index(feature)  # Get index of the feature
        if feature == selected_feature and adjusted:
            updated_ebm.term_scores_[idx][1:-1] = ebm_data[feature]["adjusted_y_vals"]
        else:
            updated_ebm.term_scores_[idx][1:-1] = ebm_data[feature]["y_vals"][ebm_data[feature]["current_iteration"]]

    """idx = ebm.feature_names_in_.index(ebm_data[selected_feature]["feature_name"])
    if adjusted:
        updated_ebm.term_scores_[idx][1:-1] = ebm_data[selected_feature]["adjusted_y_vals"]  # preserve missing and unknown bins
    else:
        updated_ebm.term_scores_[idx][1:-1] = ebm_data[selected_feature]["y_vals"][ebm_data[selected_feature]["current_iteration"]]
"""
    return updated_ebm

def keep_changes(ebm_data: dict, selected_feature: str, state):
    """
    Keeps the adjusted shape function and updates the original function with the adjusted one.

    Args:
        ebm_data (dict): The session state.
        selected_feature (str): The selected feature name.
    """
    feature_data = ebm_data[selected_feature]
    feature_data["y_vals"].append(list(feature_data["adjusted_y_vals"]))
    feature_data["current_iteration"] += 1
    #feature_data["adjusted_visible"] = False
    state["adjusted_visible"] = False



def discard_changes(ebm_data: dict, selected_feature: str, state):
    """
    Discards the adjusted changes and reverts to the original shape function.

    Args:
        feature (str): The selected feature name.
        state (dict): The session state.
    """
    ebm_data[selected_feature]["adjusted_y_vals"] = []
    #ebm_data[selected_feature]["adjusted_visible"] = False
    state["adjusted_visible"] = False


def save_adjusted_model(
        ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
        ebm_data: dict,
        save_path: str
):
    """
    Save the adjusted model by updating its term_scores_ with values from ebm_data.

    Args:
        ebm (ExplainableBoostingClassifier/ExplainableBoostingRegressor): The EBM model in memory.
        ebm_data (dict): The updated EBM data containing y_vals for each feature.
        save_path (str): Path to save the adjusted EBM.
    """
    for feature_name, feature_data in ebm_data.items():
        idx = ebm.feature_names_in_.index(feature_name)
        ebm.term_scores_[idx][1:-1] = feature_data["y_vals"][feature_data["current_iteration"]] # preserve missing and unknown bins
    joblib.dump(ebm, save_path)

def previous_iteration(ebm_data: dict, selected_feature: str):
    feature_data = ebm_data[selected_feature]
    if feature_data["current_iteration"] > 0:
        feature_data["current_iteration"] -= 1
        st.rerun()
    else:
        st.warning("No earlier versions available!")

def next_iteration(ebm_data: dict, selected_feature: str):
    feature_data = ebm_data[selected_feature]
    if feature_data["current_iteration"] < len(feature_data["y_vals"]) - 1:
        feature_data["current_iteration"] += 1
        st.rerun()
    else:
        st.warning("No later versions available!")
