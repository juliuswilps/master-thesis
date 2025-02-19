import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
from typing import Union
from dotenv import load_dotenv
load_dotenv()
from t2ebm.graphs import extract_graph
from functions import adjust_graph


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
                "y_vals": [list(graph.scores)],
                "adjusted_y_vals": [],
                "adjusted_visible": False,
                "stds": graph.stds,
                "explanation": f"Graph for {graph.feature_name}",
                "feature_type": graph.feature_type,
                "feature_name": graph.feature_name,
                "current_iteration": 0,
            }
        return ebm, ebm_data


def create_shape_function_plot(feature_data):
    # Set x_key and x_label for both feature types
    x_key = "X"
    x_label = "Category" if feature_data["feature_type"] == "nominal" else feature_data["feature_name"]

    # Create data for plotting
    plot_data = pd.DataFrame({
        "X": feature_data["x_vals"],
        "Influence Curve": feature_data["y_vals"][feature_data["current_iteration"]],
    })

    # Add the adjusted shape function if visible
    if feature_data["adjusted_visible"]:
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


# Helper function to generate adjusted graph
def generate_adjusted_graph(feature_name: str, state):
    """
    Generate adjusted shape values for the selected feature.

    Args:
        feature_name (str): Name of the selected feature.
    """
    if feature_name == "Credit Score":
        adjusted_scores = [-y for y in state["ebm_data"][feature_name]["y_vals"][state["ebm_data"][feature_name]["current_iteration"]]]
        explanation = "I inverted the y-values because the original shape function contradicted domain knowledge by assigning lower probabilities of loan repayment to higher credit scores. In reality, a higher credit score should indicate a lower risk of default, meaning the function should have positive values for high scores and negative values for low scores. This simple inversion corrects the direction while preserving the relative differences between values."
        state["ebm_data"][feature_name]["explanation"] = explanation

    else:
        llm = "gpt-4o-mini"
        ebm = state.ebm
        idx = ebm.feature_names_in_.index(feature_name)

        adjusted_scores, explanation = adjust_graph(llm, ebm, idx)

    # Save adjusted values to session state
    state["ebm_data"][feature_name]["adjusted_y_vals"] = adjusted_scores
    state["ebm_data"][feature_name]["explanation"] = explanation
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


def update_term_scores(ebm, feature_data, adjusted=False):
    """
    Update the term_scores_ of the EBM model with adjusted values from ebm_data.

    Args:
        ebm: The Explainable Boosting Machine model.
        feature_data: The dictionary containing adjusted y_vals the selected feature.

    Returns:
        Updated EBM model with adjusted term_scores_.
    """
    updated_ebm = ebm.copy()
    idx = ebm.feature_names_in_.index(feature_data["feature_name"])
    if adjusted:
        updated_ebm.term_scores_[idx][1:-1] = feature_data["adjusted_y_vals"]  # preserve missing and unknown bins
    else:
        updated_ebm.term_scores_[idx][1:-1] = feature_data["y_vals"][feature_data["current_iteration"]]

    return updated_ebm

def keep_changes(ebm_data: dict, selected_feature: str):
    """
    Keeps the adjusted shape function and updates the original function with the adjusted one.

    Args:
        ebm_data (dict): The session state.
        selected_feature (str): The selected feature name.
    """
    feature_data = ebm_data[selected_feature]
    feature_data["y_vals"].append(list(feature_data["adjusted_y_vals"]))
    feature_data["current_iteration"] += 1
    feature_data["adjusted_visible"] = False


def discard_changes(ebm_data: dict, selected_feature: str):
    """
    Discards the adjusted changes and reverts to the original shape function.

    Args:
        feature (str): The selected feature name.
        state (dict): The session state.
    """
    ebm_data[selected_feature]["adjusted_y_vals"] = []
    ebm_data[selected_feature]["adjusted_visible"] = False


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