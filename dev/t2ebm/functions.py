"""
TalkToEBM: A Natural Language Interface to Explainable Boosting Machines.
"""

import inspect

from typing import Union

import t2ebm
import t2ebm.llm
from t2ebm.llm import AbstractChatModel

from graphs import extract_graph, graph_to_text

import prompts

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)

import json

###################################################################################################
# Talk to the EBM about other things than graphs.
###################################################################################################
def adjust_graph(
    llm: Union[AbstractChatModel, str],
    ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
    feature_index: int,
    num_sentences: int = 7,
    **kwargs,
):
    """Ask the LLM to describe a graph. Uses chain-of-thought reasoning.

    The function accepts additional keyword arguments that are passed to extract_graph, graph_to_text, and describe_graph_cot.

    Args:
        llm (Union[AbstractChatModel, str]): The LLM.
        ebm (Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor]): The EBM.
        feature_index (int): The index of the feature to describe.
        num_sentences (int, optional): The desired number of senteces for the description. Defaults to 7.

    Returns:
        dict:  The description of the graph.
    """

    # llm setup
    llm = t2ebm.llm.setup(llm)

    print(f"Feature to adjust: {ebm.feature_names_in_[feature_index]}")
    print(f"ebm term scores: {len(ebm.term_scores_[feature_index][1:-1])}")

    # extract the graph from the EBM
    #extract_kwargs = list(inspect.signature(extract_graph).parameters)
    #extract_dict = {k: kwargs[k] for k in dict(kwargs) if k in extract_kwargs}
    #graph = extract_graph(ebm, feature_index, **extract_dict)

    num_scores = len(graph.scores)
    print(f"graph.scores: {num_scores}")

    # convert the graph to text
    to_text_kwargs = list(inspect.signature(graph_to_text).parameters)
    to_text_dict = {k: kwargs[k] for k in dict(kwargs) if k in to_text_kwargs}
    graph_string = graph_to_text(graph, confidence_bounds=False, **to_text_dict)

    print(f"graph_string: {graph_string}")

    # get a cot sequence of messages to describe the graph
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {k: kwargs[k] for k in dict(kwargs) if k in llm_descripe_kwargs}
    messages = prompts.describe_graph_cot(
        graph_string, num_scores=num_scores, num_sentences=num_sentences, **llm_descripe_dict
    )

    # execute the prompt
    response = t2ebm.llm.chat_completion(llm, messages)

    # the last message contains the summary
    # Extract and parse the last response

    response_content = response[-1]["content"]  # Get the last response
    code = response_content.strip("```json").strip("```").strip()
    print(f"response: {code}")
    parsed_response = json.loads(code)

    # Extract values safely
    adjusted_scores = parsed_response.get("adjusted_scores", [])
    explanation = parsed_response.get("reason", "No explanation provided.")

    # Ensure correct length
    if len(adjusted_scores) != num_scores:
        print(f"Warning: Adjusted scores length ({len(adjusted_scores)}) does not match expected ({num_scores})")

        # If too many values, truncate
        if len(adjusted_scores) > num_scores:
            adjusted_scores = adjusted_scores[:num_scores]
        # If too few values, pad with the last known value (or 0)
        else:
            last_value = adjusted_scores[-1] if adjusted_scores else 0
            adjusted_scores.extend([last_value] * (num_scores - len(adjusted_scores)))

    print(f"len(adjusted_scores) final:{len(adjusted_scores)}")

    return adjusted_scores, explanation


################################################################################################################
# Ask the LLM to perform high-level tasks directly on the EBM.
################################################################################################################

def describe_graph(
    llm: Union[AbstractChatModel, str],
    ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
    feature_index: int,
    num_sentences: int = 7,
    **kwargs,
):
    """Ask the LLM to describe a graph. Uses chain-of-thought reasoning.

    The function accepts additional keyword arguments that are passed to extract_graph, graph_to_text, and describe_graph_cot.

    Args:
        llm (Union[AbstractChatModel, str]): The LLM.
        ebm (Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor]): The EBM.
        feature_index (int): The index of the feature to describe.
        num_sentences (int, optional): The desired number of senteces for the description. Defaults to 7.

    Returns:
        str:  The description of the graph.
    """

    # llm setup
    llm = t2ebm.llm.setup(llm)

    # extract the graph from the EBM
    extract_kwargs = list(inspect.signature(extract_graph).parameters)
    extract_dict = {k: kwargs[k] for k in dict(kwargs) if k in extract_kwargs}
    graph = extract_graph(ebm, feature_index, **extract_dict)

    # convert the graph to text
    to_text_kwargs = list(inspect.signature(graph_to_text).parameters)
    to_text_dict = {k: kwargs[k] for k in dict(kwargs) if k in to_text_kwargs}
    graph = graph_to_text(graph, **to_text_dict)

    # get a cot sequence of messages to describe the graph
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {k: kwargs[k] for k in dict(kwargs) if k in llm_descripe_kwargs}
    messages = prompts.describe_graph_cot(
        graph, num_sentences=num_sentences, **llm_descripe_dict
    )

    # execute the prompt
    messages = t2ebm.llm.chat_completion(llm, messages)

    # the last message contains the summary
    return messages[-1]["content"]

def describe_ebm(
    llm: Union[AbstractChatModel, str],
    ebm: Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor],
    num_sentences: int = 30,
    **kwargs,
):
    """Ask the LLM to describe an EBM.

    The function accepts additional keyword arguments that are passed to extract_graph, graph_to_text, and describe_graph_cot.

    Args:
        llm (Union[AbstractChatModel, str]): The LLM.
        ebm (Union[ExplainableBoostingClassifier, ExplainableBoostingRegressor]): The EBM.
        num_sentences (int, optional): The desired number of senteces for the description. Defaults to 30.

    Returns:
        str: The description of the EBM.
    """

    # llm setup
    llm = t2ebm.llm.setup(llm)

    # Note: We convert all objects to text before we prompt the LLM the first time.
    # The idea is that if there is an error processing one of the graphs, we get it before we prompt the LLM.
    feature_importances = feature_importances_to_text(ebm)

    # extract the graphs from the EBM
    extract_kwargs = list(inspect.signature(extract_graph).parameters)
    extract_dict = {k: kwargs[k] for k in dict(kwargs) if k in extract_kwargs}
    graphs = []
    for feature_index in range(len(ebm.feature_names_in_)):
        graphs.append(extract_graph(ebm, feature_index, **extract_dict))

    # convert the graphs to text
    to_text_kwargs = list(inspect.signature(graph_to_text).parameters)
    to_text_dict = {k: kwargs[k] for k in dict(kwargs) if k in to_text_kwargs}
    graphs = [graph_to_text(graph, **to_text_dict) for graph in graphs]

    # get a cot sequence of messages to describe the graph
    llm_descripe_kwargs = list(inspect.signature(prompts.describe_graph_cot).parameters)
    llm_descripe_kwargs.extend(
        list(inspect.signature(prompts.describe_graph).parameters)
    )
    llm_descripe_dict = {
        k: kwargs[k]
        for k in dict(kwargs)
        if k in llm_descripe_kwargs and k != "num_sentences"
    }
    messages = [
        prompts.describe_graph_cot(graph, num_sentences=7, **llm_descripe_dict)
        for graph in graphs
    ]

    # execute the prompts
    graph_descriptions = [
        t2ebm.llm.chat_completion(llm, msg)[-1]["content"] for msg in messages
    ]

    # combine the graph descriptions in a single string
    graph_descriptions = "\n\n".join(
        [
            ebm.feature_names_in_[idx] + ": " + graph_description
            for idx, graph_description in enumerate(graph_descriptions)
        ]
    )

    # print(graph_descriptions)

    # now, ask the llm to summarize the different descriptions
    llm_summarize_kwargs = list(inspect.signature(prompts.summarize_ebm).parameters)
    llm_summarize_dict = {
        k: kwargs[k] for k in dict(kwargs) if k in llm_summarize_kwargs
    }
    messages = prompts.summarize_ebm(
        feature_importances,
        graph_descriptions,
        num_sentences=num_sentences,
        **llm_summarize_dict,
    )

    # execute the prompt and return the summary
    return t2ebm.llm.chat_completion(llm, messages)[-1]["content"]