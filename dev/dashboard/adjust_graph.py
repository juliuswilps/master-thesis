import api_helpers


def adjust_graph(feature_data: dict, model: str = "gpt-4o-mini"):
    llm = api_helpers.setup()

    # Set domain and prediction target
    domain = "financial risk modeling"
    prediction_target = "predict whether a loan will be paid off (1) or default (0)"

    # First API call: analyze the graph and adjust the y-values
    messages = [
        {
            "role": "system",
            "content": api_helpers.get_system_msg(feature_data, domain, prediction_target)
        },
        {
            "role": "user",
            "content": api_helpers.get_analyze_graph_prompt(feature_data, domain, prediction_target)
        },
        {
            "role": "user",  # Adjust graph prompt
            "content": api_helpers.get_adjust_graph_prompt(feature_data)
        }
    ]

    # Send the request for the adjusted graph
    completion = llm.chat.completions.create(
        model=model,
        messages=messages,
    )

    adjusted_graph_response = completion.choices[0].message.content
    #print(f" adjusted_graph_response: {adjusted_graph_response} (type: {type(adjusted_graph_response)}")
    adjusted_y_vals = api_helpers.parse_adjusted_y_vals(adjusted_graph_response, feature_data)


    messages.append({
        "role": "system",
        "content": api_helpers.get_explanation_system_msg(domain)
    })
    messages.append({
        "role": "user",
        "content": api_helpers.get_explanation_prompt(feature_data, domain)
    })

    # Call the API for the explanation of adjustments
    completion = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    explanation_response = completion.choices[0].message.content
    #print(f"explanation response: {explanation_response}")

    # Extract the explanation
    explanation = explanation_response.strip()

    #print(f"explanation: {explanation}")  # or store it as needed

    return adjusted_y_vals, explanation


