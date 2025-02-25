import api_helpers
import data


def adjust_graph(feature_data: dict, model: str = "gpt-4o"):
    client = api_helpers.setup()

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
    completion = client.chat.completions.create(
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
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    explanation_response = completion.choices[0].message.content
    #print(f"explanation response: {explanation_response}")

    # Extract the explanation
    explanation = explanation_response.strip()

    #print(f"explanation: {explanation}")  # or store it as needed

    return adjusted_y_vals, explanation

def adjust_graph_reasoning(feature_data: dict, model: str = "o1-mini"):
    client = api_helpers.setup()

    # Set domain and prediction target
    domain = "financial risk modeling"
    prediction_target = "predict whether a loan will be paid off (1) or default (0)"

    # Send the request for the adjusted graph
    completion = client.chat.completions.create(
        model=model,
        #reasoning_effort="medium",
        messages= [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": api_helpers.get_reasoning_prompt(feature_data, domain, prediction_target)
                    },
                ],
            }
        ]
    )

    response = completion.choices[0].message.content

    adjusted_y_vals, explanation = api_helpers.parse_response_reasoning(response)

    expected_length = len(feature_data["x_vals"])
    actual_length = len(adjusted_y_vals)

    # Adjust the length to match the expected size
    if actual_length < expected_length:
        print(
            f"[Warning] Adjusted y-values list is too short. Expected {expected_length}, got {actual_length}. Adding {expected_length - actual_length} values.")
        last_value = adjusted_y_vals[-1] if adjusted_y_vals else 0  # Default to 0 if empty
        adjusted_y_vals.extend([last_value] * (expected_length - actual_length))
    elif actual_length > expected_length:
        print(
            f"[Warning] Adjusted y-values list is too long. Expected {expected_length}, got {actual_length}. Removing {actual_length - expected_length} values.")
        adjusted_y_vals = adjusted_y_vals[:expected_length]

    return adjusted_y_vals, explanation

#y, x = adjust_graph_reasoning(data.data_score)

#print(f"adjusted y: {y}")
#print(f"explanation: {x}")
