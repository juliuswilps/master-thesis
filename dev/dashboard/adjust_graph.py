import api_helpers

def adjust_graph(feature_data: dict):
    llm = api_helpers.setup()

    graph_dict = dict(zip(feature_data["x_vals"], feature_data["y_vals"][feature_data["current_iteration"]]))

    #system_msg = helpers.get_system_msg(graph_dict)
    messages = [
        {
            "role": "system",
            "content": api_helpers.get_system_msg(graph_dict)
        },
        {
            "role": "user",
            "content": f"Here is the shape function for the feature {feature_data['feature_name']}"
                       f"{graph_dict}."
        }
    ]

    completion = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    response = completion.choices[0].message.content

    print(response)


data = {
    'x_vals': [0.25, 1.0, 2.0, 3.0, 3.7755275347400925, 4.301055069480185, 5.0255275347400925, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.75],
    'y_vals': [[0.0437886000866321, 0.0950148801030437, 0.11993452670786084, 0.09587951914623816, 0.05198270196535033, -0.0185413782835275, -0.0455523678622698, -0.0988308217763614, -0.11672959293929981, -0.23257763260353448, -0.2796777718819023, -0.3130135433384958, -0.3613624434271742, -0.4322217201709816, -0.4973927641625033, -0.4383435052103781, -0.410719230742905, -0.5252174092979337]],
    'adjusted_y_vals': [],
    'adjusted_visible': False,
    'explanation': 'Graph for Number of Active Credit Cards/Lines',
    'feature_type': 'continuous',
    'feature_name': 'Number of Active Credit Cards/Lines',
    'current_iteration': 0
}

adjust_graph(data)