import streamlit as st
import dashboard_helpers as help

# TODO Graph mit D3.js und streamlit-d3 interaktiv machen

st.set_page_config(
    page_title="Shape Function Dashboard",
    layout="wide",
)

# Initialize session state for 'ebm_data'
if "ebm_data" not in st.session_state:
    st.session_state.ebm, st.session_state.ebm_data = help.load_ebm_data("../trained_ebm.pkl")
ebm = st.session_state.ebm
ebm_data = st.session_state.ebm_data

# Title
st.title("Feature Adjustment Dashboard")

# Dropdown menu for feature selection
selected_feature = st.selectbox("Select Feature", list(ebm_data.keys()))
feature_data = ebm_data[selected_feature]

# Display accuracy
st.subheader("Model Accuracy")
col1, col2 = st.columns(2)
with col1:
    original_model_accuracy = help.calculate_model_accuracy(ebm, "test_dataset.csv")
    st.metric(label="Original Model Accuracy", value=f"{original_model_accuracy:.2%}")
with col2:
    if feature_data["adjusted_visible"]:
        adjusted_ebm = help.update_term_scores(ebm, feature_data)
        adjusted_model_accuracy = help.calculate_model_accuracy(adjusted_ebm, "test_dataset.csv")
        st.metric(label="Adjusted Model Accuracy", value=f"{adjusted_model_accuracy:.2%}")

# Plot the shape function
col1, col2 = st.columns([3, 1])
with col1:
    fig = help.create_shape_function_plot(feature_data)
    st.plotly_chart(fig, use_container_width=True)

# Explanation text box
with col2:
    explanation_text = feature_data["explanation"] if feature_data["adjusted_visible"] else "This is where explanations for generated adjusted graphs will appear."
    st.text_area(
        "Explanation",
        explanation_text,
        height=200,
        disabled=True,
    )

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("⬅️ Previous Iteration"):
        help.previous_iteration(ebm_data, selected_feature)

with col3:
    if st.button("➡️ Next Iteration"):
        help.next_iteration(ebm_data, selected_feature)

st.write(f"Iteration: {feature_data['current_iteration'] + 1} / {len(feature_data['history'])}")

# Buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if feature_data["adjusted_visible"] and st.button("✅ Keep Changes"):
        help.keep_changes(ebm_data, selected_feature)
        st.rerun()
with col3:
    if feature_data["adjusted_visible"] and st.button("❌ Discard Changes"):
        help.discard_changes(ebm_data, selected_feature)
        st.rerun()
with col2:
    if not feature_data["adjusted_visible"] and st.button("🛠️ Generate Adjusted Shape Function"):
        help.generate_adjusted_graph(selected_feature, feature_data["feature_type"], feature_data["x_vals"], st.session_state)
        st.rerun()

with col4:
    if st.button("💾 Save Updated Model"):
        help.save_adjusted_model(ebm, ebm_data, "../updated_ebm.pkl")
        st.success("Model saved to ../updated_ebm.pkl")

