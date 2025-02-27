import streamlit as st
import dashboard_helpers as helpers
import os

st.set_page_config(
    page_title="Shape Function Dashboard",
    layout="wide",
)


ebm_path = "/mount/src/master-thesis/dev/dashboard/final-ebm-model.pkl"
description_path = "/mount/src/master-thesis/dev/dashboard/heloc-description.json"
test_data_path = "/mount/src/master-thesis/dev/dashboard/heloc-test.csv"

# Initialize session state for 'ebm_data'
if "ebm_data" not in st.session_state:
    st.session_state.ebm, st.session_state.ebm_data = helpers.load_ebm_data(ebm_path, description_path)
    st.session_state.adjusted_visible = False
ebm = st.session_state.ebm
ebm_data = st.session_state.ebm_data
#adjusted_visible = st.session_state.adjusted_visible

# Title
# st.title("Feature Adjustment Dashboard")

# Dropdown menu for feature selection
selected_feature = st.selectbox("Select Factor", list(ebm_data.keys()), disabled=st.session_state.adjusted_visible)
feature_data = ebm_data[selected_feature]
#print(feature_data)

#description = helpers.load_description(feature_data, description_path)
st.text_area(
    "Factor Description",
    feature_data["feature_description"],
    height=75,
    disabled=True,
)

# Display accuracy
#st.subheader("AI Model Prediction Accuracy")
col1, col2 = st.columns(2)
with col1:
    current_ebm = helpers.update_term_scores(ebm, ebm_data, selected_feature)
    original_model_accuracy = helpers.calculate_model_accuracy(current_ebm, test_data_path)
    st.metric(label="AI Model Prediction Accuracy", value=f"{original_model_accuracy:.2%}")
with col2:
    #if feature_data["adjusted_visible"]:
    if st.session_state.adjusted_visible:
        adjusted_ebm = helpers.update_term_scores(ebm, ebm_data, selected_feature, adjusted=True)
        adjusted_model_accuracy = helpers.calculate_model_accuracy(adjusted_ebm, test_data_path)
        st.metric(label="Accuracy after Adjustment", value=f"{adjusted_model_accuracy:.2%}")

# Plot the shape function
col1, col2 = st.columns([3, 1])
with col1:
    fig = helpers.create_shape_function_plot(feature_data, st.session_state)
    st.plotly_chart(fig, use_container_width=True)

# Explanation text box
with col2:
    explanation_text = feature_data["explanation"] if st.session_state.adjusted_visible else "This is where explanations for generated adjusted graphs will appear."
    st.text_area(
        "Explanation for Suggested Adjustment",
        explanation_text,
        height=200,
        disabled=True,
    )

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("⬅️ Previous"):
        helpers.previous_iteration(ebm_data, selected_feature)

with col3:
    if st.button("➡️ Next"):
        helpers.next_iteration(ebm_data, selected_feature)

st.write(f"History: {feature_data['current_iteration'] + 1} / {len(feature_data['y_vals'])}")

# Buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.session_state.adjusted_visible and st.button("✅ Keep Adjustment"):
        helpers.keep_changes(ebm_data, selected_feature, st.session_state)
        st.rerun()
with col3:
    if st.session_state.adjusted_visible and st.button("❌ Discard Adjustment"):
        helpers.discard_changes(ebm_data, selected_feature, st.session_state)
        st.rerun()
with col2:
    if not st.session_state.adjusted_visible and st.button("🛠️ Generate Adjusted Curve"):
        helpers.generate_adjusted_graph(ebm_data, selected_feature, st.session_state, reasoning=True, simplify=True)
        st.rerun()

#with col4:
#    if st.button("💾 Save Updated AI Model"):
#        helpers.save_adjusted_model(ebm, ebm_data, "../updated_ebm.pkl")
#        st.success("Model saved to ../updated_ebm.pkl")

