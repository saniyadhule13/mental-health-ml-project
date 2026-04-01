import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Mental Health Prediction", page_icon="🧠", layout="wide")

BUNDLE_PATH = Path(__file__).parent / "mental_health_bundle.pkl"


@st.cache_resource
def load_bundle():
    with open(BUNDLE_PATH, "rb") as file:
        return pickle.load(file)


bundle = load_bundle()
models = bundle["models"]
scaler = bundle["scaler"]
label_encoders = bundle["label_encoders"]
input_columns = bundle["input_columns"]
numeric_columns = bundle["numeric_columns"]
category_options = bundle["category_options"]
numeric_metadata = bundle["numeric_metadata"]
accuracies = bundle["accuracies"]
target_labels = bundle["target_labels"]

st.title("🧠 Student Mental Health Prediction")
st.write("This app predicts whether a student may need mental health support based on the training dataset.")
st.caption("Educational use only. This is not medical advice.")
st.markdown("---")

st.subheader("Enter Student Details")
user_input = {}
left_col, right_col = st.columns(2)

for index, column in enumerate(input_columns):
    current_col = left_col if index % 2 == 0 else right_col
    with current_col:
        if column in category_options:
            options = category_options[column]
            user_input[column] = st.selectbox(column, options)
        else:
            meta = numeric_metadata[column]
            user_input[column] = st.number_input(
                column,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(meta["default"]),
                step=float(meta["step"]),
            )

st.markdown("---")
model_choice = st.selectbox("Choose Model", list(models.keys()))


def preprocess_input(raw_input: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([raw_input], columns=input_columns)

    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column].astype(str))

    if numeric_columns:
        input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    return input_df


if st.button("Predict", use_container_width=True):
    processed_input = preprocess_input(user_input)
    selected_model = models[model_choice]

    prediction_code = selected_model.predict(processed_input)[0]
    prediction_label = label_encoders[bundle["target_column"]].inverse_transform([prediction_code])[0]

    probability_block = None
    if hasattr(selected_model, "predict_proba"):
        probability_block = selected_model.predict_proba(processed_input)[0]

    st.subheader("Prediction Result")
    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        if str(prediction_label).lower() == "yes":
            st.error("⚠️ Depression Likely")
        else:
            st.success("✅ No Depression Detected")

    with result_col2:
        if probability_block is not None and len(probability_block) == len(target_labels):
            class_probabilities = dict(zip(target_labels, probability_block))
            yes_probability = class_probabilities.get("yes", 0.0) * 100
            st.metric("Depression %", f"{yes_probability:.2f}%")
        else:
            st.metric("Depression %", "N/A")

    with result_col3:
        if probability_block is not None and len(probability_block) == len(target_labels):
            class_probabilities = dict(zip(target_labels, probability_block))
            no_probability = class_probabilities.get("no", 0.0) * 100
            st.metric("No Depression %", f"{no_probability:.2f}%")
        else:
            st.metric("No Depression %", "N/A")

    st.markdown("---")
    st.subheader("Top Features Behind This Prediction")

    if model_choice == "Random Forest":
        importances = selected_model.feature_importances_
    elif model_choice == "Decision Tree":
        importances = selected_model.feature_importances_
    else:
        importances = np.abs(selected_model.coef_[0])

    importance_df = pd.DataFrame(
        {
            "Feature": input_columns,
            "Importance": importances,
        }
    ).sort_values("Importance", ascending=False).head(5)

    for _, row in importance_df.iterrows():
        feature = row["Feature"]
        value = user_input[feature]
        impact = row["Importance"] * 100
        st.info(f"**{feature}** = {value} → impact: {impact:.1f}%")

    st.markdown("---")
    st.caption(f"Training rows used: {bundle['training_rows']}")
    st.caption(f"Selected model test accuracy: {accuracies[model_choice] * 100:.1f}%")
