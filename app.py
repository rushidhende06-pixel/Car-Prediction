
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = joblib.load("gradient_boosting_model.joblib")
    pipeline = joblib.load("preprocessing_pipeline.joblib")
    return model, pipeline

try:
    model, pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


st.title("⚙️ Machine Failure Prediction")
st.write(
    "Enter machine operating parameters to predict "
    "whether the machine is likely to fail."
)

st.divider()


with st.form("prediction_form"):

    st.subheader("Machine Parameters")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=300.0
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=310.0
        )

    with col2:
        rot_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=0,
            max_value=10000,
            value=1500
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=200.0,
            value=40.0
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=500,
            value=100
        )

    submit = st.form_submit_button(
        "Predict Machine Failure",
        use_container_width=True
    )


if submit:

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    try:
        transformed_data = pipeline.transform(input_data)
        prediction = model.predict(transformed_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Machine Failure Predicted")
            st.warning(
                "The model predicts a potential machine failure. "
                "Please inspect the machine."
            )
        else:
            st.success("✅ No Machine Failure Predicted")
            st.info(
                "The model predicts that the machine is operating "
                "without a predicted failure."
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
