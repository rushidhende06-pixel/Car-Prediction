
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# 1. Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------
# 2. Load Saved ML Model
# ---------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("car_price_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    return model, preprocessor


try:
    model, preprocessor = load_model()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()


# Get the exact columns used during preprocessing
if hasattr(preprocessor, "feature_names_in_"):
    expected_columns = list(preprocessor.feature_names_in_)
else:
    st.error(
        "The preprocessor does not expose feature_names_in_. "
        "Please check how it was trained."
    )
    st.stop()


# ---------------------------------
# 3. Application Header
# ---------------------------------
st.title("🚗 Car Price Prediction")
st.write(
    "Enter the details of a used car to estimate its selling price "
    "using a Machine Learning model."
)

st.divider()


# ---------------------------------
# 4. Input Field Configuration
# ---------------------------------
numeric_features = {
    "Levy": (0.0, 0.0, 100000.0, 100.0),
    "Prod. year": (2015, 1980, 2026, 1),
    "Mileage": (50000.0, 0.0, 1000000.0, 1000.0),
    "Cylinders": (4.0, 1.0, 16.0, 1.0),
    "Airbags": (4, 0, 20, 1),
    "Engine": (2.0, 0.1, 10.0, 0.1),
    "Age": (5, 0, 60, 1),
    "Turbo": (0, 0, 1, 1)
}

categorical_options = {
    "Leather interior": ["Yes", "No"],
    "Fuel type": [
        "Petrol", "Diesel", "Hybrid", "LPG",
        "CNG", "Plug-in Hybrid", "Hydrogen"
    ],
    "Category": [
        "Sedan", "Jeep", "Hatchback", "Universal",
        "Coupe", "Minivan", "Goods wagon", "Pickup"
    ],
    "Gear box type": [
        "Automatic", "Manual", "Tiptronic", "Variator"
    ],
    "Drive wheels": ["Front", "Rear", "4x4"],
    "Wheel": ["Left wheel", "Right-hand drive"]
}


def create_input(feature):
    """Create an input widget based on the expected feature."""

    if feature in numeric_features:
        default, minimum, maximum, step = numeric_features[feature]

        return st.number_input(
            feature,
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=step
        )

    if feature in categorical_options:
        return st.selectbox(
            feature,
            categorical_options[feature]
        )

    if feature == "Manufacturer":
        return st.text_input(feature, value="Toyota")

    if feature == "Model":
        return st.text_input(feature, value="Corolla")

    if feature == "Color":
        return st.text_input(feature, value="Black")

    if feature == "Engine volume":
        return st.text_input(feature, value="2.0 Turbo")

    if feature == "Doors":
        return st.text_input(feature, value="04-May")

    return st.text_input(feature, value="")


# ---------------------------------
# 5. Car Details Form
# ---------------------------------
st.subheader("Enter Car Details")

with st.form("car_prediction_form"):

    col1, col2 = st.columns(2)

    input_values = {}

    for index, feature in enumerate(expected_columns):
        with col1 if index % 2 == 0 else col2:
            input_values[feature] = create_input(feature)

    submitted = st.form_submit_button(
        "Predict Car Price 🚘",
        use_container_width=True
    )


# ---------------------------------
# 6. Prediction
# ---------------------------------
if submitted:

    input_data = pd.DataFrame(
        [input_values],
        columns=expected_columns
    )

    try:
        transformed_data = preprocessor.transform(input_data)

        prediction = model.predict(transformed_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        st.success("Car price predicted successfully!")

        st.metric(
            label="Estimated Car Price",
            value=f"₹ {prediction:,.2f}"
        )

        st.caption(
            "This is a machine learning estimate, not a guaranteed "
            "market selling price."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
