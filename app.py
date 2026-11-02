            step=1
        )

    with col2:
        fuel_type = st.selectbox("Fuel type", fuel_options)
        engine_volume = st.number_input(
            "Engine volume (litres)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.1
        )
        turbo = st.checkbox("Turbo engine")
        mileage = st.number_input(
            "Mileage (km)",
            min_value=1,
            max_value=999999,
            value=100000,
            step=1000
        )

    with col3:
        leather = st.selectbox("Leather interior", leather_options)
        cylinders = st.number_input(
            "Cylinders",
            min_value=min_cyl,
            max_value=max_cyl,
            value=min(max_cyl, 4),
            step=1
        )
        airbags = st.number_input(
            "Airbags",
            min_value=min_airbags,
            max_value=max_airbags,
            value=min(max_airbags, 6),
            step=1
        )
        color = st.selectbox("Color", colors)

    levy = st.number_input(
        "Levy",
        min_value=0.0,
        max_value=100000.0,
        value=float(data["Levy"].median()),
        step=10.0,
        help="Enter the levy amount. Use 0 if it is not applicable."
    )

    submitted = st.form_submit_button(
        "Predict car price",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "Manufacturer": str(manufacturer),
        "Model": str(car_model),
        "Category": str(category),
        "Leather interior": str(leather),
        "Fuel type": str(fuel_type),
        "Color": str(color),
        "Levy": float(levy) if levy > 0 else np.nan,
        "Prod. year": int(production_year),
        "Mileage": float(mileage),
        "Cylinders": float(cylinders),
        "Airbags": float(airbags),
        "Engine": float(engine_volume),
        "Turbo": int(turbo),
        "Age": 2020 - int(production_year)
    }])[MODEL_FEATURES]

    try:
        prediction = float(model.predict(input_data)[0])
        st.divider()
        st.subheader("Estimated car price")
        st.metric(
            label="Predicted price (dataset currency)",
            value=f"{prediction:,.0f}"
        )
        st.caption(
            "This is an estimate from a machine learning model, not a "
            "guaranteed market or resale price."
        )
    except Exception as error:
        st.error("Prediction failed. Please check the entered car details.")
        st.exception(error)

with st.expander("About this project"):
    st.write(
        "This app uses a Random Forest Regression model with a preprocessing "
        "pipeline. Categorical inputs are one-hot encoded, missing numeric "
        "values are imputed, and numeric features are scaled."
    )
    st.write(f"Training rows used: {len(data):,}")
    st.write("Target variable: Price")
    st.write("Model: RandomForestRegressor")
