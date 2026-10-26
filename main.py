
import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessor
model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

st.set_page_config(
    page_title="Travel Package Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Travel Package Purchase Prediction")
st.write("Enter customer details to predict travel package purchase.")

st.divider()

with st.form("prediction_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 35)
        contact = st.selectbox(
            "Type of Contact",
            ["Self Enquiry", "Company Invited"]
        )
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Free Lancer",
             "Small Business", "Large Business"]
        )
        gender = st.selectbox(
            "Gender", ["Male", "Female", "Fe Male"]
        )

    with col2:
        pitch_duration = st.number_input(
            "Duration of Pitch", min_value=0.0, value=10.0
        )
        followups = st.number_input(
            "Number of Follow-ups", min_value=0, value=4
        )
        product = st.selectbox(
            "Product Pitched",
            ["Basic", "Standard", "Deluxe",
             "Super Deluxe", "King"]
        )
        property_star = st.selectbox(
            "Preferred Property Star", [3, 4, 5]
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Single", "Unmarried", "Divorced"]
        )

    with col3:
        trips = st.number_input(
            "Number of Trips", min_value=0, value=3
        )
        passport = st.selectbox("Has Passport?", ["Yes", "No"])
        satisfaction = st.selectbox(
            "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
        )
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        income = st.number_input(
            "Monthly Income", min_value=0.0, value=25000.0
        )

    st.subheader("Travel Details")

    col4, col5 = st.columns(2)

    with col4:
        people = st.number_input(
            "Number of People Visiting",
            min_value=1, value=2
        )

    with col5:
        children = st.number_input(
            "Number of Children Visiting",
            min_value=0, value=0
        )

    predict = st.form_submit_button(
        "🔍 Predict Purchase",
        use_container_width=True
    )

if predict:

    customer = pd.DataFrame([{
        "Age": float(age),
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": float(pitch_duration),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfFollowups": float(followups),
        "ProductPitched": product,
        "PreferredPropertyStar": float(property_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(trips),
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "Designation": designation,
        "MonthlyIncome": float(income),
        "TotalVisiting": people + children
    }])

    try:
        transformed = preprocessor.transform(customer)
        prediction = model.predict(transformed)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "Customer is likely to purchase the travel package! ✅"
            )
        else:
            st.info(
                "Customer is not likely to purchase the travel package."
            )

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(transformed)[0][1]
            st.metric(
                "Estimated Purchase Probability",
                f"{probability * 100:.2f}%"
            )
            st.progress(float(probability))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
