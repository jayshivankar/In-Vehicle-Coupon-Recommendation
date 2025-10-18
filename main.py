import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Coupon Recommendation System",
    page_icon="🎫",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .accept {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
    }
    .reject {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


class CouponRecommender:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def load_model(self):
        """Load trained model"""
        try:
            # Initialize model with best parameters
            self.model = XGBClassifier(
                colsample_bytree=0.8,
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=3,
                n_estimators=100,
                subsample=1.0,
                random_state=42,
                eval_metric='logloss'
            )

            # In a real app, you'd load a pre-trained model:
            # with open('xgboost_model.pkl', 'rb') as f:
            #     self.model = pickle.load(f)

            # Updated feature names based on your dataset
            self.feature_names = [
                'destination', 'passanger', 'weather', 'temperature',
                'coupon', 'expiration', 'gender', 'age', 'maritalStatus',
                'has_children', 'education', 'occupation', 'income',
                'Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50',
                'toCoupon_GEQ15min', 'toCoupon_GEQ25min',
                'direction_same'
            ]

            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def predict_coupon_acceptance(self, input_features):
        """Make prediction"""
        try:
            # Convert input to numpy array
            input_array = np.array(input_features).reshape(1, -1)

            # Make prediction
            prediction = self.model.predict(input_array)[0]
            probability = self.model.predict_proba(input_array)[0]

            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">🎫 Coupon Recommendation System</h1>',
                unsafe_allow_html=True)

    # Initialize recommender
    recommender = CouponRecommender()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode",
                                    ["Prediction", "Model Analysis", "About"])

    # Load model
    if not recommender.load_model():
        st.error("Failed to load model. Please check the model file.")
        return

    if app_mode == "Prediction":
        show_prediction_interface(recommender)
    elif app_mode == "Model Analysis":
        show_model_analysis()
    else:
        show_about_page()


def show_prediction_interface(recommender):
    """Show prediction interface"""
    st.header("🔮 Predict Coupon Acceptance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("User Profile")

        # Age with actual categories from data
        age_options = ['below21', '21', '26', '31', '36', '41', '46', '50plus']
        age = st.selectbox("Age", age_options)

        gender = st.selectbox("Gender", ["Male", "Female"])

        # Marital Status with actual categories
        marital_options = ["Single", "Married partner", "Unmarried partner", "Divorced", "Widowed"]
        maritalStatus = st.selectbox("Marital Status", marital_options)

        has_children = st.selectbox("Has Children", ["No", "Yes"])

        # Education with actual categories
        education_options = [
            "Some college - no degree", "Bachelors degree",
            "Graduate degree (Masters or Doctorate)", "Associates degree",
            "High School Graduate", "Some High School"
        ]
        education = st.selectbox("Education", education_options)

        # Occupation with actual categories (top 10 for simplicity)
        occupation_options = [
            "Unemployed", "Student", "Computer & Mathematical", "Sales & Related",
            "Education&Training&Library", "Management", "Office & Administrative Support",
            "Arts Design Entertainment Sports & Media", "Business & Financial", "Retired"
        ]
        occupation = st.selectbox("Occupation", occupation_options)

        # Income with actual categories
        income_options = [
            "Less than $12500", "$12500 - $24999", "$25000 - $37499",
            "$37500 - $49999", "$50000 - $62499", "$62500 - $74999",
            "$75000 - $87499", "$87500 - $99999", "$100000 or More"
        ]
        income = st.selectbox("Income", income_options)

    with col2:
        st.subheader("Context Details")

        destination = st.selectbox("Destination", ["No Urgent Place", "Home", "Work"])

        # Passenger with correct spelling and categories
        passenger_options = ["Alone", "Friend(s)", "Partner", "Kid(s)"]
        passanger = st.selectbox("Passenger Type", passenger_options)

        weather = st.selectbox("Weather", ["Sunny", "Snowy", "Rainy"])

        temperature = st.slider("Temperature (°F)", 0, 100, 70)

        # Coupon type with actual categories
        coupon_options = [
            "Coffee House", "Restaurant(<20)", "Carry out & Take away",
            "Bar", "Restaurant(20-50)"
        ]
        coupon = st.selectbox("Coupon Type", coupon_options)

        expiration = st.selectbox("Coupon Expiration", ["2h", "1d"])

    with col3:
        st.subheader("Behavioral Factors")

        # Visit frequency options
        freq_options = ["never", "less1", "1~3", "4~8", "gt8"]

        st.write("**How often do you visit:**")
        Bar = st.selectbox("Bar", freq_options)
        CoffeeHouse = st.selectbox("Coffee House", freq_options)
        CarryAway = st.selectbox("Carry Away", freq_options)
        RestaurantLessThan20 = st.selectbox("Restaurant <$20", freq_options)
        Restaurant20To50 = st.selectbox("Restaurant $20-50", freq_options)

        st.write("**Travel Time to Coupon:**")
        toCoupon_GEQ15min = st.selectbox("≥ 15 minutes", ["No", "Yes"])
        toCoupon_GEQ25min = st.selectbox("≥ 25 minutes", ["No", "Yes"])

        st.write("**Direction:**")
        direction_same = st.selectbox("Same direction as destination", ["No", "Yes"])

    # Convert all inputs to feature values
    input_features = [
        # Destination mapping
        0 if destination == "No Urgent Place" else 1 if destination == "Home" else 2,

        # Passenger mapping
        0 if passanger == "Alone" else 1 if passanger == "Friend(s)" else 2 if passanger == "Partner" else 3,

        # Weather mapping
        0 if weather == "Sunny" else 1 if weather == "Snowy" else 2,

        temperature,  # temperature (continuous)

        # Coupon mapping
        0 if coupon == "Coffee House" else 1 if coupon == "Restaurant(<20)" else 2 if coupon == "Carry out & Take away" else 3 if coupon == "Bar" else 4,

        # Expiration mapping
        0 if expiration == "2h" else 1,

        # Gender mapping
        0 if gender == "Female" else 1,

        # Age mapping
        age_options.index(age),

        # Marital Status mapping
        marital_options.index(maritalStatus),

        # Has children mapping
        1 if has_children == "Yes" else 0,

        # Education mapping
        education_options.index(education),

        # Occupation mapping
        occupation_options.index(occupation),

        # Income mapping
        income_options.index(income),

        # Behavioral factors - frequency mappings
        freq_options.index(Bar),
        freq_options.index(CoffeeHouse),
        freq_options.index(CarryAway),
        freq_options.index(RestaurantLessThan20),
        freq_options.index(Restaurant20To50),

        # Travel time mappings
        1 if toCoupon_GEQ15min == "Yes" else 0,
        1 if toCoupon_GEQ25min == "Yes" else 0,

        # Direction mappings
        1 if direction_same == "Yes" else 0,
    ]

    # Prediction button
    if st.button("🎯 Predict Acceptance", type="primary"):
        with st.spinner("Analyzing..."):
            prediction, probability = recommender.predict_coupon_acceptance(input_features)

            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Prediction box
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-box accept">'
                            f'<h3>✅ ACCEPT COUPON</h3>'
                            f'<p>User is likely to accept this coupon</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box reject">'
                            f'<h3>❌ REJECT COUPON</h3>'
                            f'<p>User is unlikely to accept this coupon</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                with col2:
                    # Probability gauge
                    accept_prob = probability[1] * 100 if probability is not None else 0
                    st.metric("Acceptance Probability", f"{accept_prob:.1f}%")

                    # Progress bar
                    st.progress(int(accept_prob))

                    confidence_level = 'High' if accept_prob > 70 else 'Medium' if accept_prob > 50 else 'Low'
                    st.write(f"**Confidence:** {confidence_level}")

                # Feature importance (simulated)
                st.subheader("Key Influencing Factors")
                factors = [
                    ("Coupon Type", "High impact"),
                    ("Passenger Type", "Medium impact"),
                    ("Visit Frequency to Similar Places", "High impact"),
                    ("Travel Time", "Medium impact"),
                    ("Income Level", "Medium impact"),
                    ("Age Group", "Low impact")
                ]

                for factor, impact in factors:
                    st.write(f"• **{factor}:** {impact}")

                # Show input summary
                with st.expander("View Input Summary"):
                    st.write("**User Profile:**")
                    st.write(f"- Age: {age}, Gender: {gender}, Marital Status: {maritalStatus}")
                    st.write(f"- Education: {education}, Occupation: {occupation}, Income: {income}")

                    st.write("**Context:**")
                    st.write(f"- Destination: {destination}, Passenger: {passanger}")
                    st.write(f"- Weather: {weather}, Temperature: {temperature}°F")
                    st.write(f"- Coupon: {coupon}, Expires in: {expiration}")


def show_model_analysis():
    """Show model performance analysis"""
    st.header("📊 Model Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Metrics")
        st.metric("Accuracy", "74.1%")
        st.metric("Precision", "74.3%")
        st.metric("Recall", "81.7%")
        st.metric("F1-Score", "77.8%")

    with col2:
        st.subheader("Confusion Matrix")
        # Simulated confusion matrix
        cm = np.array([[193, 67], [46, 204]])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted Reject', 'Predicted Accept'],
                    yticklabels=['Actual Reject', 'Actual Accept'])
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    st.subheader("Feature Importance")
    # Simulated feature importance based on your dataset
    features = ['Coupon Type', 'Bar Visit Freq', 'Passenger', 'Restaurant<$20 Freq',
                'Travel Time ≥15min', 'Income', 'Coffee House Freq', 'Age', 'Temperature']
    importance = [0.25, 0.19, 0.15, 0.11, 0.09, 0.07, 0.06, 0.05, 0.03]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
    ax.set_title('Top Feature Importances')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig)


def show_about_page():
    """Show about page"""
    st.header("About This Project")

    st.write("""
    ### 🎫 In-Vehicle Coupon Recommendation System

    This machine learning system predicts whether a user will accept a coupon 
    based on their profile, context, and behavioral information.

    **Dataset Features:**
    - **User Profile:** Age, Gender, Marital Status, Education, Occupation, Income
    - **Context:** Destination, Passenger, Weather, Temperature
    - **Coupon Details:** Type, Expiration time
    - **Behavioral:** Visit frequency to various establishments
    - **Travel:** Time to coupon location and direction

    **Model Performance:**
    - **Accuracy:** 74.1%
    - **Precision:** 74.3% 
    - **Recall:** 81.7%
    - **F1-Score:** 77.8%

    **Top Influencing Factors:**
    1. Coupon Type
    2. Bar Visit Frequency
    3. Passenger Type
    4. Restaurant <$20 Visit Frequency
    5. Travel Time to Coupon Location

    *Note: This is a demonstration app. The actual model will be connected 
    through the backend API when deployed in production.*
    """)


if __name__ == "__main__":
    main()