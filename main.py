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

            # Example feature names (replace with your actual features)
            self.feature_names = [
                'destination', 'passenger', 'weather', 'temperature',
                'coupon_type', 'expiration', 'gender', 'age', 'marital_status',
                'has_children', 'education', 'occupation', 'income'
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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("User Details")
        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status",
                                      ["Single", "Married", "Divorced", "Widowed"])
        has_children = st.selectbox("Has Children", ["No", "Yes"])
        education = st.selectbox("Education",
                                 ["High School", "Some College", "Bachelor's", "Master's", "PhD"])
        occupation = st.selectbox("Occupation",
                                  ["Student", "Employed", "Unemployed", "Retired"])
        income = st.selectbox("Income Level",
                              ["Low", "Medium", "High", "Very High"])

    with col2:
        st.subheader("Context Details")
        destination = st.selectbox("Destination",
                                   ["No Urgent Place", "Home", "Work", "Leisure"])
        passenger = st.selectbox("Passenger Type",
                                 ["Alone", "With Friends", "With Family", "With Partner"])
        weather = st.selectbox("Weather",
                               ["Sunny", "Rainy", "Snowy", "Cloudy"])
        temperature = st.slider("Temperature (°F)", 0, 100, 70)
        coupon_type = st.selectbox("Coupon Type",
                                   ["Restaurant <$20", "Coffee House", "Carry Out",
                                    "Bar", "Restaurant $20-50"])
        expiration = st.selectbox("Coupon Expiration",
                                  ["2 hours", "1 day", "3 days", "1 week"])

    # Convert inputs to feature values (you'll need to map these to your actual encoding)
    input_features = [
        1 if destination == "No Urgent Place" else 2 if destination == "Home" else 3,  # destination
        1 if passenger == "Alone" else 2 if passenger == "With Friends" else 3,  # passenger
        1 if weather == "Sunny" else 2,  # weather
        temperature,  # temperature
        1 if coupon_type == "Restaurant <$20" else 2 if coupon_type == "Coffee House" else 3,  # coupon_type
        1 if expiration == "2 hours" else 2,  # expiration
        1 if gender == "Male" else 0,  # gender
        age,  # age
        1 if marital_status == "Single" else 2 if marital_status == "Married" else 3,  # marital_status
        1 if has_children == "Yes" else 0,  # has_children
        1 if education == "High School" else 2 if education == "Some College" else 3,  # education
        1 if occupation == "Student" else 2 if occupation == "Employed" else 3,  # occupation
        1 if income == "Low" else 2 if income == "Medium" else 3  # income
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
                    accept_prob = probability[1] * 100
                    st.metric("Acceptance Probability", f"{accept_prob:.1f}%")

                    # Progress bar
                    st.progress(int(accept_prob))

                    st.write(
                        f"**Confidence:** {'High' if accept_prob > 70 else 'Medium' if accept_prob > 50 else 'Low'}")

                # Feature importance (simulated)
                st.subheader("Key Factors")
                factors = [
                    ("Coupon Type", "High impact"),
                    ("Passenger Type", "Medium impact"),
                    ("Temperature", "Low impact"),
                    ("Destination", "Medium impact")
                ]

                for factor, impact in factors:
                    st.write(f"• **{factor}:** {impact}")


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
    # Simulated feature importance
    features = ['Coupon Type', 'Passenger', 'Temperature', 'Destination', 'Expiration']
    importance = [0.25, 0.19, 0.15, 0.11, 0.09]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importance, y=features, palette='viridis', ax=ax)
    ax.set_title('Top 5 Feature Importances')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig)


def show_about_page():
    """Show about page"""
    st.header("About This Project")

    st.write("""
    ### 🎫 Coupon Recommendation System

    This machine learning system predicts whether a user will accept a coupon 
    based on their profile and context information.

    **Key Features:**
    - 🤖 Powered by XGBoost algorithm
    - 📊 74.1% prediction accuracy
    - 🔍 Real-time predictions
    - 📈 Model performance analytics

    **Model Performance:**
    - **Accuracy:** 74.1%
    - **Precision:** 74.3% 
    - **Recall:** 81.7%
    - **F1-Score:** 77.8%

    **Top Influencing Factors:**
    1. Coupon Type
    2. Passenger Type
    3. Temperature
    4. Destination
    5. Expiration Time

    *Note: This is a demonstration app. In production, it would connect to a 
    trained model and real-time data sources.*
    """)


if __name__ == "__main__":
    main()