import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from xgboost import XGBClassifier

# page config
st.set_page_config(
    page_title="CouponCraft",
    page_icon="🎫",
    layout='wide',
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style type="text/css">
    blockquote {
        margin: 1em 0px 1em -1px;
        padding: 0px 0px 0px 1.2em;
        font-size: 20px;
        border-left: 5px solid rgb(230, 234, 241);
    }
    blockquote p {
        font-size: 30px;
        color: #FFFFFF;
    }
    [data-testid=stSidebar] {
        background-color: rgb(129, 164, 182);
        color: #FFFFFF;
    }
    [aria-selected="true"] {
         color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Defining the functions for each page
def home():
    st.subheader('Welcome to :blue[CouponCraft:] Your Personalized Coupon Recommendation System!')
    st.write(
        "Discover the power of data-driven coupon recommendations. Our advanced machine learning system analyzes customer behavior and contextual factors to provide personalized coupon offers that maximize acceptance rates and customer engagement.")

    st.write(
        "Traditional coupon strategies often miss the mark, leading to wasted marketing efforts and customer dissatisfaction. By leveraging XGBoost machine learning, our system identifies the perfect coupon for each customer profile, ensuring higher conversion rates and improved customer loyalty.")

    st.subheader(':blue[Key Features]')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(':orange[Data Insights Explorer:]')
        st.write(
            'Dive deep into comprehensive data analysis with interactive visualizations. Understand customer patterns, coupon preferences, and behavioral trends that drive coupon acceptance decisions.')

    with col2:
        st.subheader(':green[Smart Prediction Engine:]')
        st.write(
            "Utilize our trained XGBoost model to predict coupon acceptance with 74% accuracy. Get real-time predictions based on customer demographics, context, and historical behavior.")

    st.write("Sincerely,")
    st.subheader("Your Name Here")  # Replace with your name
    linkedin_url = "https://www.linkedin.com/in/your-profile/"  # Replace with your LinkedIn
    github_url = "https://github.com/your-profile"  # Replace with your GitHub

    # Add links to your LinkedIn and GitHub profiles
    st.write(f"LinkedIn: [My LinkedIn Profile]({linkedin_url})", f"GitHub: [My GitHub Profile]({github_url})")


# ------------------------------------------------------------------------------------------------------------ #

def eda():
    data = pd.read_csv('in-vehicle-coupon-recommendation.csv')
    st.subheader('Exploratory Data Analysis: Understanding Coupon Acceptance Patterns')
    st.write(
        'This section provides comprehensive insights into the factors influencing coupon acceptance. Through detailed visualizations and statistical analysis, we uncover the key drivers behind customer decisions to accept or reject coupons.')

    # Data preprocessing (same as your colab)
    data = data.drop(['car', 'toCoupon_GEQ5min', 'direction_opp'], axis=1)

    # Mode imputation for missing values
    data['Bar'] = data['Bar'].fillna(data['Bar'].value_counts().index[0])
    data['CoffeeHouse'] = data['CoffeeHouse'].fillna(data['CoffeeHouse'].value_counts().index[0])
    data['CarryAway'] = data['CarryAway'].fillna(data['CarryAway'].value_counts().index[0])
    data['RestaurantLessThan20'] = data['RestaurantLessThan20'].fillna(
        data['RestaurantLessThan20'].value_counts().index[0])
    data['Restaurant20To50'] = data['Restaurant20To50'].fillna(data['Restaurant20To50'].value_counts().index[0])

    # Remove duplicates
    data = data.drop_duplicates()

    # Data overview
    with st.expander("**Dataset Overview and Missing Values Analysis**"):
        missing_percentage = np.round(data.isnull().mean() * 100, 2)
        missing_count = data.isnull().sum()
        data_types = data.dtypes
        data_show = data.head(2).T
        data_show.columns = ['Sample 1', 'Sample 2']
        detail_report = pd.DataFrame(
            {'Data Type': data_types, 'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
        detail_report = pd.concat([data_show, detail_report], axis=1)
        st.dataframe(detail_report)

    st.divider()

    # Target distribution
    st.subheader('Target Variable Distribution')
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        data['Accept(Y/N?)'].value_counts().plot(kind='pie', autopct='%0.2f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Coupon Acceptance Distribution')
        st.pyplot(fig)

    with col2:
        st.write("**Key Observation:**")
        st.write(
            "The dataset shows a relatively balanced distribution between accepted and rejected coupons, which is ideal for training machine learning models without significant class imbalance issues.")

    st.divider()

    # Numerical features distribution
    st.subheader('Numerical Features Distribution by Coupon Acceptance')

    numerical_features = [col for col in data.columns if data[col].dtype != 'O' and col != 'Accept(Y/N?)']

    if numerical_features:
        # Create subplots for numerical features
        num_plots = len(numerical_features)
        cols = 3
        rows = (num_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, feature in enumerate(numerical_features):
            if i < len(axes):
                sns.kdeplot(data, x=feature, hue='Accept(Y/N?)',
                            fill=True, palette=["#8000ff", "#da8829"],
                            alpha=0.5, ax=axes[i])
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')
                axes[i].set_title(f'{feature} Distribution')
                axes[i].grid(color='#000000', ls=':', axis='y', dashes=(1, 5))

        # Hide empty subplots
        for i in range(len(numerical_features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        st.write("**Observation on Quantitative Data Distribution:**")
        st.write(
            "For the quantitative variables, the distribution of rejection follows a similar pattern to acceptance but with varying densities across different feature ranges.")

    st.divider()

    # Categorical features analysis
    st.subheader('Categorical Features Analysis')

    categorical_features = [col for col in data.columns if
                            data[col].dtype == 'O' and col not in ['occupation', 'coupon']]

    # Display key categorical features in columns
    num_features = min(6, len(categorical_features))
    cols = st.columns(3)

    for i, feature in enumerate(categorical_features[:num_features]):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(8, 5))
            crosstab = pd.crosstab(data[feature], data['Accept(Y/N?)'])
            crosstab.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
            ax.set_title(f'{feature} vs Acceptance')
            ax.legend(['Rejected', 'Accepted'])
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.divider()

    # Occupation analysis
    st.subheader('Occupation-wise Coupon Acceptance')
    crosstab_df = pd.crosstab(data['occupation'], data['Accept(Y/N?)'])

    fig, ax = plt.subplots(figsize=(12, 6))
    crosstab_df.plot(kind='bar', stacked=True, ax=ax, color=['#FF6B6B', '#4ECDC4'])
    ax.set_title('Occupation-wise Coupon Acceptance')
    ax.set_xlabel('Occupation')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    ax.legend(['Rejected', 'Accepted'])
    st.pyplot(fig)

    st.divider()

    # Coupon type analysis for accepted coupons only
    st.subheader('Coupon Type Analysis (Accepted Coupons Only)')
    data_accepted = data[data['Accept(Y/N?)'] == 1]

    analysis_features = ['destination', 'education', 'expiration', 'passanger']

    cols = st.columns(2)
    for i, feature in enumerate(analysis_features):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(10, 5))
            crosstab_df = pd.crosstab(data_accepted['coupon'], data_accepted[feature])
            crosstab_df.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'Coupon Type by {feature}')
            ax.set_xlabel('Coupon Type')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.write("**Key Insights from Categorical Analysis:**")
    st.write("""
    1. Customers with 'No Urgent Place' as destination show higher coupon acceptance
    2. Solo travelers demonstrate higher acceptance rates
    3. Sunny weather conditions correlate with increased coupon acceptance
    4. Coffee House and Restaurant(<20) coupons are most frequently accepted
    5. 1-day expiration coupons are preferred over 2-hour coupons
    6. Younger age groups (21-26) show highest engagement
    7. Middle-income brackets ($25K-$50K) are most responsive to coupons
    """)


# -------------------------------------------------------------------------------------------------------------- #
def prediction():
    st.subheader('Coupon Acceptance Prediction')
    st.write(
        "Use our trained XGBoost model to predict whether a customer will accept a coupon based on their profile and context.")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        destination_display = ('No Urgent Place', 'Home', 'Work')
        destination_options = list(range(len(destination_display)))
        destination = st.selectbox('Destination', destination_options, format_func=lambda x: destination_display[x])

        passanger_display = ('Alone', 'Friend(s)', 'Partner', 'Kid(s)')
        passanger_options = list(range(len(passanger_display)))
        passanger = st.selectbox('Passenger', passanger_options, format_func=lambda x: passanger_display[x])

        weather_display = ('Sunny', 'Snowy', 'Rainy')
        weather_options = list(range(len(weather_display)))
        weather = st.selectbox('Weather', weather_options, format_func=lambda x: weather_display[x])

        temperature_display = ('80', '55', '30')
        temperature_options = list(range(len(temperature_display)))
        temperature = st.selectbox('Temperature', temperature_options, format_func=lambda x: temperature_display[x])

    with col2:
        time_display = ('2PM', '10AM', '6PM', '7AM', '10PM')
        time_options = list(range(len(time_display)))
        time = st.selectbox('Time', time_options, format_func=lambda x: time_display[x])

        coupon_display = ('Restaurant(<20)', 'Coffee House', 'Carry out & Take away', 'Bar', 'Restaurant(20-50)')
        coupon_options = list(range(len(coupon_display)))
        coupon = st.selectbox('Coupon Type', coupon_options, format_func=lambda x: coupon_display[x])

        expiration_display = ('1d', '2h')
        expiration_options = list(range(len(expiration_display)))
        expiration = st.selectbox('Expiration', expiration_options, format_func=lambda x: expiration_display[x])

        gender_display = ('Female', 'Male')
        gender_options = list(range(len(gender_display)))
        gender = st.selectbox('Gender', gender_options, format_func=lambda x: gender_display[x])

    with col3:
        age_display = ('21', '46', '26', '31', '41', '50plus', '36', 'below21')
        age_options = list(range(len(age_display)))
        age = st.selectbox('Age', age_options, format_func=lambda x: age_display[x])

        maritalStatus_display = ('Unmarried partner', 'Single', 'Married partner', 'Divorced', 'Widowed')
        maritalStatus_options = list(range(len(maritalStatus_display)))
        maritalStatus = st.selectbox('Marital Status', maritalStatus_options,
                                     format_func=lambda x: maritalStatus_display[x])

        has_children_display = ('1', '0')
        has_children_options = list(range(len(has_children_display)))
        has_children = st.selectbox('Has Children', has_children_options, format_func=lambda x: has_children_display[x])

        education_display = ('Some college - no degree', 'Bachelors degree', 'Associates degree',
                             'High School Graduate',
                             'Graduate degree (Masters or Doctorate)', 'Some High School')
        education_options = list(range(len(education_display)))
        education = st.selectbox('Education', education_options, format_func=lambda x: education_display[x])

    with col4:
        occupation_display = ('Unemployed', 'Architecture & Engineering', 'Student', 'Education&Training&Library',
                              'Healthcare Support', 'Sales & Related', 'Management',
                              'Arts Design Entertainment Sports & Media',
                              'Computer & Mathematical', 'Life Physical Social Science', 'Personal Care & Service',
                              'Community & Social Services', 'Office & Administrative Support',
                              'Construction & Extraction',
                              'Legal', 'Installation Maintenance & Repair', 'Business & Financial',
                              'Food Preparation & Serving Related', 'Production Occupations',
                              'Building & Grounds Cleaning & Maintenance', 'Transportation & Material Moving',
                              'Protective Service', 'Healthcare Practitioners & Technical',
                              'Farming Fishing & Forestry',
                              'Retired', 'Military')
        occupation_options = list(range(len(occupation_display)))
        occupation = st.selectbox('Occupation', occupation_options, format_func=lambda x: occupation_display[x])

        income_display = ('$25000 - $37499', '$12500 - $24999', '$37500 - $49999', '$100000 or More',
                          '$50000 - $62499', 'Less than $12500', '$87500 - $99999', '$75000 - $87499',
                          '$62500 - $74999')
        income_options = list(range(len(income_display)))
        income = st.selectbox('Income', income_options, format_func=lambda x: income_display[x])

        Bar_display = ('never', 'less1', '1~3', 'gt8', '4~8')
        Bar_options = list(range(len(Bar_display)))
        Bar = st.selectbox('Bar Visits', Bar_options, format_func=lambda x: Bar_display[x])

        CoffeeHouse_display = ('never', 'less1', '4~8', '1~3', 'gt8')
        CoffeeHouse_options = list(range(len(CoffeeHouse_display)))
        CoffeeHouse = st.selectbox('CoffeeHouse Visits', CoffeeHouse_options,
                                   format_func=lambda x: CoffeeHouse_display[x])

    with col5:
        CarryAway_display = ('never', 'less1', '1~3', '4~8', 'gt8')
        CarryAway_options = list(range(len(CarryAway_display)))
        CarryAway = st.selectbox('CarryAway Visits', CarryAway_options, format_func=lambda x: CarryAway_display[x])

        RestaurantLessThan20_display = ('4~8', '1~3', 'less1', 'never', 'gt8')
        RestaurantLessThan20_options = list(range(len(RestaurantLessThan20_display)))
        RestaurantLessThan20 = st.selectbox('Restaurant <$20 Visits', RestaurantLessThan20_options,
                                            format_func=lambda x: RestaurantLessThan20_display[x])

        Restaurant20To50_display = ('1~3', 'less1', 'never', '4~8', 'gt8')
        Restaurant20To50_options = list(range(len(Restaurant20To50_display)))
        Restaurant20To50 = st.selectbox('Restaurant $20-50 Visits', Restaurant20To50_options,
                                        format_func=lambda x: Restaurant20To50_display[x])

        toCoupon_GEQ15min_display = ('1', '0')
        toCoupon_GEQ15min_options = list(range(len(toCoupon_GEQ15min_display)))
        toCoupon_GEQ15min = st.selectbox('toCoupon >=15min', toCoupon_GEQ15min_options,
                                         format_func=lambda x: toCoupon_GEQ15min_display[x])

    with col6:
        toCoupon_GEQ25min_display = ('1', '0')
        toCoupon_GEQ25min_options = list(range(len(toCoupon_GEQ25min_display)))
        toCoupon_GEQ25min = st.selectbox('toCoupon >=25min', toCoupon_GEQ25min_options,
                                         format_func=lambda x: toCoupon_GEQ25min_display[x])

        direction_same_display = ('0', '1')
        direction_same_options = list(range(len(direction_same_display)))
        direction_same = st.selectbox('Direction Same', direction_same_options,
                                      format_func=lambda x: direction_same_display[x])

    if st.button('Predict Coupon Acceptance'):
        try:
            # Load your XGBoost model
            model = pickle.load(open('xgb.pkl', 'rb'))

            # Prepare input data (match the order your model expects)
            input_data = [
                destination, passanger, weather, temperature, time, coupon,
                expiration, gender, age, maritalStatus, has_children,
                education, occupation, income, Bar, CoffeeHouse,
                CarryAway, RestaurantLessThan20, Restaurant20To50,
                toCoupon_GEQ15min, toCoupon_GEQ25min, direction_same
            ]

            # Make prediction
            prediction = model.predict([input_data])
            prediction_proba = model.predict_proba([input_data])

            # Display results
            st.subheader("Prediction Results")

            if prediction[0] == 1:
                st.success(f"🎉 The customer is likely to **ACCEPT** the **{coupon_display[coupon]}** coupon")
                st.metric("Acceptance Probability", f"{prediction_proba[0][1]:.2%}")
            else:
                st.error(f"❌ The customer is likely to **REJECT** the **{coupon_display[coupon]}** coupon")
                st.metric("Rejection Probability", f"{prediction_proba[0][0]:.2%}")

            # Show model performance metrics
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)

            # Note: These would typically come from your test set evaluation
            with col1:
                st.metric("Accuracy", "73.91%")
            with col2:
                st.metric("Precision", "74.29%")
            with col3:
                st.metric("Recall", "81.26%")
            with col4:
                st.metric("F1-Score", "77.62%")

            st.balloons()

        except FileNotFoundError:
            st.error("Model file 'xgb.pkl' not found. Please ensure the model file is in the correct directory.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")


# -------------------------------------------------------------------------------------------------------------- #

# Creating the sidebar with 3 options
options = {
    'Home': '🏠',
    'EDA': '📊',
    'Prediction': '🔮'
}

# Display the selected page content
st.sidebar.title('Navigation')
selected_page = st.sidebar.radio("Select a page", list(options.keys()))
if selected_page == 'Home':
    home()
elif selected_page == 'EDA':
    eda()
elif selected_page == 'Prediction':
    prediction()