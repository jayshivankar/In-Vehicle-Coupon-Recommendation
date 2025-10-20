# 🧾 CouponCraft – Intelligent Coupon Recommendation System

## 📋 Project Overview
**CouponCraft** is an intelligent coupon recommendation system that predicts whether a customer will accept a coupon based on their **demographics**, **contextual factors**, and **behavioral patterns**.  
It leverages **machine learning (XGBoost)** to provide personalized recommendations, helping businesses **optimize marketing strategies** and **boost customer engagement**.

**🎥 Website link : https://in-vehicle-coupon-recommendation-qbwmhbfcbqbmzyylymhy39.streamlit.app/

---

## 🚀 Features

### 🔍 Exploratory Data Analysis (EDA)
- Comprehensive data visualization and insights  
- Interactive charts showing coupon acceptance patterns  
- Demographic and behavioral analysis  
- Missing value analysis and data preprocessing  

### 🔮 Smart Prediction
- Real-time coupon acceptance prediction using **XGBoost**  
- **74% accuracy** in predicting customer behavior  
- Probability scores for acceptance/rejection  
- User-friendly Streamlit interface with **21 feature inputs**  

---

## 📊 Key Insights
- Identifies which coupon types work best for different customer profiles  
- Analyzes the impact of **weather, time, and destination** on acceptance  
- Reveals **demographic and behavioral trends** in coupon preferences  
- Provides actionable insights for **marketing optimization**

---

## 🧠 Dataset Information
**Dataset:** In-Vehicle Coupon Recommendation  

- **Instances:** 12,684  
- **Features:** 26  
- **Target Variable:** `Accept (Y/N)`  

### Feature Categories:
- 👤 **User Demographics:** Age, Gender, Income, Occupation, Education  
- 🌤️ **Contextual Attributes:** Destination, Weather, Temperature, Passenger  
- ☕ **Behavioral Patterns:** Bar, Coffeehouse, Restaurant visits  
- 🎟️ **Coupon Attributes:** Type, Expiration  

---

## ⚙️ Installation & Setup

### 🧩 Prerequisites
- Python 3.8 or higher  
- pip (Python package manager)

### 🏗️ Project Structure
```
CouponCraft/
│
├── main.py                         # Streamlit application
├── in-vehicle-coupon-recommendation.csv  # Dataset
├── xgb.pkl                         # Trained XGBoost model
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── images/                         # Screenshots and assets
```

---

## 🎯 Model Performance

| Metric | Score |
|:-------|:------:|
| **Accuracy** | 73.91% |
| **Precision** | 74.29% |
| **Recall** | 81.26% |
| **F1-Score** | 77.62% |

### 🔝 Top 5 Most Important Features:
| Feature | Importance | Description |
|:---------|:------------:|:-------------|
| **Expiration** | 0.117 | Coupon expiration time |
| **Coupon Type** | 0.116 | Type of coupon offered |
| **toCoupon_GEQ25min** | 0.069 | Travel time > 25 minutes |
| **Bar Visits** | 0.058 | Frequency of bar visits |
| **CoffeeHouse Visits** | 0.053 | Frequency of coffee shop visits |

---

## 🖥️ Usage Guide

### 🏠 Home Page
- Project overview and introduction  
- Key features and developer info  

### 📈 EDA Page
- Interactive visualizations  
- Demographic and behavioral insights  
- Missing value and correlation analysis  

### 🧮 Prediction Page
- Input form for **21 user attributes**  
- Real-time acceptance prediction  
- Display of probability scores and model metrics  

---

## 🔧 Technical Details

### 🧹 Data Preprocessing
- Removed redundant features: `car`, `toCoupon_GEQ5min`, `direction_opp`  
- Mode imputation for missing values  
- Duplicate removal  
- Categorical encoding for non-numeric features  

### 🤖 Machine Learning Model
- **Algorithm:** XGBoost Classifier  
- **Hyperparameters:**
  - Learning Rate = 0.1  
  - Max Depth = 7  
  - N_estimators = 100  
  - Subsample = 1.0  
- **Validation:** Train-Test Split (80/20)

### 💻 Technologies Used
| Component | Tools |
|------------|-------|
| **Frontend** | Streamlit |
| **Backend** | Python, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | XGBoost, Scikit-learn |
| **Model Handling** | Pickle |

---

## 📈 Key Findings

- ☕ **Coupon Type Matters:** Coffee House and Restaurant(<$20) coupons have the highest acceptance.  
- 🌤️ **Context is Crucial:** Sunny weather and "No Urgent Place" destinations improve acceptance.  
- 👩‍💼 **Demographic Trends:** Younger age (21–26) and middle-income users are most responsive.  
- 🍽️ **Behavioral Insights:** Frequent restaurant and coffeehouse visitors accept more coupons.  
- ⏱️ **Timing:** 1-day expiration coupons outperform 2-hour ones.

---

## 🤝 Contributing

We welcome contributions!  

### Contribution Steps:
1. **Fork** the repository  
2. **Create a feature branch:**  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes:**  
   ```bash
   git commit -m "Add some AmazingFeature"
   ```
4. **Push to your branch:**  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

---

## 🧑‍💻 Author
**Developed by:** Jay Shivankar  
**Email:** shivankarjay@gmail.com  
**LinkedIn:** www.linkedin.com/in/jay-shivankar  

---

⭐ *If you like this project, don’t forget to give it a star!*
