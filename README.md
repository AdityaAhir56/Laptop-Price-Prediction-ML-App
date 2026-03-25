# 💻 Laptop Price Prediction ML App

## 📌 Project Overview

This project focuses on predicting laptop prices using machine learning based on hardware specifications. It includes data preprocessing, feature engineering, model development using XGBoost, and deployment through a Streamlit web application.

---

## 🎯 Objectives

- Analyze factors affecting laptop pricing  
- Build a predictive machine learning model  
- Handle categorical and numerical features effectively  
- Deploy the model into an interactive web application  
- Provide real-time price predictions  

---

## 🗂️ Dataset Description

The dataset contains laptop specifications along with their corresponding prices.

### 🔹 Features Used:

- RAM  
- CPU Frequency  
- Primary Storage  
- Primary Storage Type  
- GPU Model  
- Type Name  
- Screen Size  
- Screen Height  
- Screen Width  
- Screen Definition  
- Weight  
- Price (target variable)  

---

## 🧹 Data Preprocessing

- Cleaned and structured dataset  
- Handled missing values  
- Encoded categorical variables:
  - Label Encoding  
  - Target Encoding (GPU model)  
- Applied feature scaling using StandardScaler  

---

## 📊 Exploratory Data Analysis

EDA was performed to understand relationships between features and price.

### 🔹 Key Analysis:

- Price vs Weight  
- Price distribution across laptop types  
- Impact of screen definition on price  
- Price variation across CPU and GPU companies  
- Storage type vs price comparison

---

## ⚙️ Feature Engineering

- Transformed categorical features into numerical format  
- Normalized numerical features  
- Selected relevant features for modeling  

---

## 🤖 Model Building

- Model Used: **XGBoost Regressor**  

### 🔹 Why XGBoost?

- Handles non-linear relationships  
- High performance on structured data  
- Robust and efficient  

### 🔹 Steps:

- Train-test split  
- Applied preprocessing pipeline  
- Model training  

---

## 📊 Model Performance

- Achieved strong predictive performance  
- Captures complex relationships between features  
- Validated using model predictions and feature importance  

---

## 🚀 Streamlit Web Application

The project includes a fully interactive Streamlit app for real-time predictions.

---

### 🔹 Features:

- User-friendly input interface  
- Real-time price prediction  
- Input summary display  
- Feature importance visualization  
- Clean and responsive UI  

---

## 🌐 Live Demo

🚀 Try the app here:
👉 https://laptop-price-prediction-ml-app-mqxcvvbdrfnpbdk6rahret.streamlit.app/ 

---

## ▶️ How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/your-username/laptop-price-prediction-ml-app.git
```

2. Navigate to the project folder:
```bash
cd laptop-price-prediction-ml-app
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```bash
├── data/
│   └── laptop_prices.csv
├── model/
│   ├── XGB_model.pkl
│   ├── label_encoders.pkl
│   ├── Standard_scalers.pkl
│   ├── target_means.pkl
├── notebooks/
│   └── analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  

---

## 🔍 Key Insights

- RAM is a major contributor to laptop price  
- SSD storage increases price significantly  
- GPU type strongly impacts pricing  
- Laptop type influences cost structure  

---

## 🚀 Future Improvements

- Add more advanced models  
- Improve UI/UX design  
- Integrate with real-time product data  
- Deploy on cloud platforms  

---

## 👨‍💻 Author

**Adityakumar Umeshkumar Ahir**  
Data Analyst Intern | Linkedin: https://www.linkedin.com/in/adityaahir/
