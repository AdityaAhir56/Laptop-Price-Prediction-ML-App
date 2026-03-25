import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

# ------------------ LOAD FILES ------------------
model = joblib.load("model/XGB_model.pkl")
scalers = joblib.load("model/Standard_scalers.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
target_encoding = joblib.load("model/target_means.pkl")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<p class="title">💻 Laptop Price Predictor</p>', unsafe_allow_html=True)
st.write("Predict laptop prices using Machine Learning 🚀")

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("🔧 Enter Laptop Specifications")

ram = st.sidebar.slider("RAM (GB)", 2, 64, 8)
screen_height = st.sidebar.number_input("Screen Height", value=1080)
screen_width = st.sidebar.number_input("Screen Width", value=1920)
cpu_freq = st.sidebar.slider("CPU Frequency (GHz)", 1.0, 5.0, 2.5)
primary_storage = st.sidebar.slider("Storage (GB)", 128, 2048, 512)
weight = st.sidebar.slider("Weight (kg)", 0.5, 5.0, 2.0)
screen_size = st.sidebar.slider("Screen Size (inches)", 10.0, 20.0, 15.6)

primary_storage_type = st.sidebar.selectbox(
    "Storage Type", label_encoders['Primary_Storage_Type'].classes_
)
type_name = st.sidebar.selectbox(
    "Laptop Type", label_encoders['Type_Name'].classes_
)
screen_definition = st.sidebar.selectbox(
    "Screen Definition", label_encoders['Screen_Definition'].classes_
)

gpu_model = st.sidebar.selectbox(
    "GPU Model", list(pd.Series(target_encoding.keys()).unique())
)

# ------------------ FEATURE LIST ------------------
feature_names = [
    'Ram','Primary_Storage_Type','Screen_Height','Screen_Width','Type_Name',
    'CPU_frequency','GPU_model','Primary_Storage','Weight','Screen_Size','Screen_Definition'
]

# ------------------ MAIN LAYOUT ------------------
col1, col2 = st.columns([2, 1])

# ------------------ PREDICTION ------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction")

    if st.button("🔮 Predict Price"):
        with st.spinner("Running model inference..."):
            time.sleep(0.6)

            # Encoding
            primary_storage_type_enc = label_encoders['Primary_Storage_Type'].transform([primary_storage_type])[0]
            type_name_enc = label_encoders['Type_Name'].transform([type_name])[0]
            screen_definition_enc = label_encoders['Screen_Definition'].transform([screen_definition])[0]
            gpu_encoded = float(target_encoding[gpu_model])

            # Input dictionary
            input_dict = {
                'Ram': float(ram),
                'Primary_Storage_Type': float(primary_storage_type_enc),
                'Screen_Height': float(screen_height),
                'Screen_Width': float(screen_width),
                'Type_Name': float(type_name_enc),
                'CPU_frequency': float(cpu_freq),
                'GPU_model': float(gpu_encoded),
                'Primary_Storage': float(primary_storage),
                'Weight': float(weight),
                'Screen_Size': float(screen_size),
                'Screen_Definition': float(screen_definition_enc)
            }

            input_data = np.array([list(input_dict.values())], dtype=float)

            # Scaling
            for i, col in enumerate(input_dict.keys()):
                if col in scalers:
                    input_data[:, i] = scalers[col].transform(
                        input_data[:, i].reshape(-1,1)
                    ).flatten()

            prediction = model.predict(input_data)[0]

        # -------- RESULT --------
        st.success(f"💰 Estimated Price: €{prediction:.2f}")

        # -------- INPUT SUMMARY --------
        st.subheader("🧾 Input Summary")
        summary_df = pd.DataFrame(input_dict.items(), columns=["Feature", "Value"])
        st.dataframe(summary_df, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ GLOBAL FEATURE IMPORTANCE ------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Model Insights")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Developed as part of Data Analyst Internship Project | C.K. Pithawalla College of Engineering and Technology GTU BE CO Final Year")