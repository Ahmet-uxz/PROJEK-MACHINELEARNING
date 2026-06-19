import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺"
)

st.title("🩺 Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi risiko diabetes.")

# Input pengguna
pregnancies = st.number_input(
    "Jumlah Kehamilan",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Kadar Glukosa",
    min_value=0,
    max_value=300,
    value=120
)

bloodpressure = st.number_input(
    "Tekanan Darah",
    min_value=0,
    max_value=200,
    value=70
)

skinthickness = st.number_input(
    "Ketebalan Lipatan Kulit",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Kadar Insulin",
    min_value=0,
    max_value=1000,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Usia",
    min_value=1,
    max_value=120,
    value=30
)

# Tombol prediksi
if st.button("Prediksi"):

    data = np.array([[
        pregnancies,
        glucose,
        bloodpressure,
        skinthickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    hasil = model.predict(data)

    st.subheader("Hasil Prediksi")

    if hasil[0] == 1:
        st.error("⚠️ Pasien Terindikasi Diabetes")
    else:
        st.success("✅ Pasien Tidak Terindikasi Diabetes")