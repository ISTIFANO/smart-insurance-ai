import streamlit as st
import pandas as pd
import joblib as lib

mod = lib.load(r"C:\Users\aamir\Desktop\YC\P\mart-insurance-ai\src\domain\best_model_xg.pkl")

st.title(" Estimation des Charges Medicales")

age = st.number_input("age", min_value=0, max_value=100, value=30)
sex = st.selectbox("sexe", ["male", "female"])
bmi = st.number_input("BMI (Indice de mase corporelle)", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=1)
smoker = st.selectbox("Fumeur ?", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])


# age = int(input("Entrez l'âge : "))
# sex = input("Entrez le sexe (male/female) : ").lower()
# bmi = float(input("Entrez le BMI (ex: 27.5) : "))
# children = int(input("Entrez le nombre d'enfants : "))
# smoker = input("Fumeur ? (yes/no) : ").lower()
# region = input("Entrez la région (northeast, northwest, southeast, southwest) : ").lower()


if st.button("Predire les charges"):
    fake_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
    
    prediction = mod.predict(fake_data)
    
    st.succes(f" Charge medicale estimee : **{prediction[0]:.2f}**")
