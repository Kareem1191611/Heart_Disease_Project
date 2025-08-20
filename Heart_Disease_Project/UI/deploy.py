import streamlit as st
import pickle
import pandas as pd


# Example input fields (adjust to your dataset features)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
ca = st.selectbox("number of major vessels (0-3) colored by flourosopy",  [0 , 1, 2, 3])
thal = st.selectbox("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect",  [3,6,7])
cp = st.selectbox("chest pain", [1, 2, 3 , 4])
thalach = st.slider("maximum heart rate achieved", 60, 210)
# employment_years = st.number_input("Years of Employment", min_value=0, value=5)
 
# Load the pickle model
with open(r"C:\Users\kareem\Desktop\Heart_Disease_Project\model\loan_model2.pkl", "rb") as f:
    pipe = pickle.load(f)


columns = ["thalach" , "thal" , "cp" , "ca" , "age"]

if st.button("Predict"):
    new_data = {
        "thalach": [thalach],
        "thal": [thal],
        "cp": [cp],
        "ca": [ca],
        "age": [age]
    }
   

    

    new_df = pd.DataFrame(new_data)
    new_df = new_df[columns]

    prediction = pipe.predict(new_df)
    st.success(f"Prediction: {prediction[0]}")
