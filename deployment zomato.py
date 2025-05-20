import pickle
import requests
import os
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------
# Load model from Google Drive
# -------------------------------
FILE_ID = "1sJR5gmNxfnu8dysxVt1bf5sOEW8Qg0Hj"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_FILENAME = "Final_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        try:
            st.info("Downloading model from Google Drive...")
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_FILENAME, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                st.success("Model downloaded successfully.")
            else:
                st.error(f"Failed to download model. Status code: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None
    with open(MODEL_FILENAME, "rb") as f:
        model = pickle.load(f)
    return model

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Zomato Restaurant Recommendation System")
st.header("Find the best restaurant for you!")
st.subheader("Enter your preferences below:")

st.sidebar.header("Restaurant Details")
st.sidebar.text("Please fill in the following details to get a recommendation.")
st.sidebar.text("Note: All fields are mandatory.")

online_order = st.selectbox("Online Order", ['Yes', 'No'])
book_table = st.selectbox("Book Table", ['Yes', 'No'])

location = st.selectbox('Location', [
    'Banashankari', 'Basavanagudi', 'other', 'Jayanagar', 'JP Nagar',
    'Bannerghatta Road', 'BTM', 'Electronic City', 'Wilson Garden',
    'Shanti Nagar', 'Koramangala 5th Block', 'Richmond Road', 'HSR',
    'Koramangala 7th Block', 'Bellandur', 'Sarjapur Road',
    'Marathahalli', 'Whitefield', 'Old Airport Road', 'Indiranagar',
    'Koramangala 1st Block', 'Frazer Town', 'MG Road', 'Brigade Road',
    'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road',
    'Shivajinagar', 'St. Marks Road', 'Cunningham Road',
    'Commercial Street', 'Vasanth Nagar', 'Domlur',
    'Koramangala 8th Block', 'Ejipura', 'Jeevan Bhima Nagar',
    'Kammanahalli', 'Koramangala 6th Block', 'Brookefield',
    'Koramangala 3rd Block', 'Koramangala 4th Block', 'Banaswadi',
    'Kalyan Nagar', 'Malleshwaram', 'Rajajinagar', 'New BEL Road'
])

Restaurant_type = st.selectbox('Restaurant Type', [1, 2])
Cuisines = st.selectbox('Cuisines', [3, 2, 1, 4, 5, 8, 7, 6])
approx_cost = st.slider('Approx. Cost (INR)', min_value=30, max_value=6000, value=300, step=50)

listed_in_type = st.selectbox('Listed In (Type)', [
    'Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
    'Drinks & nightlife', 'Pubs and bars'
])

listed_in_city = st.selectbox('Listed In (City)', [
    'Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
    'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
    'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
    'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
    'Koramangala 4th Block', 'Koramangala 5th Block',
    'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
    'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
    'Old Airport Road', 'Rajajinagar', 'Residency Road',
    'Sarjapur Road', 'Whitefield'
])

# -------------------------------
# Prepare input
# -------------------------------
data_frame_deploy = pd.DataFrame({
    'online_order': [online_order],
    'book_table': [book_table],
    'location': [location],
    'rest_type': [Restaurant_type],
    'cuisines': [Cuisines],
    'approx_cost': [approx_cost],
    'listed_in(type)': [listed_in_type],
    'listed_in(city)': [listed_in_city]
})

# -------------------------------
# Predict
# -------------------------------
Confirm = st.sidebar.button("Click to Predict")

if Confirm and model:
    result = model.predict(data_frame_deploy)
    if result[0] == 1:
        st.sidebar.success("✅ Recommended!")
        st.sidebar.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*vNVomBKGpJZEKI4PrQT5hQ.jpeg", width=300)
    else:
        st.sidebar.error("❌ Not Recommended!")
        st.sidebar.image("https://testteach.co.uk/wp-content/uploads/2021/01/AdobeStock_275170280-1.png", width=300)
