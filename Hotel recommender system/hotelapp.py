import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
@st.cache_data
def load_data():
    try:
        # Try relative path first, then absolute path
        if os.path.exists('hotel_features.csv'):
            return pd.read_csv('hotel_features.csv')
        else:
            return pd.read_csv('hotel_features.csv')
    except FileNotFoundError:
        st.error("hotel_features.csv file not found. Please check the file path.")
        return pd.DataFrame()
                                                  
# Load users hotel history for insights
@st.cache_data
def load_users_hotel_data():
    try:
        # Try relative path first, then absolute path
        if os.path.exists('users_hotel_history.csv'):
            return pd.read_csv('users_hotel_history.csv')
        else:
            return pd.read_csv('users_hotel_history.csv')
    except FileNotFoundError:
        st.error("users_hotel_history.csv file not found. Please check the file path.")
        return pd.DataFrame()

hotel_features = load_data()
users_hotel = load_users_hotel_data()

# Define the recommendation system function
def recommendation_system(place, price_range, days_range):
    """
    Recommend hotels based on place, price range, and days range.
    """
    filtered_hotels_by_place = hotel_features[hotel_features['place'] == place]
    filtered_hotels_by_price = filtered_hotels_by_place[
        (filtered_hotels_by_place['price'] >= price_range[0]) & 
        (filtered_hotels_by_place['price'] <= price_range[1])
    ]
    filtered_hotels_by_day = filtered_hotels_by_price[
        (filtered_hotels_by_price['days'] >= days_range[0]) & 
        (filtered_hotels_by_price['days'] <= days_range[1])
    ]
    return filtered_hotels_by_day

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", "Hotel Recommendations")

# Hotel Recommendations Section
if section == "Hotel Recommendations":
    st.title("Hotel Recommendation System")
    st.markdown("This section allows you to get hotel recommendations based on your preferences.")

    if not hotel_features.empty:
        place = st.selectbox("Select a Place:", hotel_features['place'].unique())
        price_range = st.slider("Select Price Range:", 
                                min_value=int(hotel_features['price'].min()), 
                                max_value=int(hotel_features['price'].max()), 
                                value=(int(hotel_features['price'].min()), int(hotel_features['price'].max())))
        days_range = st.slider("Select Stay Duration (Days):", 
                               min_value=int(hotel_features['days'].min()), 
                               max_value=int(hotel_features['days'].max()), 
                               value=(int(hotel_features['days'].min()), int(hotel_features['days'].max())))

        if st.button("Get Recommendations"):
            recommendations = recommendation_system(place, price_range, days_range)
            if not recommendations.empty:
                st.write("Recommended Hotels:")
                st.dataframe(recommendations)
            else:
                st.write("No hotels found matching your criteria.")
    else:
        st.error("No hotel data available.")

    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1631049307264-da0ec9d70304?w=1200');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
   
