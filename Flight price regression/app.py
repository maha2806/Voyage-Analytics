import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and encoders
model = joblib.load('flight_price_model_rf .joblib')
le_from = joblib.load('le_from.joblib')
le_to = joblib.load('le_to.joblib')
le_type = joblib.load('le_type.joblib')
le_agency = joblib.load('le_agency.joblib')

# Load the CSV file containing time and distance information
try:
    new_df = pd.read_csv('new_df.csv')
except FileNotFoundError:
    # If new_df.csv is not found, create it from the current df
    new_df = pd.DataFrame(columns=['from', 'to', 'time', 'distance']) # Initialize an empty DataFrame
    print("new_df.csv not found. Please ensure it's generated or present.")

def flight_price_prediction_page():
    """Function for the Flight Price Prediction page."""
    # Set page title
    st.title('Flight Price Predictor')

    # Create input form
    st.header('Enter Flight Details')

    # Get unique values for dropdowns
    from_cities = le_from.classes_
    to_cities = le_to.classes_

    # Ensure 'From' and 'To' are not the same
    from_location = st.selectbox("Select Departure Location", new_df['from'].unique() if not new_df.empty else [])
    to_location = st.selectbox("Select Arrival Location", new_df['to'].unique() if not new_df.empty else [])

    if from_location and to_location and from_location == to_location:
        st.write("Departure and Arrival locations cannot be the same. Please select different locations.")
    elif from_location and to_location:
        # Automatically fill 'time' and 'distance' based on selected locations
        filtered_row = new_df[(new_df['from'] == from_location) & (new_df['to'] == to_location)]
        if not filtered_row.empty:
            time = filtered_row.iloc[0]['time']
            distance = filtered_row.iloc[0]['distance']
        else:
            st.warning("No data available for the selected route.")
            time, distance = 0, 0

        # Display time and distance to the user
        st.write(f"Time: {time} hours")
        st.write(f"Distance: {distance} km")

        # Dropdown for flight type
        flight_types = ['economic', 'premium', 'firstClass']
        flight_type = st.selectbox('Flight Type:', flight_types)

        # Dropdown for agency
        agencies = le_agency.classes_
        agency = st.selectbox('Select agency:', agencies)

        # Number of passengers
        num_passengers = st.number_input("Select number of Passengers", min_value=1, value=1, step=1)

        # Add predict button
        if st.button('Predict Price'):
            # Transform inputs using label encoders
            from_encoded = le_from.transform([from_location])[0]
            to_encoded = le_to.transform([to_location])[0]
            type_encoded = le_type.transform([flight_type])[0]
            agency_encoded = le_agency.transform([agency])[0]

            # Create input array
            input_data = np.array([[from_encoded, to_encoded, agency_encoded, type_encoded, time, distance]])

            # Make prediction
            prediction = model.predict(input_data)[0]
            total_price = prediction * num_passengers

            # Display result
            st.success(f'Estimated price for {num_passengers} passenger(s): R$ {total_price:.2f}')

if __name__ == '__main__':
    st.set_page_config(initial_sidebar_state="collapsed")
    
    # Add background image
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #808080;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    [data-testid="stToolbar"] {
        right: 2rem;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # Add custom styling
    st.markdown('''
        <style>
        .main {
            padding: 2rem;
        }
        .stSlider > label {
            font-size: 18px;
            font-weight: bold;
        }
        h1 {
            color: #1f77b4;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .metric-card {
            background-color: rgba(255,255,255,0.95);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        </style>
    ''', unsafe_allow_html=True)
    
    # Add footer with branding
    # Add footer with branding
    st.markdown('---')
    st.markdown('<div style="text-align: center; color: #333333; margin-top: 2rem;">'
                '<p>✈️ <strong>Flight Price Predictor</strong> | Powered by Machine Learning</p>'
                '</div>', unsafe_allow_html=True)
    
    flight_price_prediction_page()



