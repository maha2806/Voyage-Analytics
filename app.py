from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Voyage Analytics - combined Streamlit application
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
FLIGHT_DIR = ROOT / "Flight price regression"
GENDER_DIR = ROOT / "Gender classification files"
HOTEL_DIR = ROOT / "Hotel recommender system"

st.set_page_config(
    page_title="Voyage Analytics | Travel Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2.5rem;}
      [data-testid="stSidebar"] {background: #f3f6fb;}
      .hero {padding: 1.6rem 1.8rem; border-radius: 18px;
             background: linear-gradient(120deg,#14243b,#24577a); color: white;
             margin-bottom: 1.2rem;}
      .hero h1 {color: white; margin-bottom: .35rem;}
      .hero p {color: #e2edf7; margin-bottom: 0;}
      .small-note {color: #64748b; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading flight prediction model...")
def load_flight_assets():
    model = joblib.load(FLIGHT_DIR / "flight_price_model_rf .joblib")
    encoders = {
        "from": joblib.load(FLIGHT_DIR / "le_from.joblib"),
        "to": joblib.load(FLIGHT_DIR / "le_to.joblib"),
        "type": joblib.load(FLIGHT_DIR / "le_type.joblib"),
        "agency": joblib.load(FLIGHT_DIR / "le_agency.joblib"),
    }
    return model, encoders


@st.cache_resource(show_spinner="Loading name classification model...")
def load_gender_assets():
    from sentence_transformers import SentenceTransformer

    model_data = joblib.load(GENDER_DIR / "name_gender_classifier.pkl")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model_data["classifier"], model_data["label_encoder"], embedding_model


@st.cache_data(show_spinner=False)
def load_hotel_data():
    path = HOTEL_DIR / "hotel_features.csv"
    return pd.read_csv(path)


def show_home():
    st.markdown(
        """
        <div class="hero">
          <h1>✈️ Voyage Analytics</h1>
          <p>Your Travel Intelligence Hub — Predict. Analyze. Explore.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Explore our machine-learning solutions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🛫 Flight Price Prediction")
        st.write("Estimate a flight's price using route, airline, cabin type, travel time and distance.")
    with c2:
        st.markdown("### 👤 Name Gender Classification")
        st.write("Explore the name-based classification model and its predicted label and confidence.")
    with c3:
        st.markdown("### 🏨 Hotel Recommendations")
        st.write("Find hotels that match a destination, nightly price range and stay duration.")
    st.divider()
    st.info("Choose a tool from the navigation menu in the sidebar to get started.")
    st.caption("Academic project demonstration. Model predictions are estimates and should not be treated as booking or identity advice.")


def flight_page():
    st.title("🛫 Flight Price Predictor")
    st.write("Enter the flight details below to estimate the total ticket price.")
    st.warning(
        "The original project ZIP does not contain `new_df.csv` (the route-to-time/distance lookup). "
        "For now, enter the estimated flight time and distance manually."
    )

    try:
        model, enc = load_flight_assets()
    except Exception as exc:
        st.error(f"Could not load the flight model or encoders: {exc}")
        st.stop()

    with st.form("flight_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            from_city = st.selectbox("Departure city", enc["from"].classes_)
            flight_type = st.selectbox("Flight type", enc["type"].classes_)
            flight_time = st.number_input("Estimated flight time (hours)", min_value=0.1, max_value=40.0, value=2.0, step=0.5)
        with col2:
            to_options = [city for city in enc["to"].classes_ if city != from_city]
            to_city = st.selectbox("Arrival city", to_options)
            agency = st.selectbox("Airline / agency", enc["agency"].classes_)
            distance = st.number_input("Flight distance (km)", min_value=1.0, max_value=20000.0, value=1000.0, step=50.0)
        passengers = st.number_input("Number of passengers", min_value=1, max_value=20, value=1, step=1)
        submitted = st.form_submit_button("Predict flight price", type="primary")

    if submitted:
        try:
            # Keep feature order aligned with the model's stored training columns:
            # from_num, to_num, flightType_num, agency_num, time, distance
            features = np.array([[
                enc["from"].transform([from_city])[0],
                enc["to"].transform([to_city])[0],
                enc["type"].transform([flight_type])[0],
                enc["agency"].transform([agency])[0],
                flight_time,
                distance,
            ]], dtype=float)
            per_passenger = float(model.predict(features)[0])
            total = per_passenger * passengers
            st.success(f"Estimated total price for {passengers} passenger(s): **R$ {total:,.2f}**")
            st.caption(f"Estimated price per passenger: R$ {per_passenger:,.2f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def gender_page():
    st.title("👤 Name-Based Gender Classification")
    st.write("Enter a name to see the label predicted by the trained classification model.")
    name = st.text_input("Enter a name", placeholder="e.g. Alex")
    if st.button("Classify name", type="primary", disabled=not name.strip()):
        try:
            classifier, label_encoder, embedding_model = load_gender_assets()
            with st.spinner("Creating name embedding and predicting..."):
                embedding = embedding_model.encode([name.strip()])
                prediction = classifier.predict(embedding)
                probabilities = classifier.predict_proba(embedding)
                label = label_encoder.inverse_transform(prediction)[0]
                confidence = float(np.max(probabilities[0]) * 100)
            st.metric("Model-predicted label", str(label).capitalize(), f"{confidence:.1f}% model confidence")
            st.caption("This is a statistical prediction from the training data. A name does not reliably determine a person's gender.")
        except ImportError:
            st.error("The `sentence-transformers` package is not installed. Install the packages listed in requirements.txt and try again.")
        except Exception as exc:
            st.error(f"Could not run the classification model: {exc}")


def hotel_page():
    st.title("🏨 Hotel Recommendation System")
    st.write("Filter the available hotel records by destination, price and stay duration.")
    try:
        hotels = load_hotel_data()
    except Exception as exc:
        st.error(f"Could not load hotel data: {exc}")
        return

    required = {"hotelName", "place", "price", "days"}
    if hotels.empty or not required.issubset(hotels.columns):
        st.error("Hotel data is empty or is missing one or more required columns: hotelName, place, price, days.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        place = st.selectbox("Destination", sorted(hotels["place"].dropna().unique()))
    with col2:
        price_min = float(hotels["price"].min())
        price_max = float(hotels["price"].max())
        price_range = st.slider("Price per day", min_value=price_min, max_value=price_max,
                                value=(price_min, price_max))
    with col3:
        day_min = int(hotels["days"].min())
        day_max = int(hotels["days"].max())
        day_range = st.slider("Stay duration (days)", min_value=day_min, max_value=day_max,
                              value=(day_min, day_max))

    results = hotels[
        (hotels["place"] == place)
        & hotels["price"].between(price_range[0], price_range[1])
        & hotels["days"].between(day_range[0], day_range[1])
    ].sort_values("price")

    st.subheader(f"Matching hotels ({len(results)})")
    if results.empty:
        st.info("No hotels match those filters. Try widening the price range or stay duration.")
    else:
        st.dataframe(results.rename(columns={
            "hotelName": "Hotel", "place": "Destination", "price": "Price per day", "days": "Stay (days)"
        }), use_container_width=True, hide_index=True)


with st.sidebar:
    st.title("✈️ Voyage Analytics")
    st.caption("Travel intelligence powered by machine learning")
    page = st.radio(
        "Navigation",
        ["Home", "Flight Price Prediction", "Gender Classification", "Hotel Recommendations"],
        index=0,
    )
    st.divider()
    st.caption("MLOps Capstone Project")

if page == "Home":
    show_home()
elif page == "Flight Price Prediction":
    flight_page()
elif page == "Gender Classification":
    gender_page()
elif page == "Hotel Recommendations":
    hotel_page()
