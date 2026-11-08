
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# VOYAGE ANALYTICS - COMBINED STREAMLIT APPLICATION
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

# -----------------------------------------------------------------------------
# CUSTOM CSS - MINIMAL, ELEGANT AND PROFESSIONAL
# -----------------------------------------------------------------------------


st.markdown(
    """
    <style>

    /* GLOBAL BACKGROUND */
    .stApp {
        background-color: #171D29;
        color: #E6EAF0;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #111722;
        border-right: 1px solid #303B4B;
    }

    [data-testid="stSidebar"] h1 {
        color: #F1F4F8 !important;
        font-size: 1.4rem;
        font-weight: 750;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #B5C0CF;
    }

    [data-testid="stSidebar"] hr {
        border-color: #354153;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] [role="radio"] {
        padding: 9px 12px;
        border-radius: 8px;
    }

    /* MAIN TEXT */
    h1, h2, h3 {
        color: #F1F4F8 !important;
        letter-spacing: -0.5px;
    }

    h1 {
        font-weight: 750 !important;
    }

    h2, h3 {
        font-weight: 650 !important;
    }

    [data-testid="stMain"] p,
    [data-testid="stMain"] label {
        color: #C0CAD7;
    }

    /* HERO */
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #202D40, #304D68);
        border: 1px solid #40546B;
        margin-bottom: 2rem;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 2.6rem;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #D2DFEB !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* FEATURE CARDS */
    .feature-card {
        background: #222C3B;
        border: 1px solid #354458;
        border-radius: 12px;
        padding: 1.4rem;
        min-height: 185px;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .feature-card h3 {
        color: #F1F4F8 !important;
        font-size: 1.22rem;
        margin-top: 0;
        margin-bottom: 0.8rem;
    }

    .feature-card p {
        color: #B8C5D4 !important;
        font-size: 0.96rem;
        line-height: 1.65;
        margin-bottom: 0;
    }

    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
    }

    /* INFORMATION BANNER */
    .notice-card {
        background: #26364A;
        border-left: 4px solid #8AAAC8;
        padding: 16px 20px;
        border-radius: 8px;
        color: #E5EDF5;
        font-weight: 500;
        margin-top: 1rem;
    }

    /* BUTTONS */
    .stButton button,
    .stFormSubmitButton button {
        background: #496681;
        color: #FFFFFF !important;
        border: 1px solid #607F9C;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton button:hover,
    .stFormSubmitButton button:hover {
        background: #5A7895;
        border-color: #86A5C1;
        color: #FFFFFF !important;
    }

    /* INPUTS */
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    [data-baseweb="select"] > div {
        background-color: #222C3B !important;
        color: #F1F4F8 !important;
        border: 1px solid #46566A !important;
        border-radius: 8px;
    }

    input::placeholder {
        color: #91A0B2 !important;
    }

    /* SELECTBOX DROPDOWN */
    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background-color: #222C3B !important;
        color: #F1F4F8 !important;
    }

    [role="option"] {
        color: #F1F4F8 !important;
    }

    /* SLIDERS */
    [data-testid="stSlider"] {
        padding-top: 0.5rem;
    }

    /* ALERTS */
    [data-testid="stAlert"] {
        border-radius: 9px;
    }

    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        border: 1px solid #354458;
        border-radius: 10px;
        overflow: hidden;
    }

    /* DIVIDERS */
    hr {
        border-color: #354153;
    }

    /* FOOTER */
    .footer-note {
        color: #95A4B5;
        font-size: 0.85rem;
        margin-top: 1.2rem;
    }

    /* RESPONSIVE */
    @media (max-width: 768px) {
        .hero {
            padding: 1.4rem;
        }

        .hero h1 {
            font-size: 2rem;
        }

        .block-container {
            padding-top: 1.2rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# MODEL AND DATA LOADING
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading flight prediction model...")
def load_flight_assets():

    model = joblib.load(
        FLIGHT_DIR / "flight_price_model_rf .joblib"
    )

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

    model_data = joblib.load(
        GENDER_DIR / "name_gender_classifier.pkl"
    )

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return (
        model_data["classifier"],
        model_data["label_encoder"],
        embedding_model,
    )


@st.cache_data(show_spinner=False)
def load_hotel_data():

    path = HOTEL_DIR / "hotel_features.csv"

    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------------------------

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

    st.subheader("Explore Our Machine Learning Solutions")

    st.write(
        "Discover intelligent tools designed to simplify travel-related "
        "analysis through machine learning."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🛫</div>
                <h3>Flight Price Prediction</h3>
                <p>
                    Estimate flight prices using route, airline, cabin type,
                    travel time and distance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">👤</div>
                <h3>Name Gender Classification</h3>
                <p>
                    Explore name-based classification using machine learning
                    and view predicted labels with confidence scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🏨</div>
                <h3>Hotel Recommendations</h3>
                <p>
                    Discover hotel options based on destination,
                    nightly price range and stay duration.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        """
        <div class="notice-card">
            Choose a tool from the navigation menu to get started.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer-note">
            MLOps Capstone Project | Academic Demonstration
            <br>
            Model predictions are estimates and should not be treated
            as booking or identity advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# FLIGHT PRICE PREDICTION
# -----------------------------------------------------------------------------

def flight_page():

    st.title("🛫 Flight Price Predictor")

    st.write(
        "Enter your flight details below to estimate the total ticket price."
    )

    st.warning(
        "The original project ZIP does not contain new_df.csv "
        "(the route-to-time/distance lookup). For now, enter the "
        "estimated flight time and distance manually."
    )

    try:
        model, enc = load_flight_assets()

    except Exception as exc:
        st.error(
            f"Could not load the flight model or encoders: {exc}"
        )
        st.stop()

    with st.form("flight_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            from_city = st.selectbox(
                "Departure city",
                enc["from"].classes_,
            )

            flight_type = st.selectbox(
                "Flight type",
                enc["type"].classes_,
            )

            flight_time = st.number_input(
                "Estimated flight time (hours)",
                min_value=0.1,
                max_value=40.0,
                value=2.0,
                step=0.5,
            )

        with col2:

            to_options = [
                city for city in enc["to"].classes_
                if city != from_city
            ]

            to_city = st.selectbox(
                "Arrival city",
                to_options,
            )

            agency = st.selectbox(
                "Airline / agency",
                enc["agency"].classes_,
            )

            distance = st.number_input(
                "Flight distance (km)",
                min_value=1.0,
                max_value=20000.0,
                value=1000.0,
                step=50.0,
            )

        passengers = st.number_input(
            "Number of passengers",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
        )

        submitted = st.form_submit_button(
            "Predict Flight Price",
            type="primary",
        )

    if submitted:

        try:

            features = np.array(
                [[
                    enc["from"].transform([from_city])[0],
                    enc["to"].transform([to_city])[0],
                    enc["type"].transform([flight_type])[0],
                    enc["agency"].transform([agency])[0],
                    flight_time,
                    distance,
                ]],
                dtype=float,
            )

            per_passenger = float(
                model.predict(features)[0]
            )

            total = per_passenger * passengers

            st.success(
                f"Estimated total price for {passengers} passenger(s): "
                f"**R$ {total:,.2f}**"
            )

            st.metric(
                "Estimated price per passenger",
                f"R$ {per_passenger:,.2f}",
            )

        except Exception as exc:

            st.error(f"Prediction failed: {exc}")


# -----------------------------------------------------------------------------
# NAME-BASED GENDER CLASSIFICATION
# -----------------------------------------------------------------------------

def gender_page():

    st.title("👤 Name-Based Gender Classification")

    st.write(
        "Enter a name to see the label predicted by the trained "
        "classification model."
    )

    name = st.text_input(
        "Enter a name",
        placeholder="e.g. Alex",
    )

    if st.button(
        "Classify Name",
        type="primary",
        disabled=not name.strip(),
    ):

        try:

            classifier, label_encoder, embedding_model = (
                load_gender_assets()
            )

            with st.spinner(
                "Creating name embedding and predicting..."
            ):

                embedding = embedding_model.encode(
                    [name.strip()]
                )

                prediction = classifier.predict(embedding)

                probabilities = classifier.predict_proba(
                    embedding
                )

                label = label_encoder.inverse_transform(
                    prediction
                )[0]

                confidence = float(
                    np.max(probabilities[0]) * 100
                )

            st.metric(
                "Model-predicted label",
                str(label).capitalize(),
                f"{confidence:.1f}% model confidence",
            )

            st.caption(
                "This is a statistical prediction from the training data. "
                "A name does not reliably determine a person's gender."
            )

        except ImportError:

            st.error(
                "The sentence-transformers package is not installed. "
                "Install the packages listed in requirements.txt."
            )

        except Exception as exc:

            st.error(
                f"Could not run the classification model: {exc}"
            )


# -----------------------------------------------------------------------------
# HOTEL RECOMMENDATION SYSTEM
# -----------------------------------------------------------------------------

def hotel_page():

    st.title("🏨 Hotel Recommendation System")

    st.write(
        "Filter the available hotel records by destination, price "
        "and stay duration."
    )

    try:

        hotels = load_hotel_data()

    except Exception as exc:

        st.error(f"Could not load hotel data: {exc}")
        return

    required = {
        "hotelName",
        "place",
        "price",
        "days",
    }

    if hotels.empty or not required.issubset(hotels.columns):

        st.error(
            "Hotel data is empty or is missing one or more required "
            "columns: hotelName, place, price, days."
        )

        return

    col1, col2, col3 = st.columns(3)

    with col1:

        destinations = sorted(
            hotels["place"].dropna().unique()
        )

        place = st.selectbox(
            "Destination",
            destinations,
        )

    with col2:

        price_min = float(hotels["price"].min())
        price_max = float(hotels["price"].max())

        price_range = st.slider(
            "Price per day",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
        )

    with col3:

        day_min = int(hotels["days"].min())
        day_max = int(hotels["days"].max())

        day_range = st.slider(
            "Stay duration (days)",
            min_value=day_min,
            max_value=day_max,
            value=(day_min, day_max),
        )

    results = hotels[
        (hotels["place"] == place)
        & hotels["price"].between(
            price_range[0],
            price_range[1],
        )
        & hotels["days"].between(
            day_range[0],
            day_range[1],
        )
    ].sort_values("price")

    st.subheader(
        f"Matching Hotels ({len(results)})"
    )

    if results.empty:

        st.info(
            "No hotels match those filters. Try widening the price "
            "range or stay duration."
        )

    else:

        display_results = results.rename(
            columns={
                "hotelName": "Hotel",
                "place": "Destination",
                "price": "Price per day",
                "days": "Stay (days)",
            }
        )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
        )


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------

with st.sidebar:

    st.title("✈️ Voyage Analytics")

    st.caption(
        "Travel intelligence powered by machine learning"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Flight Price Prediction",
            "Gender Classification",
            "Hotel Recommendations",
        ],
        index=0,
    )

    st.divider()

    st.caption(
        "MLOps Capstone Project"
    )

    st.caption(
        "2026 | Academic Demonstration"
    )


# -----------------------------------------------------------------------------
# APPLICATION ROUTING
# -----------------------------------------------------------------------------

if page == "Home":

    show_home()

elif page == "Flight Price Prediction":

    flight_page()

elif page == "Gender Classification":

    gender_page()

elif page == "Hotel Recommendations":

    hotel_page()
