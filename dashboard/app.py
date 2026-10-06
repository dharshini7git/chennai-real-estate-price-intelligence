import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_price_model.pkl"
DATA_PATH = BASE_DIR / "datasets" / "chennai_housing_combined.csv"
EDA_PATH = BASE_DIR / "eda_plots"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chennai Real Estate Price Intelligence",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
data = load_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_price(price):
    """Convert INR price into Lakhs/Crores."""

    if price >= 1_00_00_000:
        return f"₹{price / 1_00_00_000:.2f} Cr"

    return f"₹{price / 1_00_000:.2f} L"


def price_category(gap_percent):
    """Classify property based on price gap."""

    if gap_percent <= -10:
        return "Potentially Underpriced"

    elif gap_percent >= 10:
        return "Potentially Overpriced"

    else:
        return "Fairly Priced"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 Chennai Real Estate")

st.sidebar.markdown(
    """
### Navigation

Use the sections below to explore the system.

- 🏠 Price Prediction
- 📊 Dataset Insights
- 🤖 Model Information
"""
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
**Project**

Chennai-focused real-estate price intelligence system.

The system predicts a fair property price and helps identify potentially underpriced or overpriced properties.
"""
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🏠 Chennai Real Estate Price Intelligence System")

st.markdown(
    """
### Predict fair property prices and analyze pricing

Enter the property details below to estimate its predicted fair market price.
"""
)

st.markdown("---")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🏠 Price Prediction",
        "📊 Dataset Insights",
        "🤖 Model Information"
    ]
)


# ============================================================
# TAB 1 — PRICE PREDICTION
# ============================================================

with tab1:

    st.header("Property Details")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # BASIC PROPERTY DETAILS
    # --------------------------------------------------------

    with col1:

        st.subheader("Basic Information")

        area_sqft = st.number_input(
            "Area (sqft)",
            min_value=300.0,
            max_value=10000.0,
            value=1200.0,
            step=50.0
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

        total_rooms = st.number_input(
            "Total Rooms",
            min_value=1,
            max_value=20,
            value=4,
            step=1
        )

        building_age = st.number_input(
            "Building Age (years)",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    with col2:

        st.subheader("Property Information")

        location_options = sorted(
            data["location"].dropna().astype(str).unique().tolist()
        )

        property_type_options = sorted(
            data["property_type"].dropna().astype(str).unique().tolist()
        )

        parking_options = sorted(
            data["parking"].dropna().astype(str).unique().tolist()
        )

        construction_options = sorted(
            data["construction_status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        location = st.selectbox(
            "Location",
            location_options
        )

        property_type = st.selectbox(
            "Property Type",
            property_type_options
        )

        parking = st.selectbox(
            "Parking",
            parking_options
        )

        construction_status = st.selectbox(
            "Construction Status",
            construction_options
        )

    # --------------------------------------------------------
    # AMENITIES
    # --------------------------------------------------------

    st.subheader("Amenities")

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        swimming_pool = st.checkbox("Swimming Pool")
        gymnasium = st.checkbox("Gymnasium")
        landscaped_gardens = st.checkbox("Landscaped Gardens")

    with a2:
        club_house = st.checkbox("Club House")
        power_backup = st.checkbox("Power Backup")
        intercom = st.checkbox("Intercom")

    with a3:
        security_24x7 = st.checkbox("24×7 Security")
        childrens_play_area = st.checkbox("Children's Play Area")

    # Convert boolean values to the numeric representation
    # expected by the model.
    swimming_pool = int(swimming_pool)
    gymnasium = int(gymnasium)
    landscaped_gardens = int(landscaped_gardens)
    club_house = int(club_house)
    power_backup = int(power_backup)
    intercom = int(intercom)
    security_24x7 = int(security_24x7)
    childrens_play_area = int(childrens_play_area)

    st.markdown("---")

    # ========================================================
    # OPTIONAL ACTUAL/LISTED PRICE
    # ========================================================

    st.subheader("Optional: Listed / Actual Price")

    st.caption(
        "Enter the property's listed or actual price if you want to compare it with the predicted fair price."
    )

    actual_price_lakhs = st.number_input(
        "Listed Price (₹ Lakhs)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Fair Price",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [
                {
                    "area_sqft": area_sqft,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "total_rooms": total_rooms,
                    "building_age": building_age,
                    "location": location,
                    "property_type": property_type,
                    "parking": parking,
                    "construction_status": construction_status,
                    "swimming_pool": swimming_pool,
                    "gymnasium": gymnasium,
                    "landscaped_gardens": landscaped_gardens,
                    "club_house": club_house,
                    "power_backup": power_backup,
                    "intercom": intercom,
                    "security_24x7": security_24x7,
                    "childrens_play_area": childrens_play_area
                }
            ]
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        predicted_price = float(model.predict(input_data)[0])

        # Prevent negative display values
        predicted_price = max(predicted_price, 0)

        # ----------------------------------------------------
        # DISPLAY PREDICTION
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader("🎯 Predicted Fair Price")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Predicted Price",
                format_price(predicted_price)
            )

        with result_col2:

            st.metric(
                "Price in Lakhs",
                f"₹{predicted_price / 1_00_000:.2f} L"
            )

        with result_col3:

            st.metric(
                "Price per sqft",
                f"₹{predicted_price / area_sqft:,.0f}"
            )

        # ----------------------------------------------------
        # PRICE INTELLIGENCE
        # ----------------------------------------------------

        if actual_price_lakhs > 0:

            actual_price = actual_price_lakhs * 1_00_000

            price_difference = actual_price - predicted_price

            gap_percent = (
                price_difference / predicted_price
            ) * 100 if predicted_price > 0 else 0

            category = price_category(gap_percent)

            st.markdown("---")

            st.subheader("💡 Price Intelligence")

            p1, p2, p3 = st.columns(3)

            with p1:

                st.metric(
                    "Listed Price",
                    format_price(actual_price)
                )

            with p2:

                st.metric(
                    "Price Difference",
                    format_price(abs(price_difference))
                )

            with p3:

                st.metric(
                    "Price Gap",
                    f"{gap_percent:+.2f}%"
                )

            if category == "Potentially Underpriced":

                st.success(
                    "🟢 Potentially Underpriced\n\n"
                    "The listed price is more than 10% below the predicted fair price."
                )

            elif category == "Potentially Overpriced":

                st.error(
                    "🔴 Potentially Overpriced\n\n"
                    "The listed price is more than 10% above the predicted fair price."
                )

            else:

                st.info(
                    "🔵 Fairly Priced\n\n"
                    "The listed price is within ±10% of the predicted fair price."
                )

            st.caption(
                "This classification is an analytical indication based on the model prediction, not a guarantee of the property's true market value."
            )


# ============================================================
# TAB 2 — DATASET INSIGHTS
# ============================================================

with tab2:

    st.header("📊 Dataset Insights")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Properties",
            f"{len(data):,}"
        )

    with c2:
        st.metric(
            "Features",
            f"{data.shape[1]:,}"
        )

    with c3:
        st.metric(
            "Median Price",
            format_price(data["price"].median())
        )

    with c4:
        st.metric(
            "Average Price",
            format_price(data["price"].mean())
        )

    st.markdown("---")

    st.subheader("Price Distribution")

    image_path = EDA_PATH / "01_price_distribution.png"

    if image_path.exists():
        st.image(
            str(image_path),
            width=700
        )

    st.subheader("Area vs Price")

    image_path = EDA_PATH / "02_area_vs_price.png"

    if image_path.exists():
        st.image(
            str(image_path),
            width=700
        )

    st.subheader("Average Price by Location")

    image_path = EDA_PATH / "04_average_price_by_location.png"

    if image_path.exists():
        st.image(
            str(image_path),
            width=700
        )

    st.subheader("Price by Bedrooms")

    image_path = EDA_PATH / "03_price_by_bedrooms.png"

    if image_path.exists():
        st.image(
            str(image_path),
            width=700
        )


# ============================================================
# TAB 3 — MODEL INFORMATION
# ============================================================

with tab3:

    st.header("🤖 Model Information")

    st.markdown(
        """
### Final Model

**Algorithm:** Random Forest Regressor

**Number of Trees:** 300

**Maximum Tree Depth:** 20

**Minimum Samples per Leaf:** 2

**Training Strategy:** 80/20 train-test split

**Cross-Validation:** 5-fold cross-validation
"""
    )

    st.markdown("---")

    st.subheader("Model Performance")

    evaluation_path = (
        BASE_DIR
        / "final_results"
        / "final_model_comparison.csv"
    )

    if evaluation_path.exists():

        evaluation = pd.read_csv(evaluation_path)

        st.dataframe(
            evaluation,
            width=700,
            hide_index=True
        )

    st.markdown("---")

    st.subheader("Feature Importance")

    importance_path = (
        BASE_DIR
        / "explainability"
        / "01_feature_importance.png"
    )

    if importance_path.exists():

        st.image(
            str(importance_path),
            width=700
        )

    st.markdown("---")

    st.subheader("Actual vs Predicted")

    actual_predicted_path = (
        BASE_DIR
        / "explainability"
        / "02_actual_vs_predicted.png"
    )

    if actual_predicted_path.exists():

        st.image(
            str(actual_predicted_path),
            width=700
        )

    st.markdown("---")

    st.subheader("Prediction Residuals")

    residual_path = (
        BASE_DIR
        / "explainability"
        / "03_prediction_residuals.png"
    )

    if residual_path.exists():

        st.image(
            str(residual_path),
            width=700
        )

    st.markdown("---")

    st.info(
        """
**Important:** The model predicts an estimated fair price based on the
patterns learned from the available Chennai property datasets. The
prediction should be treated as an analytical estimate rather than a
guaranteed market price.
"""
    )