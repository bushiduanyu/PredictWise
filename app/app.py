"""Streamlit interface for the PredictWise Random Forest model."""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import prepare_model_input


MODEL_DIR = PROJECT_ROOT / "models"


@st.cache_resource
def load_artifacts():
    """Load the trained model, its feature schema, and decision threshold."""
    model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
    threshold = float(joblib.load(MODEL_DIR / "recommended_threshold.pkl"))
    return model, feature_columns, threshold


st.set_page_config(
    page_title="PredictWise | Predictive Maintenance",
    page_icon="🛠️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            background: #f7f9fc;
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 1rem;
        }
        .model-card {
            background: linear-gradient(90deg, #eef6ff 0%, #f8fbff 100%);
            border: 1px solid #cfe3fa;
            border-left: 5px solid #2878bd;
            border-radius: 0.75rem;
            padding: 1rem 1.25rem;
            margin: 1rem 0 1.5rem 0;
        }
        .footer {
            color: #64748b;
            text-align: center;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PredictWise")
st.markdown(
    "Predictive maintenance dashboard for estimating machine failure risk from "
    "operating conditions in the AI4I 2020 dataset."
)

try:
    model, feature_columns, threshold = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files were not found. Run the 'Save Best Model' section in "
        "notebooks/Baseline_Model.ipynb first."
    )
    st.stop()

st.markdown(
    f"""
    <div class="model-card">
        <strong>Model:</strong> Random Forest (200 trees) &nbsp;•&nbsp;
        <strong>Average Precision:</strong> 0.881 &nbsp;•&nbsp;
        <strong>Decision threshold:</strong> {threshold:.2f}<br>
        <span style="color:#526579; font-size:0.9rem;">
            Selected for its strong balance between detecting failures and controlling false alarms.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Machine inputs")
    st.caption("Enter the current operating conditions, then run the assessment.")

    with st.form("prediction_form"):
        product_type = st.selectbox("Product Type", options=["L", "M", "H"])
        air_temperature = st.number_input(
            "Air temperature [K]", min_value=250.0, max_value=350.0, value=298.0
        )
        process_temperature = st.number_input(
            "Process temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=308.0,
        )
        rotational_speed = st.number_input(
            "Rotational speed [rpm]", min_value=0, max_value=5000, value=1500
        )
        torque = st.number_input(
            "Torque [Nm]", min_value=0.0, max_value=150.0, value=40.0
        )
        tool_wear = st.number_input(
            "Tool wear [min]", min_value=0, max_value=260, value=100
        )
        submitted = st.form_submit_button(
            "Run risk assessment", type="primary", use_container_width=True
        )

    st.caption(
        "Torque can strongly affect the prediction because it influences both the "
        "raw torque feature and the engineered power proxy feature."
    )

if submitted:
    raw_input = pd.DataFrame(
        [
            {
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type": product_type,
            }
        ]
    )
    model_input = prepare_model_input(raw_input, feature_columns)
    failure_probability = float(model.predict_proba(model_input)[0, 1])
    predicted_failure = failure_probability >= threshold

    if failure_probability < 0.40:
        risk_level = "Low"
        risk_icon = "🟢"
        recommendation = "Continue normal operation and routine monitoring."
    elif failure_probability < 0.70:
        risk_level = "Moderate"
        risk_icon = "🟠"
        recommendation = "Inspect the machine soon and monitor operating conditions."
    else:
        risk_level = "High"
        risk_icon = "🔴"
        recommendation = "Schedule a maintenance inspection as soon as practical."

    st.subheader("Risk assessment")
    probability_column, risk_column, threshold_column = st.columns(3)
    probability_column.metric("Failure Probability", f"{failure_probability:.1%}")
    risk_column.metric("Risk Level", f"{risk_icon} {risk_level}")
    threshold_column.metric("Decision Threshold", f"{threshold:.0%}")

    st.progress(
        min(max(failure_probability, 0.0), 1.0),
        text=f"Estimated failure probability: {failure_probability:.1%}",
    )

    predicted_label = "Failure risk detected" if predicted_failure else "No failure risk detected"
    if predicted_failure:
        st.warning(f"**Model decision:** {predicted_label}")
    else:
        st.success(f"**Model decision:** {predicted_label}")

    st.subheader("Maintenance recommendation")
    if risk_level == "High":
        st.error(f"{risk_icon} **High risk:** {recommendation}")
    elif risk_level == "Moderate":
        st.warning(f"{risk_icon} **Moderate risk:** {recommendation}")
    else:
        st.success(f"{risk_icon} **Low risk:** {recommendation}")

    st.subheader("Input summary")
    input_summary = pd.DataFrame(
        {
            "Input": [
                "Product Type",
                "Air temperature",
                "Process temperature",
                "Rotational speed",
                "Torque",
                "Tool wear",
            ],
            "Value": [
                product_type,
                f"{air_temperature:.1f} K",
                f"{process_temperature:.1f} K",
                f"{rotational_speed:,} rpm",
                f"{torque:.1f} Nm",
                f"{tool_wear} min",
            ],
        }
    )
    st.dataframe(input_summary, hide_index=True, use_container_width=True)
else:
    st.info("Use the sidebar to enter machine conditions and run a risk assessment.")

with st.expander("How PredictWise works"):
    st.markdown(
        """
        **Engineered features.** The app calculates temperature difference from process and
        air temperatures, estimates a power proxy from torque and rotational speed, and
        groups tool wear into Low, Medium, or High levels. The input is then one-hot encoded
        and aligned to the exact columns used during training.

        **Random Forest model.** The selected model combines 200 decision trees to capture
        nonlinear relationships between mechanical load, speed, temperature, and tool wear.

        **Threshold tuning.** A decision threshold of 0.40 was selected after comparing
        precision, recall, and F1 across several thresholds. A probability at or above 0.40
        is classified as failure risk.

        **Why precision and recall matter.** Recall measures how many real failures are
        detected; low recall can mean costly missed failures. Precision measures how often
        a failure alert is correct; low precision can create unnecessary inspections and
        alarm fatigue. Predictive maintenance needs a practical balance between both.
        """
    )

st.markdown(
    '<div class="footer">PredictWise • Predictive Maintenance Engineering Portfolio Project</div>',
    unsafe_allow_html=True,
)
