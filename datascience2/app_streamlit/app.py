"""Bank Personal Loan Prediction — Streamlit application."""

import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (APP_SUBTITLE, APP_TITLE, COLORS, DEMOGRAPHIC_FEATURES,
                    EDUCATION_MAPPING, FEATURE_ORDER, FINANCIAL_FEATURES,
                    METRICS_PATH, MODEL_PATH, PROFESSIONAL_FEATURES,
                    SAMPLE_PROFILES, SHOW_FEATURE_IMPORTANCE)
from utils import (DataValidator, ModelLoader, PredictionEngine, ResultFormatter,
                   RiskAnalysis, create_summary_stats)

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { padding-top: 0.5rem; }
    .header-title    { font-size: 2.5rem; font-weight: 700; color: #1E293B;
                       margin-bottom: 0.5rem; }
    .header-subtitle { font-size: 1.1rem; color: #64748B; font-weight: 400; }
    .section-header  { font-size: 1.3rem; font-weight: 600; color: #1E293B;
                       margin-top: 1.5rem; margin-bottom: 1rem;
                       padding-bottom: 0.5rem; border-bottom: 2px solid #E2E8F0; }
    .stButton > button { width: 100%; padding: 0.75rem 1.5rem; font-size: 1rem;
                         font-weight: 600; border-radius: 0.5rem; border: none;
                         cursor: pointer; transition: all 0.3s ease; }
    .footer { text-align: center; padding: 2rem 0; color: #64748B; font-size: 0.85rem;
              border-top: 1px solid #E2E8F0; margin-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_resource():
    """Load model + metrics JSON exactly once per session."""
    loader = ModelLoader(MODEL_PATH, METRICS_PATH)
    if loader.load():
        return loader.get_model(), loader.get_metrics()
    return None, {}


# ---- helpers --------------------------------------------------------------

def _seed_form_state(profile: dict | None = None) -> None:
    """Initialise session_state widget keys; optionally pre-fill from a profile."""
    defaults = {
        "Age": 45, "Experience": 20, "Income": 75.0, "Family": 2,
        "CCAvg": 2.0, "Education": 1, "Mortgage": 0.0,
        "Securities Account": 0, "CD Account": 0, "Online": 0, "CreditCard": 0,
    }
    source = profile or defaults
    for k, v in source.items():
        st.session_state[k] = v


# ---- header / sidebar ------------------------------------------------------

def render_header(metrics: dict) -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<div class="header-title">🏦 {APP_TITLE}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="header-subtitle">{APP_SUBTITLE}</div>',
                    unsafe_allow_html=True)
    with col2:
        acc = metrics.get("accuracy", 0.0)
        st.info(f"Model Accuracy: {acc * 100:.2f}%")


def render_sidebar(metrics: dict) -> str:
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Choose a view:",
        ["Prediction", "Model Information", "Analytics"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sample Profiles")
    profile_names = list(SAMPLE_PROFILES.keys())
    selected = st.sidebar.selectbox("Load a sample:", ["—"] + profile_names)
    if selected != "—":
        if st.sidebar.button("Apply sample"):
            _seed_form_state(SAMPLE_PROFILES[selected])
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Performance")
    st.sidebar.metric("Accuracy",  f"{metrics.get('accuracy',  0)*100:.2f}%")
    st.sidebar.metric("Precision", f"{metrics.get('precision', 0)*100:.2f}%")
    st.sidebar.metric("Recall",    f"{metrics.get('recall',    0)*100:.2f}%")
    st.sidebar.metric("F1-Score",  f"{metrics.get('f1_score',  0)*100:.2f}%")

    return page


# ---- prediction page -------------------------------------------------------

def render_prediction_page(model, metrics: dict) -> None:
    if model is None:
        st.error("Model file not found. Run `python train_model.py` first.")
        return

    # Make sure all keys exist before any widget is rendered.
    for k in FEATURE_ORDER:
        if k not in st.session_state:
            _seed_form_state()
            break

    st.markdown('<div class="section-header">👥 Customer Information</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Demographic Information**")
        st.slider("Age",
                  int(DEMOGRAPHIC_FEATURES["Age"]["min"]),
                  int(DEMOGRAPHIC_FEATURES["Age"]["max"]),
                  key="Age",
                  help=DEMOGRAPHIC_FEATURES["Age"]["description"])
        st.slider("Family Size",
                  int(DEMOGRAPHIC_FEATURES["Family"]["min"]),
                  int(DEMOGRAPHIC_FEATURES["Family"]["max"]),
                  key="Family",
                  help=DEMOGRAPHIC_FEATURES["Family"]["description"])
        st.selectbox("Education Level", options=[1, 2, 3],
                     format_func=lambda x: EDUCATION_MAPPING.get(x, "Unknown"),
                     key="Education",
                     help=DEMOGRAPHIC_FEATURES["Education"]["description"])
    with col2:
        st.markdown("**Financial Information**")
        st.number_input("Annual Income ($1000s)",
                        min_value=float(FINANCIAL_FEATURES["Income"]["min"]),
                        max_value=float(FINANCIAL_FEATURES["Income"]["max"]),
                        step=5.0, key="Income",
                        help=FINANCIAL_FEATURES["Income"]["description"])
        st.number_input("Credit Card Average Spending ($1000s)",
                        min_value=float(FINANCIAL_FEATURES["CCAvg"]["min"]),
                        max_value=float(FINANCIAL_FEATURES["CCAvg"]["max"]),
                        step=0.5, key="CCAvg",
                        help=FINANCIAL_FEATURES["CCAvg"]["description"])
        st.number_input("Mortgage Amount ($1000s)",
                        min_value=float(FINANCIAL_FEATURES["Mortgage"]["min"]),
                        max_value=float(FINANCIAL_FEATURES["Mortgage"]["max"]),
                        step=10.0, key="Mortgage",
                        help=FINANCIAL_FEATURES["Mortgage"]["description"])

    st.markdown("**Professional Experience**")
    st.slider("Years of Professional Experience",
              int(PROFESSIONAL_FEATURES["Experience"]["min"]),
              int(PROFESSIONAL_FEATURES["Experience"]["max"]),
              key="Experience",
              help=PROFESSIONAL_FEATURES["Experience"]["description"])

    st.markdown("**Banking Relationship**")
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        st.checkbox("Securities Account",
                    value=bool(st.session_state["Securities Account"]),
                    key="_sec",
                    help="Customer has an investment securities account")
    with bcol2:
        st.checkbox("CD Account",
                    value=bool(st.session_state["CD Account"]),
                    key="_cd",
                    help="Customer has a certificate of deposit")
    with bcol3:
        st.checkbox("Online Banking",
                    value=bool(st.session_state["Online"]),
                    key="_online",
                    help="Customer uses online banking")
    with bcol4:
        st.checkbox("Credit Card",
                    value=bool(st.session_state["CreditCard"]),
                    key="_cc",
                    help="Customer has a credit card with the bank")

    # Bridge checkbox bool -> int for FEATURE_ORDER lookup
    st.session_state["Securities Account"] = int(st.session_state["_sec"])
    st.session_state["CD Account"]         = int(st.session_state["_cd"])
    st.session_state["Online"]             = int(st.session_state["_online"])
    st.session_state["CreditCard"]         = int(st.session_state["_cc"])

    features_dict = {k: st.session_state[k] for k in FEATURE_ORDER}

    is_valid, errors = DataValidator.validate_all(features_dict)
    if not is_valid:
        for err in errors:
            st.warning(f"⚠ {err}")
        return

    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        predict_button = st.button("🚀 Make Prediction", use_container_width=True)
    with bcol2:
        if st.button("🔄 Reset Form", use_container_width=True):
            _seed_form_state()
            st.rerun()

    if predict_button:
        engine = PredictionEngine(model, FEATURE_ORDER)
        features_df = engine.prepare_features(features_dict)
        prediction, probability = engine.predict(features_df)

        st.markdown('<div class="section-header">🎯 Prediction Result</div>',
                    unsafe_allow_html=True)
        text, color = ResultFormatter.get_prediction_text(prediction)
        # Probability returned is P(loan accepted). Show it directly so the
        # user can see the actual model confidence in either direction.
        confidence = probability if prediction == 1 else 1 - probability

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%);
                        padding: 2rem; border-radius: 1rem;
                        border-left: 5px solid {color}; margin-bottom: 1.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {color};
                            margin-bottom: 0.5rem;">{text}</div>
                <div style="font-size: 1.1rem; color: #64748B;">
                    Model Confidence: <strong>
                    {ResultFormatter.format_probability(confidence)}
                    </strong>
                    &nbsp;·&nbsp; P(Accept) = {probability * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(Accept)", f"{probability*100:.1f}%")
        m2.metric("Income", f"${features_dict['Income']:.0f}K")
        m3.metric("CC Avg", f"${features_dict['CCAvg']:.2f}K")
        m4.metric("Experience", f"{int(features_dict['Experience'])} yrs")

        if SHOW_FEATURE_IMPORTANCE:
            st.markdown('<div class="section-header">📊 Feature Importance</div>',
                        unsafe_allow_html=True)
            importance = engine.get_feature_importance()
            if importance:
                df = (pd.DataFrame(list(importance.items()),
                                   columns=["Feature", "Importance"])
                      .sort_values("Importance", ascending=True))
                fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Viridis",
                             title="Feature Importance (from trained model)")
                fig.update_layout(height=420, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">⚠️ Risk Analysis</div>',
                    unsafe_allow_html=True)
        analysis = RiskAnalysis.analyze_risk_factors(features_dict)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Positive Factors**")
            if analysis["positive_factors"]:
                for f in analysis["positive_factors"]:
                    st.success(f"✓ {f}")
            else:
                st.info("No standout positive factors")
        with c2:
            st.markdown("**Risk Factors**")
            if analysis["risk_factors"]:
                for f in analysis["risk_factors"]:
                    st.warning(f"✗ {f}")
            else:
                st.info("No significant risk factors identified")

        st.markdown('<div class="section-header">📋 Customer Summary</div>',
                    unsafe_allow_html=True)
        summary = create_summary_stats(features_dict)
        cols = st.columns(3)
        for idx, (label, value) in enumerate(summary.items()):
            cols[idx % 3].info(f"**{label}**: {value}")


# ---- model info page -------------------------------------------------------

def render_model_info_page(model, metrics: dict) -> None:
    st.markdown('<div class="section-header">🤖 Model Information</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Specifications**")
        params = metrics.get("best_params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items()) or "n/a"
        st.info(
            f"""
            - **Algorithm**: Decision Tree Classifier
            - **Train / Test**: {metrics.get('n_train','?')} / {metrics.get('n_test','?')} records
            - **Features**: {len(FEATURE_ORDER)} input variables
            - **Best Params**: {params_str}
            - **CV (5-fold)**: {metrics.get('cv_mean',0)*100:.2f}% ± {metrics.get('cv_std',0)*100:.2f}%
            """
        )

    with col2:
        st.markdown("**Performance Metrics**")
        df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Score":  [f"{metrics.get(k, 0)*100:.2f}%"
                       for k in ("accuracy", "precision", "recall", "f1_score")],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">📈 Performance Visualization</div>',
                unsafe_allow_html=True)
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    fig = go.Figure(data=[
        go.Bar(
            x=["Accuracy", "Precision", "Recall", "F1-Score"],
            y=[metrics.get(k, 0) * 100 for k in metric_keys],
            marker=dict(color=["#00D084", "#4F8FFF", "#FFB800", "#FF5252"]),
            text=[f"{metrics.get(k, 0)*100:.2f}%" for k in metric_keys],
            textposition="auto",
        )
    ])
    fig.update_layout(title="Model Performance Metrics", yaxis_title="Score (%)",
                      height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    cm = metrics.get("confusion_matrix")
    if cm:
        st.markdown('<div class="section-header">🧮 Confusion Matrix (test set)</div>',
                    unsafe_allow_html=True)
        cm_df = pd.DataFrame(cm,
                             index=["Actual: No Loan", "Actual: Loan"],
                             columns=["Predicted: No Loan", "Predicted: Loan"])
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto",
                           color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-header">🎯 Feature Importance</div>',
                unsafe_allow_html=True)
    importance = (metrics.get("feature_importance")
                  or (PredictionEngine(model, FEATURE_ORDER).get_feature_importance()
                      if model is not None else {}))
    if importance:
        df_imp = (pd.DataFrame(list(importance.items()),
                               columns=["Feature", "Importance"])
                  .sort_values("Importance", ascending=True))
        fig2 = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Blues",
                      title="Feature Contribution to Prediction")
        fig2.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)


# ---- analytics page --------------------------------------------------------

def render_analytics_page(metrics: dict) -> None:
    st.markdown('<div class="section-header">📊 Analytics & Insights</div>',
                unsafe_allow_html=True)

    importance = metrics.get("feature_importance", {})
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    bullets = "\n".join(f"• **{name}** — {value*100:.1f}% importance"
                        for name, value in top) or "(metrics file missing)"
    st.info(f"**Top model drivers (from training run):**\n\n{bullets}")

    st.markdown('<div class="section-header">💡 Business Recommendations</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**High-Probability Segments**")
        st.success("""
        ✓ Income > $100K
        ✓ Credit Card Spending > $5K
        ✓ Experience > 15 years
        ✓ Has CD or Securities account
        """)
    with c2:
        st.markdown("**Marketing Strategy**")
        st.info("""
        • Focus on high-income customers
        • Target heavy credit-card users
        • Cross-sell to existing CD holders
        • Use online channel for digital natives
        """)


# ---- main ------------------------------------------------------------------

def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <p>🏦 Bank Personal Loan Prediction System</p>
            <p>Trained on the Bank Personal Loan Modelling dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    model, metrics = load_model_resource()
    render_header(metrics)
    page = render_sidebar(metrics)

    if page == "Prediction":
        render_prediction_page(model, metrics)
    elif page == "Model Information":
        render_model_info_page(model, metrics)
    elif page == "Analytics":
        render_analytics_page(metrics)

    render_footer()


if __name__ == "__main__":
    main()
