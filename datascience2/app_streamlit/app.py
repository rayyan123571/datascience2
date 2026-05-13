"""Bank Personal Loan Prediction — Streamlit application."""

import warnings
from pathlib import Path

import numpy as np
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

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Bank_Personal_Loan_Modelling.csv"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- styling --------------------------------------------------------------

st.markdown(
    """
<style>
    /* page */
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
    [data-testid="stHeader"] { background: transparent; }

    /* hero card */
    .hero {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        color: #fff;
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 30px -10px rgba(79, 70, 229, 0.35);
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }
    .hero p  { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 1rem; }
    .hero .pill {
        display: inline-block; padding: 0.3rem 0.75rem; border-radius: 999px;
        background: rgba(255,255,255,0.18); font-size: 0.82rem; margin-top: 0.6rem;
        backdrop-filter: blur(6px);
    }

    /* section header */
    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #0F172A;
        margin: 1.4rem 0 0.7rem 0;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .section-header::before {
        content: ""; width: 4px; height: 1.1rem; background: #4F46E5;
        border-radius: 2px; display: inline-block;
    }

    /* cards */
    .stat-card {
        background: #fff; border: 1px solid #E2E8F0; border-radius: 0.85rem;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }
    .stat-card .label { color: #64748B; font-size: 0.78rem; text-transform: uppercase;
                        letter-spacing: 0.05em; font-weight: 600; }
    .stat-card .value { color: #0F172A; font-size: 1.5rem; font-weight: 700; margin-top: 0.2rem; }
    .stat-card .delta { color: #16A34A; font-size: 0.82rem; margin-top: 0.1rem; }

    /* buttons */
    div.stButton > button {
        width: 100%; padding: 0.7rem 1.5rem; font-size: 0.95rem; font-weight: 600;
        border-radius: 0.6rem; border: none; cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        color: #fff !important;
        box-shadow: 0 4px 12px -3px rgba(79, 70, 229, 0.45);
    }
    div.stButton > button:hover { transform: translateY(-1px); filter: brightness(1.05); }
    div.stButton > button:active { transform: translateY(0); }
    div.stButton > button[kind="secondary"] {
        background: #fff; color: #4F46E5 !important;
        border: 1.5px solid #C7D2FE; box-shadow: none;
    }

    /* slider track */
    div[data-baseweb="slider"] > div > div > div { background: #4F46E5 !important; }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] {
        background: #F1F5F9; padding: 0.55rem 1.1rem; border-radius: 0.55rem;
        font-weight: 600; color: #475569; border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%) !important;
        color: #fff !important;
    }

    /* prediction result */
    .result-card {
        padding: 1.6rem 1.8rem; border-radius: 1rem; margin: 1rem 0;
        border-left: 6px solid var(--accent);
        background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 12%, white) 0%,
                                            color-mix(in srgb, var(--accent) 4%,  white) 100%);
    }
    .result-card .verdict { font-size: 1.45rem; font-weight: 700; color: var(--accent); }
    .result-card .meta    { font-size: 0.95rem; color: #475569; margin-top: 0.3rem; }

    /* sidebar */
    section[data-testid="stSidebar"] { background: #F8FAFC; border-right: 1px solid #E2E8F0; }
    section[data-testid="stSidebar"] .stMetric { background: #fff; padding: 0.6rem 0.7rem;
                                                  border-radius: 0.55rem; border: 1px solid #E2E8F0; }

    /* footer */
    .footer { text-align: center; padding: 1.5rem 0 0.5rem; color: #94A3B8;
              font-size: 0.82rem; border-top: 1px solid #E2E8F0; margin-top: 2rem; }

    /* hide default streamlit padding around metrics */
    [data-testid="stMetricValue"] { font-size: 1.35rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ----- resource loaders -----------------------------------------------------

@st.cache_resource
def load_model_resource():
    loader = ModelLoader(MODEL_PATH, METRICS_PATH)
    if loader.load():
        return loader.get_model(), loader.get_metrics()
    return None, {}


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


# ----- helpers --------------------------------------------------------------

def _seed_form_state(profile: dict | None = None) -> None:
    defaults = {
        "Age": 45, "Experience": 20, "Income": 75.0, "Family": 2,
        "CCAvg": 2.0, "Education": 1, "Mortgage": 0.0,
        "Securities Account": 0, "CD Account": 0, "Online": 0, "CreditCard": 0,
    }
    source = profile or defaults
    for k, v in source.items():
        st.session_state[k] = v
    # Bridge keys for checkboxes
    st.session_state["_sec"]    = bool(source.get("Securities Account", 0))
    st.session_state["_cd"]     = bool(source.get("CD Account", 0))
    st.session_state["_online"] = bool(source.get("Online", 0))
    st.session_state["_cc"]     = bool(source.get("CreditCard", 0))


def stat_card(label: str, value: str, delta: str = "") -> str:
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (f'<div class="stat-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{delta_html}</div>')


# ----- header / sidebar -----------------------------------------------------

def render_hero(metrics: dict) -> None:
    winner = metrics.get("winner", "Model")
    acc = metrics.get("accuracy", 0.0) * 100
    auc = metrics.get("roc_auc", 0.0) * 100
    st.markdown(
        f"""
        <div class="hero">
            <h1>🏦 {APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
            <span class="pill">⚡ Best Model: {winner}</span>
            <span class="pill">🎯 Accuracy: {acc:.2f}%</span>
            <span class="pill">📈 ROC-AUC: {auc:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metrics: dict) -> str:
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "View",
            ["🎯 Prediction", "📊 Model Comparison", "🤖 Model Details",
             "📈 Data Analytics", "💡 Insights"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 👤 Sample Profiles")
        profile_names = list(SAMPLE_PROFILES.keys())
        selected = st.selectbox("Load a sample:", ["—"] + profile_names,
                                label_visibility="collapsed")
        if selected != "—":
            if st.button("📥 Apply sample", use_container_width=True):
                _seed_form_state(SAMPLE_PROFILES[selected])
                st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Best Model Scorecard")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy",  f"{metrics.get('accuracy',  0)*100:.1f}%")
        c2.metric("Precision", f"{metrics.get('precision', 0)*100:.1f}%")
        c1.metric("Recall",    f"{metrics.get('recall',    0)*100:.1f}%")
        c2.metric("F1",        f"{metrics.get('f1_score',  0)*100:.1f}%")
        st.metric("ROC-AUC",   f"{metrics.get('roc_auc',   0)*100:.2f}%")

    return page


# ----- prediction page ------------------------------------------------------

def render_prediction_page(model, metrics: dict) -> None:
    if model is None:
        st.error("❌ Model file not found. Run `python train_model.py` first.")
        return

    for k in FEATURE_ORDER:
        if k not in st.session_state:
            _seed_form_state()
            break

    st.markdown('<div class="section-header">👥 Customer Profile</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🧑 Demographics", "💰 Financials", "🏦 Banking"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider("Age (years)",
                      int(DEMOGRAPHIC_FEATURES["Age"]["min"]),
                      int(DEMOGRAPHIC_FEATURES["Age"]["max"]),
                      key="Age",
                      help=DEMOGRAPHIC_FEATURES["Age"]["description"])
        with col2:
            st.slider("Family Size",
                      int(DEMOGRAPHIC_FEATURES["Family"]["min"]),
                      int(DEMOGRAPHIC_FEATURES["Family"]["max"]),
                      key="Family",
                      help=DEMOGRAPHIC_FEATURES["Family"]["description"])
        with col3:
            st.selectbox("Education Level", options=[1, 2, 3],
                         format_func=lambda x: EDUCATION_MAPPING.get(x, "Unknown"),
                         key="Education",
                         help=DEMOGRAPHIC_FEATURES["Education"]["description"])
        st.slider("Years of Professional Experience",
                  int(PROFESSIONAL_FEATURES["Experience"]["min"]),
                  int(PROFESSIONAL_FEATURES["Experience"]["max"]),
                  key="Experience",
                  help=PROFESSIONAL_FEATURES["Experience"]["description"])

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider("Annual Income ($1000s)",
                      float(FINANCIAL_FEATURES["Income"]["min"]),
                      float(FINANCIAL_FEATURES["Income"]["max"]),
                      step=1.0, key="Income",
                      help=FINANCIAL_FEATURES["Income"]["description"])
        with col2:
            st.slider("Credit Card Avg Spend ($1000s)",
                      float(FINANCIAL_FEATURES["CCAvg"]["min"]),
                      float(FINANCIAL_FEATURES["CCAvg"]["max"]),
                      step=0.1, key="CCAvg",
                      help=FINANCIAL_FEATURES["CCAvg"]["description"])
        with col3:
            st.slider("Mortgage Amount ($1000s)",
                      float(FINANCIAL_FEATURES["Mortgage"]["min"]),
                      float(FINANCIAL_FEATURES["Mortgage"]["max"]),
                      step=5.0, key="Mortgage",
                      help=FINANCIAL_FEATURES["Mortgage"]["description"])

    with tab3:
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            st.checkbox("💼 Securities Account", key="_sec",
                        help="Customer has an investment securities account")
        with bcol2:
            st.checkbox("🏛 CD Account", key="_cd",
                        help="Customer has a certificate of deposit")
        with bcol3:
            st.checkbox("🌐 Online Banking", key="_online",
                        help="Customer uses online banking")
        with bcol4:
            st.checkbox("💳 Credit Card", key="_cc",
                        help="Customer has a credit card with the bank")

    # Bridge checkbox bool -> int
    st.session_state["Securities Account"] = int(st.session_state.get("_sec", False))
    st.session_state["CD Account"]         = int(st.session_state.get("_cd",  False))
    st.session_state["Online"]             = int(st.session_state.get("_online", False))
    st.session_state["CreditCard"]         = int(st.session_state.get("_cc",  False))

    features_dict = {k: st.session_state[k] for k in FEATURE_ORDER}

    is_valid, errors = DataValidator.validate_all(features_dict)
    if not is_valid:
        for err in errors:
            st.warning(f"⚠ {err}")
        return

    st.markdown("")
    bcol1, bcol2, _ = st.columns([1, 1, 3])
    with bcol1:
        predict_button = st.button("🚀 Predict Loan Acceptance", use_container_width=True)
    with bcol2:
        if st.button("🔄 Reset Form", use_container_width=True, type="secondary"):
            _seed_form_state()
            st.rerun()

    if predict_button:
        engine = PredictionEngine(model, FEATURE_ORDER)
        features_df = engine.prepare_features(features_dict)
        prediction, probability = engine.predict(features_df)

        st.markdown('<div class="section-header">🎯 Prediction Result</div>',
                    unsafe_allow_html=True)
        text, color = ResultFormatter.get_prediction_text(prediction)
        confidence = probability if prediction == 1 else 1 - probability

        emoji = "✅" if prediction == 1 else "🚫"
        st.markdown(
            f"""
            <div class="result-card" style="--accent: {color};">
                <div class="verdict">{emoji} {text}</div>
                <div class="meta">
                    Model Confidence: <strong>{ResultFormatter.format_probability(confidence)}</strong>
                    &nbsp;·&nbsp; P(Accept) = <strong>{probability * 100:.2f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability gauge
        col_g, col_m = st.columns([2, 3])
        with col_g:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 36}},
                title={"text": "P(Loan Accepted)", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color, "thickness": 0.35},
                    "bgcolor": "#F1F5F9",
                    "steps": [
                        {"range": [0, 30],  "color": "#FEE2E2"},
                        {"range": [30, 70], "color": "#FEF3C7"},
                        {"range": [70, 100],"color": "#DCFCE7"},
                    ],
                    "threshold": {"line": {"color": "#0F172A", "width": 3},
                                  "thickness": 0.8, "value": 50},
                },
            ))
            gauge.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        with col_m:
            st.markdown('<div class="section-header">📋 Quick Snapshot</div>',
                        unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m1.markdown(stat_card("Income", f"${features_dict['Income']:.0f}K"),
                        unsafe_allow_html=True)
            m2.markdown(stat_card("CC Average", f"${features_dict['CCAvg']:.2f}K"),
                        unsafe_allow_html=True)
            m3.markdown(stat_card("Experience", f"{int(features_dict['Experience'])} yrs"),
                        unsafe_allow_html=True)
            m4.markdown(stat_card("Family Size", f"{int(features_dict['Family'])}"),
                        unsafe_allow_html=True)

        if SHOW_FEATURE_IMPORTANCE:
            st.markdown('<div class="section-header">📊 Feature Importance (Model)</div>',
                        unsafe_allow_html=True)
            importance = engine.get_feature_importance() or metrics.get("feature_importance", {})
            if importance:
                df = (pd.DataFrame(list(importance.items()),
                                   columns=["Feature", "Importance"])
                      .sort_values("Importance", ascending=True))
                fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Tealrose",
                             text=df["Importance"].apply(lambda v: f"{v*100:.1f}%"))
                fig.update_traces(textposition="outside")
                fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False,
                                  margin=dict(l=10, r=20, t=20, b=10),
                                  xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">⚖️ Risk Analysis</div>',
                    unsafe_allow_html=True)
        analysis = RiskAnalysis.analyze_risk_factors(features_dict)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Positive Factors**")
            if analysis["positive_factors"]:
                for f in analysis["positive_factors"]:
                    st.success(f"✓ {f}")
            else:
                st.info("No standout positive factors")
        with c2:
            st.markdown("**⚠️ Risk Factors**")
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
            with cols[idx % 3]:
                st.markdown(stat_card(label, value), unsafe_allow_html=True)


# ----- model comparison page ------------------------------------------------

def render_comparison_page(metrics: dict) -> None:
    st.markdown('<div class="section-header">🏆 Model Comparison Dashboard</div>',
                unsafe_allow_html=True)

    all_models = metrics.get("all_models", {})
    if not all_models:
        st.warning("No multi-model comparison data found. Re-run `python train_model.py`.")
        return

    winner = metrics.get("winner")
    st.info(
        f"All candidate models were trained with stratified 5-fold cross-validation "
        f"and tuned via GridSearch / RandomizedSearch on F1-score. "
        f"**{winner}** was selected as the production model based on the highest F1 on the test set."
    )

    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    rows = []
    for name, info in all_models.items():
        row = {"Model": name}
        for k in metric_keys:
            row[k] = info.get(k, 0.0)
        row["cv_mean"] = info.get("cv_mean", 0.0)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)

    # Top scorecard for each model
    st.markdown("##### Test-set scores at a glance")
    cols = st.columns(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        is_winner = r["Model"] == winner
        badge = "🥇 " if is_winner else ""
        accent = "#10B981" if is_winner else "#4F46E5"
        with cols[i]:
            st.markdown(
                f"""
                <div class="stat-card" style="border-top: 4px solid {accent};">
                    <div class="label">{badge}{r['Model']}</div>
                    <div class="value">{r['f1_score']*100:.2f}%</div>
                    <div class="delta">F1 Score · AUC {r['roc_auc']*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Grouped bar chart for all metrics
    st.markdown('<div class="section-header">📊 All Metrics Side-by-Side</div>',
                unsafe_allow_html=True)
    melted = df.melt(id_vars="Model",
                     value_vars=metric_keys,
                     var_name="Metric", value_name="Score")
    melted["Score"] = melted["Score"] * 100
    fig = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                 text=melted["Score"].apply(lambda v: f"{v:.1f}"),
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(height=480, yaxis_title="Score (%)", xaxis_title="",
                      yaxis=dict(range=[0, 105]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown('<div class="section-header">🕸️ Performance Radar</div>',
                unsafe_allow_html=True)
    radar = go.Figure()
    palette = px.colors.qualitative.Bold
    for i, (_, r) in enumerate(df.iterrows()):
        radar.add_trace(go.Scatterpolar(
            r=[r[k] * 100 for k in metric_keys] + [r[metric_keys[0]] * 100],
            theta=metric_keys + [metric_keys[0]],
            fill="toself",
            name=r["Model"],
            line=dict(color=palette[i % len(palette)]),
            opacity=0.55,
        ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[60, 100])),
        height=520, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )
    st.plotly_chart(radar, use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="section-header">🧮 Confusion Matrices</div>',
                unsafe_allow_html=True)
    cm_cols = st.columns(len(all_models))
    for i, (name, info) in enumerate(all_models.items()):
        cm = info.get("confusion_matrix")
        if not cm:
            continue
        with cm_cols[i]:
            cm_df = pd.DataFrame(cm,
                                 index=["No Loan", "Loan"],
                                 columns=["Pred: No", "Pred: Yes"])
            fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto",
                               color_continuous_scale="Blues",
                               title=name)
            fig_cm.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

    # Full metric table
    st.markdown('<div class="section-header">📋 Full Metrics Table</div>',
                unsafe_allow_html=True)
    table_df = df.copy()
    for c in metric_keys + ["cv_mean"]:
        table_df[c] = (table_df[c] * 100).round(2).astype(str) + "%"
    table_df.rename(columns={"f1_score": "F1", "roc_auc": "ROC-AUC",
                              "cv_mean": "CV Mean (Acc)"}, inplace=True)
    table_df.columns = [c.title() if c != "Model" else c for c in table_df.columns]
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Static plot from training run, if available
    plots_dir = Path(__file__).resolve().parent.parent / "reports" / "plots"
    roc_img = plots_dir / "roc_curves.png"
    if roc_img.exists():
        st.markdown('<div class="section-header">📈 ROC Curves (from training run)</div>',
                    unsafe_allow_html=True)
        st.image(str(roc_img), use_container_width=True)


# ----- model details page ---------------------------------------------------

def render_model_info_page(model, metrics: dict) -> None:
    winner = metrics.get("winner", "Model")
    st.markdown(f'<div class="section-header">🤖 {winner} — Production Model</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Specifications**")
        params = metrics.get("best_params", {})
        params_str = "<br>".join(f"&nbsp;&nbsp;• <code>{k}</code> = <strong>{v}</strong>"
                                  for k, v in params.items()) or "n/a"
        st.markdown(
            f"""
            <div class="stat-card">
                • <strong>Algorithm</strong>: {winner}<br>
                • <strong>Train / Test split</strong>: {metrics.get('n_train','?')} / {metrics.get('n_test','?')} records<br>
                • <strong>Features</strong>: {len(FEATURE_ORDER)} input variables<br>
                • <strong>CV (5-fold accuracy)</strong>: {metrics.get('cv_mean',0)*100:.2f}% ± {metrics.get('cv_std',0)*100:.2f}%<br>
                • <strong>Tuned hyperparameters</strong>:<br>{params_str}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("**Performance Metrics**")
        df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Score":  [f"{metrics.get(k, 0)*100:.2f}%"
                       for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc")],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=[metrics.get(k, 0) * 100 for k in metric_keys],
            marker=dict(color=["#10B981", "#4F46E5", "#06B6D4", "#F59E0B", "#EC4899"]),
            text=[f"{metrics.get(k, 0)*100:.2f}%" for k in metric_keys],
            textposition="outside",
        )
    ])
    fig.update_layout(title="Performance Metrics", yaxis_title="Score (%)",
                      yaxis=dict(range=[0, 105]),
                      height=420, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10))
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
        fig_cm.update_layout(height=380)
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
                      color="Importance", color_continuous_scale="Viridis",
                      text=df_imp["Importance"].apply(lambda v: f"{v*100:.2f}%"))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=460, showlegend=False, coloraxis_showscale=False,
                           margin=dict(l=10, r=30, t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ----- data analytics page --------------------------------------------------

def render_data_analytics_page() -> None:
    st.markdown('<div class="section-header">📈 Dataset Explorer</div>',
                unsafe_allow_html=True)
    df = load_dataset()
    if df is None:
        st.warning(f"Dataset not found at {DATA_PATH}")
        return

    # Top KPIs
    pos_rate = df["Personal Loan"].mean() * 100
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(stat_card("Records", f"{len(df):,}"), unsafe_allow_html=True)
    with k2:
        st.markdown(stat_card("Features", f"{df.shape[1]-1}"), unsafe_allow_html=True)
    with k3:
        st.markdown(stat_card("Loan Acceptance", f"{pos_rate:.2f}%"),
                    unsafe_allow_html=True)
    with k4:
        st.markdown(stat_card("Avg Income", f"${df['Income'].mean():.0f}K"),
                    unsafe_allow_html=True)

    st.markdown('<div class="section-header">🎯 Target Distribution</div>',
                unsafe_allow_html=True)
    target_counts = df["Personal Loan"].value_counts().rename(
        {0: "No Loan", 1: "Loan Accepted"})
    fig = px.pie(values=target_counts.values, names=target_counts.index,
                 hole=0.55, color_discrete_sequence=["#94A3B8", "#10B981"])
    fig.update_traces(textinfo="percent+label", textfont_size=14)
    fig.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🔥 Feature Correlation Heatmap</div>',
                unsafe_allow_html=True)
    corr = df.drop(columns=["ID", "ZIP Code"], errors="ignore").corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">📊 Feature Distributions by Loan Status</div>',
                unsafe_allow_html=True)
    feat = st.selectbox("Pick a feature:",
                        ["Income", "CCAvg", "Mortgage", "Age", "Experience", "Family"])
    fig = px.histogram(df, x=feat, color="Personal Loan",
                       barmode="overlay", nbins=40,
                       color_discrete_map={0: "#94A3B8", 1: "#10B981"})
    fig.update_layout(height=380, legend_title="Loan",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df, x="Personal Loan", y=feat,
                  color="Personal Loan",
                  color_discrete_map={0: "#94A3B8", 1: "#10B981"})
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                       xaxis=dict(tickmode="array", tickvals=[0, 1],
                                  ticktext=["No Loan", "Loan Accepted"]))
    st.plotly_chart(fig2, use_container_width=True)


# ----- insights page --------------------------------------------------------

def render_insights_page(metrics: dict) -> None:
    st.markdown('<div class="section-header">💡 Insights & Business Recommendations</div>',
                unsafe_allow_html=True)

    importance = metrics.get("feature_importance", {})
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("**🎯 Top Model Drivers**")
        for name, value in top:
            st.markdown(stat_card(name, f"{value*100:.1f}%", "importance"),
                        unsafe_allow_html=True)
            st.write("")

    with col2:
        st.markdown("**📈 Key Findings**")
        st.success("✅ **Income** is the single strongest predictor — high-income "
                   "customers are far more likely to accept loan offers.")
        st.success("✅ **Education** and **CCAvg** are strong secondary drivers — "
                   "marketing should target educated, active credit-card users.")
        st.info("ℹ️ **Online Banking** and **Securities Account** add small but "
                "consistent uplift.")
        st.warning("⚠️ The dataset is imbalanced (~9.6% positives) — F1 and "
                   "ROC-AUC are more informative than raw accuracy.")

    st.markdown('<div class="section-header">🎯 Customer Segments</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🟢 High-Probability Segments**")
        st.success(
            "• Income > $100K\n\n"
            "• Credit-Card Spend > $5K/month\n\n"
            "• 15+ years of experience\n\n"
            "• Holds CD or Securities account\n\n"
            "• Graduate or Advanced degree"
        )
    with c2:
        st.markdown("**🔵 Recommended Marketing Strategy**")
        st.info(
            "• Prioritise high-income, educated segments\n\n"
            "• Cross-sell to existing CD / Securities holders\n\n"
            "• Use online channel for digital-native customers\n\n"
            "• A/B test offers on heavy credit-card users\n\n"
            "• De-prioritise customers with <5 yrs experience"
        )


# ----- main -----------------------------------------------------------------

def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            🏦 Bank Personal Loan Prediction · Trained on the
            <em>Bank Personal Loan Modelling</em> dataset ·
            Logistic Regression · Decision Tree · Random Forest · XGBoost · KNN
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    model, metrics = load_model_resource()
    render_hero(metrics)
    page = render_sidebar(metrics)

    if "Prediction" in page:
        render_prediction_page(model, metrics)
    elif "Comparison" in page:
        render_comparison_page(metrics)
    elif "Details" in page:
        render_model_info_page(model, metrics)
    elif "Analytics" in page:
        render_data_analytics_page()
    elif "Insights" in page:
        render_insights_page(metrics)

    render_footer()


if __name__ == "__main__":
    main()
