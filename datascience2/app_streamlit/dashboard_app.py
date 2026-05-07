import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from config import COLORS, EDUCATION_MAPPING
except Exception:
    # Fallbacks if config not available
    COLORS = {
        "accept": "#00D084",
        "reject": "#FF5252",
        "neutral": "#4F8FFF",
        "background": "#F5F7FA",
        "text_primary": "#1E293B",
    }
    EDUCATION_MAPPING = {1: "Undergraduate", 2: "Graduate", 3: "Advanced"}


def load_default_data() -> Optional[pd.DataFrame]:
    # Try common locations relative to this file
    candidates = [
        os.path.join(os.path.dirname(__file__), "data", "bank_personal_loan.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "bank_personal_loan.csv"),
        os.path.join(os.path.dirname(__file__), "data", "bank.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "bank.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names
    df = df.copy()
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Attempt to find loan target column
    loan_col = None
    for c in df.columns:
        if "loan" in c.lower() or "personal" in c.lower():
            loan_col = c
            break
    if loan_col and loan_col != "Loan":
        df.rename(columns={loan_col: "Loan"}, inplace=True)

    # Ensure expected numeric columns exist
    expected = ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage", "Loan"]
    present = [c for c in expected if c in df.columns]

    # If education is numeric-coded, keep as-is; if categorical, try to map
    if "Education" in df.columns and df["Education"].dtype == object:
        # try to map textual education to codes
        mapping = {v.lower(): k for k, v in EDUCATION_MAPPING.items()}
        df["Education"] = df["Education"].apply(lambda x: next((k for k, v in EDUCATION_MAPPING.items() if str(v).lower() == str(x).lower()), x))

    return df


def apply_filters(df: pd.DataFrame, education: list, family: list, income_range: tuple):
    d = df.copy()
    if education and "All" not in education:
        # education may be numeric codes or strings
        codes = []
        for e in education:
            if isinstance(e, int):
                codes.append(e)
            else:
                # map from label to code
                for k, v in EDUCATION_MAPPING.items():
                    if v == e:
                        codes.append(k)
        if codes:
            d = d[d["Education"].isin(codes)]
    if family and "All" not in family:
        d = d[d["Family"].isin(family)]
    if income_range is not None:
        d = d[(d["Income"] >= income_range[0]) & (d["Income"] <= income_range[1])]
    return d


def overview_section(df: pd.DataFrame):
    total = len(df)
    if total == 0:
        st.info("No records to display.")
        return
    accepted = int(df["Loan"].astype(int).sum())
    rate = accepted / total

    c1, c2, c3 = st.columns([2, 3, 2])
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Loan Acceptance Rate", f"{rate:.2%}", delta=None)
    c3.metric("Accepted (count)", f"{accepted:,}")

    # small pie
    fig = px.pie(
        df,
        names=df["Loan"].map({0: "Rejected", 1: "Accepted"}),
        color_discrete_map={"Accepted": COLORS["accept"], "Rejected": COLORS["reject"]},
        title="Loan Acceptance Breakdown",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Customers with higher income show a significantly higher likelihood of accepting loans.")


def income_distribution_section(df: pd.DataFrame):
    st.subheader("Income & Age Distribution")
    fig = px.histogram(df, x="Income", nbins=50, color=df["Loan"].map({0: "Rejected", 1: "Accepted"}),
                       color_discrete_map={"Accepted": COLORS["accept"], "Rejected": COLORS["reject"]},
                       labels={"Income": "Income ($1000s)"}, title="Income Distribution by Loan Outcome")
    fig.update_layout(barmode="overlay", bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

    # Age histogram
    fig2 = px.histogram(df, x="Age", nbins=30, color=df["Loan"].map({0: "Rejected", 1: "Accepted"}),
                        color_discrete_map={"Accepted": COLORS["accept"], "Rejected": COLORS["reject"]},
                        labels={"Age": "Age (years)"}, title="Age Distribution by Loan Outcome")
    st.plotly_chart(fig2, use_container_width=True)

    # Boxplots for outliers
    cols = ["Income", "Age"]
    fig_box = go.Figure()
    for c in cols:
        fig_box.add_trace(go.Box(x=df[c], name=c, marker_color=COLORS["neutral"]))
    fig_box.update_layout(title_text="Boxplots (Income & Age)")
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("**Insight:** Boxplots reveal income has wider spread and outliers indicating a small affluent segment.")


def acceptance_vs_features_section(df: pd.DataFrame):
    st.subheader("Loan Acceptance vs. Categorical Features")
    # Education
    edu_map = df["Education"].map(EDUCATION_MAPPING) if "Education" in df.columns else df["Education"]
    edu_df = df.copy()
    edu_df["EducationLabel"] = edu_map
    edu_group = edu_df.groupby("EducationLabel")["Loan"].agg(["mean", "count"]).reset_index()
    fig = px.bar(edu_group, x="EducationLabel", y="mean", labels={"mean": "Acceptance Rate", "EducationLabel": "Education"},
                 title="Acceptance Rate by Education", color_discrete_sequence=[COLORS["neutral"]])
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Customers with higher education levels tend to accept offers at higher rates.")

    # Family
    fam_group = df.groupby("Family")["Loan"].agg(["mean", "count"]).reset_index()
    fig2 = px.bar(fam_group, x="Family", y="mean", labels={"mean": "Acceptance Rate", "Family": "Family Size"},
                  title="Acceptance Rate by Family Size", color_discrete_sequence=[COLORS["neutral"]])
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Insight:** Larger family sizes show modestly different acceptance behavior — useful for targeted offers.")


def correlation_section(df: pd.DataFrame):
    st.subheader("Correlation Heatmap")
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** Income and CCAvg are positively correlated with loan acceptance; monitor multicollinearity for modeling.")


def scatter_section(df: pd.DataFrame):
    st.subheader("Income vs. CCAvg")
    fig = px.scatter(df, x="Income", y="CCAvg", color=df["Loan"].map({0: "Rejected", 1: "Accepted"}),
                     color_discrete_map={"Accepted": COLORS["accept"], "Rejected": COLORS["reject"]},
                     size="Mortgage" if "Mortgage" in df.columns else None,
                     hover_data=["Age", "Family", "Education"], title="Income vs. CCAvg by Loan Outcome")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Insight:** High CCAvg combined with high income is a strong signal for acceptance.")


def main():
    st.set_page_config(page_title="Loan BI Dashboard", layout="wide")
    st.title("Bank Personal Loan - BI Dashboard")

    # Sidebar controls
    st.sidebar.header("Data & Filters")
    uploaded = st.sidebar.file_uploader("Upload CSV file (optional)", type=["csv"]) 

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded file: {e}")

    if df is None:
        df = load_default_data()
        if df is None:
            st.sidebar.info("No default dataset found. Please upload a CSV file.")
            st.stop()

    df = standardize_df(df)

    # Ensure necessary columns exist
    required = ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage", "Loan"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.write(df.columns.tolist())
        st.stop()

    # Sidebar filters
    edu_labels = ["All"] + [EDUCATION_MAPPING.get(int(x), str(x)) for x in sorted(df["Education"].unique())]
    sel_edu = st.sidebar.multiselect("Education", options=edu_labels, default=["All"]) 

    fam_opts = ["All"] + sorted(df["Family"].dropna().unique().tolist())
    sel_fam = st.sidebar.multiselect("Family Size", options=fam_opts, default=["All"]) 

    min_income, max_income = int(df["Income"].min()), int(df["Income"].max())
    sel_income = st.sidebar.slider("Income Range ($1000s)", min_income, max_income, (min_income, max_income))

    show_raw = st.sidebar.checkbox("Show raw data", value=False)

    # map selected education labels back to codes if needed
    if sel_edu and "All" in sel_edu:
        sel_edu = ["All"]

    # convert family selection to numeric if needed
    if sel_fam and "All" not in sel_fam:
        sel_fam = [int(x) for x in sel_fam]

    filtered = apply_filters(df, sel_edu, sel_fam, sel_income)

    # Layout sections
    st.markdown("## Overview")
    overview_section(filtered)

    st.markdown("---")
    st.markdown("## Distributions")
    income_distribution_section(filtered)

    st.markdown("---")
    st.markdown("## Acceptance vs Features")
    acceptance_vs_features_section(filtered)

    st.markdown("---")
    correlation_section(filtered)

    st.markdown("---")
    scatter_section(filtered)

    if show_raw:
        st.markdown("---")
        st.subheader("Raw Data Sample")
        st.dataframe(filtered.head(200))


if __name__ == "__main__":
    main()
